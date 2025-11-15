# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.

import json
import os
import random
import subprocess
import time
from typing import Optional

import numpy as np
import torch

from dataset import SparsePatchDataset, SparsePatchDatasetFromHDF5

desc_dim_dict = {
    "spnn": 256,
    "splg": 256,
    "dedode": 256,
    "dedodev2": 256,
    "sift": 128,
    "r2d2": 128,
    "aliked": 128,
    "disk": 128,
    "xfeat": 64,
    "xfeat-star": 64,
    "xfeat_orig": 64,
}


def set_seeds(seed: int = 42):
    """
    Set Python random seeding and PyTorch seeds.

    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_progress(
    batch_number: int,
    batch_size: int,
    number_of_batches: int,
    processed_data_in_percent: int,
    start_time: float,
    additional_status_info: str = "",
) -> int:
    if batch_size is None:
        batch_size = 1
    number_of_samples = number_of_batches * batch_size
    sample_number = batch_number * batch_size
    currently_processed_data_in_percent: int = int(sample_number / (number_of_samples / 100.0))
    if currently_processed_data_in_percent > processed_data_in_percent:
        processed_data_in_percent = currently_processed_data_in_percent
        processing_rate = sample_number / (time.time() - start_time)

        update_string = "{} - Processed {}% of data ({}/~{}) in {:.2f}min, at {:.2f} images/s".format(
            time.strftime("%H:%M:%S", time.localtime()),
            processed_data_in_percent,
            sample_number,
            number_of_samples,
            (time.time() - start_time) / 60.0,
            processing_rate,
        )
        update_string += additional_status_info
        print(
            update_string,
            end="\r",
            flush=True,
        )
    if processed_data_in_percent == 100:
        print()
    return processed_data_in_percent


def get_git_hash() -> str:
    git_hash = ""
    if subprocess.call(["git", "branch"], stderr=subprocess.STDOUT, stdout=open(os.devnull, "w")) != 0:
        git_hash = "this is not a git repository"
    else:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    return git_hash


def create_log_dir(result_path, opt, training: bool):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    with open(os.path.join(result_path, "config.json"), "w") as f:
        json.dump(opt.__dict__, f, indent=2)

    # path for saving traning logs
    opt.log_path = os.path.join(result_path, "logs")
    if training:
        opt.model_path = os.path.join(result_path, "models")

    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)
    if training:
        if not os.path.isdir(opt.model_path):
            os.makedirs(opt.model_path)


# Below function is borrowed from
def normalize_pts(pts, im_size):
    """Normalize image coordinate using the image size.

    Pre-processing of correspondences before passing them to the network to be
    independent of image resolution.
    Re-scales points such that max image dimension goes from -0.5 to 0.5.
    In-place operation.

    Keyword arguments:
    pts -- 3-dim array conainting x and y coordinates in the last dimension, first dimension should have size 1.
    im_size -- image height and width
    """

    pts[:, 0] -= float(im_size[1]) / 2
    pts[:, 1] -= float(im_size[0]) / 2
    pts /= float(max(im_size[:2]))


# The following function is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


# The following function is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


# The following function is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


# The following function is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def evaluate(
    model,
    device,
    processing_mode,
    adjust_only_second_keypoint,
    correspondence,
    patch1,
    patch2,
    cell_size=1,
    scorepatch1=None,
    scorepatch2=None,
    descriptor1=None,
    descriptor2=None,
    K1s=None,
    K2s=None,
    min_match_score_threshold: float = 0.0,  # Set to 0.0 to deactivate match score thresholding
    gt_pts2=None,
    gt_pts2_mask=None,
):
    if scorepatch1 is not None:
        scorepatch1 = scorepatch1
    if scorepatch2 is not None:
        scorepatch2 = scorepatch2

    match_score = None
    eval_result = dict()

    if processing_mode == "independent_processing" and not adjust_only_second_keypoint:
        # keypt2subpx version
        meanft = (descriptor1 + descriptor2) / 2.0

        eval_result_1 = model(patch1, scorepatch1, meanft)  # patch centered coordinate
        similarity1 = eval_result_1["similarity"]
        descr1 = eval_result_1["descr"]
        delta1 = eval_result_1["delta"].unsqueeze(-1)
        delta1 = delta1 * cell_size
        # undistort points (homogeneous coords) when intrinsics provided
        delta1_undistorted = K1s[:, :2, :2].unsqueeze(1).inverse() @ delta1 if K1s is not None else delta1
        delta1_undistorted = delta1_undistorted.squeeze(-1)

        eval_result_2 = model(patch2, scorepatch2, meanft)
        similarity2 = eval_result_2["similarity"]
        descr2 = eval_result_2["descr"]
        delta2 = eval_result_2["delta"].unsqueeze(-1)
        delta2 = delta2 * cell_size
        # undistort points (homogeneous coords) when intrinsics provided
        delta2_undistorted = K2s[:, :2, :2].unsqueeze(1).inverse() @ delta2 if K2s is not None else delta2
        delta2_undistorted = delta2_undistorted.squeeze(-1)

        updated_correspondence = correspondence  # B x N x 4
        updated_correspondence = updated_correspondence + torch.cat([delta1_undistorted, delta2_undistorted], 2)
    elif processing_mode == "independent_processing" and adjust_only_second_keypoint:
        # refine 2 from 1 version
        query_descr = descriptor1

        eval_result_2 = model(patch2, scorepatch2, query_descr)
        similarity2 = eval_result_2["similarity"]
        descr2 = eval_result_2["descr"]
        delta2 = eval_result_2["delta"].unsqueeze(-1)
        delta2 = delta2 * cell_size
        similarity1 = torch.zeros(similarity2.shape, device=device)
        delta1 = torch.zeros(delta2.shape, device=device)
        descr1 = query_descr.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, descr2.shape[-2], descr2.shape[-1])
        # undistort points (homogeneous coords) when intrinsics provided
        delta2_undistorted = K2s[:, :2, :2].unsqueeze(1).inverse() @ delta2 if K2s is not None else delta2
        delta2_undistorted = delta2_undistorted.squeeze(-1)

        updated_correspondence = correspondence  # B x N x 4
        updated_correspondence[:, :, 2:4] = updated_correspondence[:, :, 2:4] + delta2_undistorted
    elif processing_mode == "joint_processing" and not adjust_only_second_keypoint:
        eval_result = model(
            patch1,
            patch2,
            scorepatch1,
            scorepatch2,
            descriptor1,
            descriptor2,
        )
        similarity1 = eval_result["similarity1"]
        descr1 = eval_result["descr1"]
        delta1 = eval_result["delta1"].unsqueeze(-1)
        delta1 = delta1 * cell_size
        # undistort points (homogeneous coords) when intrinsics provided
        delta1_undistorted = K1s[:, :2, :2].unsqueeze(1).inverse() @ delta1 if K1s is not None else delta1
        delta1_undistorted = delta1_undistorted.squeeze(-1)

        similarity2 = eval_result["similarity2"]
        descr2 = eval_result["descr2"]
        delta2 = eval_result["delta2"].unsqueeze(-1)
        delta2 = delta2 * cell_size
        # undistort points (homogeneous coords) when intrinsics provided
        delta2_undistorted = K2s[:, :2, :2].unsqueeze(1).inverse() @ delta2 if K2s is not None else delta2
        delta2_undistorted = delta2_undistorted.squeeze(-1)

        match_score = eval_result["match_score"]

        updated_correspondence = correspondence + torch.cat([delta1_undistorted, delta2_undistorted], 2)  # B x N x 4
    elif processing_mode == "joint_processing" and adjust_only_second_keypoint:
        eval_result = model(
            patch1,
            patch2,
            scorepatch1,
            scorepatch2,
            descriptor1,
            descriptor2,
        )
        similarity1 = eval_result["similarity1"]
        descr1 = eval_result["descr1"]
        delta1 = eval_result["delta1"].unsqueeze(-1)

        similarity2 = eval_result["similarity2"]
        descr2 = eval_result["descr2"]
        delta2 = eval_result["delta2"].unsqueeze(-1)
        delta2 = delta2 * cell_size
        # undistort points (homogeneous coords) when intrinsics provided
        delta2_undistorted = K2s[:, :2, :2].unsqueeze(1).inverse() @ delta2 if K2s is not None else delta2
        delta2_undistorted = delta2_undistorted.squeeze(-1)

        match_score = eval_result["match_score"]

        updated_correspondence = correspondence
        updated_correspondence[:, :, 2:4] = updated_correspondence[:, :, 2:4] + delta2_undistorted

    if match_score is not None and min_match_score_threshold > 0.0 and match_score.shape[1] > 5:
        if match_score.shape[0] > 1:
            # We cannot filter based on match score if batch size is larger than 1,
            # because each batch index would have a different number of remaining correspondences.
            raise ValueError("Cannot filter based on match score if batch size is larger than 1.")

        # Filter out correspondences with a match score below the threshold
        number_of_remaining_matches = 0
        while number_of_remaining_matches < 5:
            # We need at least 5 remaining matches to compute the pose error
            mask = match_score >= min_match_score_threshold
            min_match_score_threshold -= 0.01
            number_of_remaining_matches = mask.sum().item()

        updated_correspondence = updated_correspondence[mask].unsqueeze(dim=0)
        delta1 = delta1[mask].unsqueeze(dim=0)
        delta2 = delta2[mask].unsqueeze(dim=0)
        similarity1 = similarity1[mask].unsqueeze(dim=0)
        similarity2 = similarity2[mask].unsqueeze(dim=0)
        descr1 = descr1[mask].unsqueeze(dim=0)
        descr2 = descr2[mask].unsqueeze(dim=0)
        match_score = match_score[mask].unsqueeze(dim=0)
        if gt_pts2 is not None:
            gt_pts2 = gt_pts2[mask].unsqueeze(dim=0)
        if gt_pts2_mask is not None:
            gt_pts2_mask = gt_pts2_mask[mask].unsqueeze(dim=0)

    return {
        "updated_correspondence": updated_correspondence,
        "delta1": delta1,
        "delta2": delta2,
        "similarity1": similarity1,
        "similarity2": similarity2,
        "descr1": descr1,
        "descr2": descr2,
        "match_score": match_score,
        "gt_pts2": gt_pts2,
        "gt_pts2_mask": gt_pts2_mask,
        "embedding1": eval_result["embedding1"] if "embedding1" in eval_result else None,
        "embedding2": eval_result["embedding2"] if "embedding2" in eval_result else None,
        "embedding_after_attention1": (
            eval_result["embedding_after_attention1"] if "embedding_after_attention1" in eval_result else None
        ),
        "embedding_after_attention2": (
            eval_result["embedding_after_attention2"] if "embedding_after_attention2" in eval_result else None
        ),
    }


def get_dataset(
    opt, dataset_names: list[str], split_info_type: str, with_score: bool, logger
) -> torch.utils.data.Dataset:
    data_dir = opt.data_dir
    if not data_dir.endswith("/"):
        data_dir += "/"
    logger.info("Datasets are assumed to be located in: " + data_dir)

    list_of_datasets = []
    for ds in dataset_names:
        logger.info("Loading split " + split_info_type + " from dataset: " + data_dir + ds)
        if ds.endswith(".hdf5"):
            file_path = data_dir + ds
            dataset = SparsePatchDatasetFromHDF5(
                file_path=file_path,
                split_info_type=split_info_type,
                nfeatures=(
                    opt.nfeatures if (split_info_type == "train" or opt.force_nfeatures) else -1
                ),  # fix num matches only for train
                overwrite_side_info=not opt.sideinfo,
                patch_type=opt.patch_type,
                cell_size=opt.cell_size,
                with_score=with_score,
                with_depth=False,
                shift_KPs_to_pixel_center=opt.shift_KPs_to_pixel_center,
                patch_radius=opt.patch_radius,
                train_epe_threshold=opt.train_epe_threshold,
                adjust_only_second_keypoint=opt.adjust_only_second_keypoint,
                total_split_number=opt.total_split,
                current_split_number=opt.current_split,
            )
        else:
            if split_info_type == "eval":
                data_paths = [data_dir + ds + "/" + "test_" + opt.detector + "/"]
            elif split_info_type == "val":
                data_paths = [data_dir + ds + "/" + "val_" + opt.detector + "/"]
            elif split_info_type == "train":
                data_paths = [data_dir + ds + "/" + "train_" + opt.detector + "/"]
            else:
                raise ValueError(f"Unknown split_info_type: {split_info_type}")
            dataset = SparsePatchDataset(
                folders=data_paths,
                nfeatures=(
                    opt.nfeatures if (split_info_type == "train" or opt.force_nfeatures) else -1
                ),  # fix num matches only for train
                overwrite_side_info=not opt.sideinfo,
                without_score=not with_score,
                train=split_info_type == "train",
                shift_KPs_to_pixel_center=opt.shift_KPs_to_pixel_center,
                patch_radius=opt.patch_radius,
                adjust_only_second_keypoint=opt.adjust_only_second_keypoint,
                total_split=opt.total_split,
                current_split=opt.current_split,
            )
        list_of_datasets.append(dataset)

    dataset = list_of_datasets[0]
    for idx in range(1, len(list_of_datasets)):
        dataset = torch.utils.data.ConcatDataset([dataset, list_of_datasets[idx]])
    return dataset


def retrieve_sub_pixel_precision(
    score_map: torch.Tensor,
    pixel_precise_keypoints: torch.Tensor,
    origin_is_center_of_first_pixel: bool,
    input_keypoints_format_is_y_x: bool = False,
    add_border: bool = False,
) -> torch.Tensor:
    """
    Retrieve sub-pixel precise local maxima,
    by fitting a paraboloid to the score values in the neighborhood of the keypoints.
    Assumes that no keypoints are located on the score_map border!

    # We compute the paraboloid fit as follows:
    # First, approximate the derivatives of the score map at the keypoint location
    Sx = S(x+1, y) - S(x-1, y) # 1st order derivative at (x,y) in x-direction
    Sy = S(x, y+1) - S(x, y-1) # 1st order derivative at (x,y) in y-direction
    Sxx = S(x+1, y) - 2*S(x, y) + S(x-1, y) # 2nd order derivative at (x,y) in x-direction
    Syy = S(x, y+1) - 2*S(x, y) + S(x, y-1) # 2nd order derivative at (x,y) in y-direction
    Sxy = (S(x+1, y+1) - S(x+1, y-1) - S(x-1, y+1) + S(x-1, y-1)) / 4 # 2nd order derivative at (x,y) in x-y-direction
    # Based on these derivatives, we compute the shift in x and y direction as follows:
    dx = -(Syy * Sx - Sy * Sxy) / (Sxx * Syy - Sxy * Sxy)
    dy = -(Sxx * Sy - Sx * Sxy) / (Sxx * Syy - Sxy * Sxy)

    Args:
        score_map (torch.Tensor):
            Score map, shape [batch_size, image_height, image_width]
        pixel_precise_keypoints (torch.Tensor):
            Keypoints with integer coordinates, shape [batch_size, k, 2].
        origin_is_center_of_first_pixel (bool):
            If True, the origin is considered to be the center of the first pixel.
            If False, the origin is considered to be the top-left corner of the first pixel.
        input_keypoints_format_is_y_x (bool):
            If True, the input keypoints are expected to be in the format [y, x].
            If False, the input keypoints are expected to be in the format [x, y].
            Defaults to False.
        add_border (bool):
            If True, the score_map will be padded with a border of zeros around it.
            If False, you need to ensure that the keypoints are not located on the border of the score_map.
            Defaults to False.

    Returns:
        torch.Tensor: Tensor containing the coordinates of sub-pixel precise local maxima.
        The returned coordinates are in the same format as the input keypoints.
    """
    if not input_keypoints_format_is_y_x:
        # In the following, we assume that the input keypoints are in the format [y, x]
        pixel_precise_keypoints = pixel_precise_keypoints.flip(-1)

    if add_border:
        score_map = torch.nn.functional.pad(score_map, (1, 1, 1, 1), mode="constant", value=0.0)
        pixel_precise_keypoints += 1

    y, x = pixel_precise_keypoints[..., 0], pixel_precise_keypoints[..., 1]

    batch_size, _, _ = score_map.shape
    batch_indices = torch.arange(batch_size, device=score_map.device)[:, None]

    # Gather score values efficiently
    center_vals = score_map[batch_indices, y, x]
    right_vals = score_map[batch_indices, y, x + 1]
    left_vals = score_map[batch_indices, y, x - 1]
    top_vals = score_map[batch_indices, y + 1, x]
    bottom_vals = score_map[batch_indices, y - 1, x]

    # Compute derivatives dx and dy
    dx = 0.5 * (right_vals - left_vals) / (2 * center_vals - right_vals - left_vals)
    dy = 0.5 * (top_vals - bottom_vals) / (2 * center_vals - top_vals - bottom_vals)

    # Handle NaNs (possible division by zero)
    dx = torch.nan_to_num(dx, nan=0.0)
    dy = torch.nan_to_num(dy, nan=0.0)

    x, y = x.float(), y.float()

    if not origin_is_center_of_first_pixel:
        x += 0.5
        y += 0.5

    if add_border:
        x -= 1
        y -= 1

    sub_pixel_precise_keypoints = torch.stack([(x + dx), (y + dy)], dim=-1)
    if input_keypoints_format_is_y_x:
        sub_pixel_precise_keypoints = sub_pixel_precise_keypoints.flip(-1)
    return sub_pixel_precise_keypoints


def color_normalization(patch, feat_axis, color_normalization_strategy):
    if color_normalization_strategy == "simple_color":
        # bring into range [-0.5, 0.5]
        patch = patch - 0.5
    elif color_normalization_strategy in ["orig", "simple_gray"]:
        # If the image patch has three channels, we assume it to be RGB and convert to grayscale, using the formula
        # as in OpenCV: Gray = 0.299 * R + 0.587 * G + 0.114 * B. Therefore, the results should be the same whether the
        # image was loaded as color or grayscale.
        if patch.shape[1] == 3:  # RGB
            # rgb scaling
            scale = patch.new_tensor([0.299, 0.587, 0.114]).view(*([1] * feat_axis), 3, 1, 1)
            patch = (patch * scale).sum(feat_axis, keepdim=True)
        # For the orig strategy from keypt2subpx, we do not normalize the image patch.
        # For the simple_gray strategy, we bring the image patch into the range [-0.5, 0.5].
        if color_normalization_strategy == "simple_gray":
            # bring into range [-0.5, 0.5]
            patch = patch - 0.5
    else:
        raise ValueError("Unknown color normalization strategy: {}".format(color_normalization_strategy))
    return patch


# The following function is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def create_meshgrid(x: torch.Tensor, normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    return torch.meshgrid(ys, xs, **kw)  # pos_y, pos_x
