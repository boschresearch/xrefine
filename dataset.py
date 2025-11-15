# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.

import math
import os
import pickle
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SparsePatchDatasetFromHDF5(Dataset):
    """Sparse correspondences dataset with ground truth pose and intrinsics, created from hdf5"""

    def __init__(
        self,
        file_path: str,
        split_info_type: str,
        nfeatures: int,
        overwrite_side_info: bool = True,
        patch_type: str = "image",
        cell_size: int = 1,
        with_score: bool = False,
        with_depth: bool = False,
        shift_KPs_to_pixel_center: bool = False,
        patch_radius: int = 5,
        train_epe_threshold: float = -1.0,
        adjust_only_second_keypoint: bool = False,
        loud: bool = False,
        total_split_number: int = -1,
        current_split_number: int = -1,
    ):
        self.split_info_type = split_info_type
        self.nfeatures = nfeatures
        if not overwrite_side_info:
            raise NotImplementedError("Side information is not implemented for SparsePatchDatasetFromHDF5.")
        self.patch_type = patch_type
        self.cell_size = cell_size
        self.with_score = with_score
        self.with_depth = with_depth
        self.shift_KPs_to_pixel_center = shift_KPs_to_pixel_center
        self.patch_radius = patch_radius
        self.train_epe_threshold = train_epe_threshold
        self.adjust_only_second_keypoint = adjust_only_second_keypoint
        self.loud = loud
        self.minset = 5  # minimal set size for essential matrices

        self.samples = []
        file_path = Path(file_path)
        self.hdf5_file = h5py.File(file_path.absolute(), "r")

        for sample in self.hdf5_file[self.split_info_type]:
            self.samples.append(sample)

        if total_split_number > 0 and current_split_number > 0:
            start_iter = len(self.samples) * (current_split_number - 1) // total_split_number
            end_iter = len(self.samples) * current_split_number // total_split_number
            self.samples = self.samples[start_iter:end_iter]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.hdf5_file[self.split_info_type][self.samples[idx]]

        # correspondence coordinates and matching ratios (side information)
        pts1 = np.array(sample_data["source_keypoints"])
        pts2 = np.array(sample_data["target_keypoints"])
        # image sizes
        im_size1 = torch.from_numpy(np.array(sample_data["source_image_size"]))
        im_size2 = torch.from_numpy(np.array(sample_data["target_image_size"]))

        # patches
        if self.patch_type == "image":
            patch1 = torch.from_numpy(np.array(sample_data["source_patches"]))
            patch2 = torch.from_numpy(np.array(sample_data["target_patches"]))
            if patch1.shape[0] > 0 and patch2.shape[0] > 0:
                if patch1.min() < 0.0 or patch1.max() > 1.0 + 1e-5 or patch2.min() < 0.0 or patch2.max() > 1.0 + 1e-5:
                    print("WARNING: Patch values are not normalized to [0, 1].")
        elif self.patch_type == "descriptor":
            patch1 = torch.from_numpy(np.array(sample_data["source_descriptor_patches"]))
            patch2 = torch.from_numpy(np.array(sample_data["target_descriptor_patches"]))
        elif self.patch_type == "early_embedding":
            patch1 = torch.from_numpy(np.array(sample_data["source_early_embedding_patches"]))
            patch2 = torch.from_numpy(np.array(sample_data["target_early_embedding_patches"]))
        elif self.patch_type == "late_embedding":
            patch1 = torch.from_numpy(np.array(sample_data["source_late_embedding_patches"]))
            patch2 = torch.from_numpy(np.array(sample_data["target_late_embedding_patches"]))
        else:
            raise ValueError("Invalid patch type: {}".format(self.patch_type))

        if "source_descriptors" in sample_data and "target_descriptors" in sample_data:
            # feature descriptors
            descriptor1 = torch.from_numpy(np.array(sample_data["source_descriptors"]))
            descriptor2 = torch.from_numpy(np.array(sample_data["target_descriptors"]))
        else:
            descriptor1 = None
            descriptor2 = None

        if "source_camera_matrix" in sample_data and "target_camera_matrix" in sample_data:
            # image calibration parameters
            K1 = torch.from_numpy(np.array(sample_data["source_camera_matrix"]))
            K2 = torch.from_numpy(np.array(sample_data["target_camera_matrix"]))
        else:
            K1 = None
            K2 = None
        if "source_to_target_rotmat" in sample_data and "source_to_target_translation" in sample_data:
            # ground truth pose
            gt_R = torch.from_numpy(np.array(sample_data["source_to_target_rotmat"]))
            gt_t = torch.from_numpy(np.array(sample_data["source_to_target_translation"]))
        else:
            gt_R = None
            gt_t = None

        gt_pts2 = None
        gt_pts2_mask = None
        patch1_warped_to_patch2 = None
        patch1_warped_to_patch2_mask = None

        # Optional: score patches
        if self.with_score:
            scorepatch1 = torch.from_numpy(np.array(sample_data["source_score_patches"]))
            scorepatch2 = torch.from_numpy(np.array(sample_data["target_score_patches"]))
        else:
            scorepatch1 = None
            scorepatch2 = None
        # Optional: depth
        if self.with_depth:
            depth1 = torch.from_numpy(np.array(sample_data["source_depth"]))
            depth2 = torch.from_numpy(np.array(sample_data["target_depth"]))
        else:
            depth1 = None
            depth2 = None

        if "smp_error" in sample_data:
            smp_error = np.array(sample_data["smp_error"])
        else:
            smp_error = None

        if "end_point_error" in sample_data:
            end_point_error = np.array(sample_data["end_point_error"])
        else:
            end_point_error = None

        if "source_image_name" in sample_data and "target_image_name" in sample_data:
            image_name_1 = sample_data["source_image_name"][()].decode("utf-8")
            image_name_2 = sample_data["target_image_name"][()].decode("utf-8")
        else:
            image_name_1 = None
            image_name_2 = None

        if self.split_info_type == "train" and self.train_epe_threshold > 0.0:
            if end_point_error is None:
                raise ValueError("Cannot use train_epe_threshold: End point error not found in HDF5 file.")
            matches_to_keep = end_point_error < self.train_epe_threshold
            if gt_pts2_mask is not None:
                # Also remove the points without available ground truth (unknown epe)
                matches_to_keep = matches_to_keep & gt_pts2_mask.astype(bool)
            pts1 = pts1[matches_to_keep]
            pts2 = pts2[matches_to_keep]
            patch1 = patch1[matches_to_keep]
            patch2 = patch2[matches_to_keep]
            if descriptor1 is not None:
                descriptor1 = descriptor1[matches_to_keep]
            if descriptor2 is not None:
                descriptor2 = descriptor2[matches_to_keep]
            end_point_error = end_point_error[matches_to_keep]
            if gt_pts2 is not None:
                gt_pts2 = gt_pts2[matches_to_keep]
                if gt_pts2_mask is not None:
                    gt_pts2_mask = gt_pts2_mask[matches_to_keep]
            if patch1_warped_to_patch2 is not None:
                patch1_warped_to_patch2 = patch1_warped_to_patch2[matches_to_keep]
                if patch1_warped_to_patch2_mask is not None:
                    patch1_warped_to_patch2_mask = patch1_warped_to_patch2_mask[matches_to_keep]
            if self.with_score:
                scorepatch1 = scorepatch1[matches_to_keep]
                scorepatch2 = scorepatch2[matches_to_keep]
            if self.with_depth:
                depth1 = depth1[matches_to_keep]
                depth2 = depth2[matches_to_keep]

        if pts1.shape[0] < self.minset and self.split_info_type == "train":
            if self.loud:
                print(
                    "WARNING! Not enough correspondences left. \
                        Only %d correspondences among would be left, so I instead sample another one."
                    % (int(pts1.shape[1]))
                )
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        return prepare_sparse_patch_dataset_data(
            pts1=pts1,
            pts2=pts2,
            patch1=patch1,
            patch2=patch2,
            cell_size=self.cell_size,
            descriptor1=descriptor1,
            descriptor2=descriptor2,
            gt_pts2=gt_pts2,
            gt_pts2_mask=gt_pts2_mask,
            patch1_warped_to_patch2=patch1_warped_to_patch2,
            patch1_warped_to_patch2_mask=patch1_warped_to_patch2_mask,
            scorepatch1=scorepatch1,
            scorepatch2=scorepatch2,
            depth1=depth1,
            depth2=depth2,
            ratios=None,
            gt_t=gt_t,
            gt_R=gt_R,
            K1=K1,
            K2=K2,
            im_size1=im_size1,
            im_size2=im_size2,
            nfeatures=self.nfeatures,
            with_score=self.with_score,
            with_depth=self.with_depth,
            shift_KPs_to_pixel_center=self.shift_KPs_to_pixel_center,
            overwrite_side_info=True,
            patch_radius=self.patch_radius,
            adjust_only_second_keypoint=self.adjust_only_second_keypoint,
            end_point_error=torch.from_numpy(end_point_error) if end_point_error is not None else None,
            smp_error=torch.from_numpy(smp_error) if smp_error is not None else None,
            image_name_1=image_name_1,
            image_name_2=image_name_2,
        )


# The following class is based on code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class SparsePatchDataset(Dataset):
    """Sparse correspondences dataset with ground truth pose and intrinsics."""

    def __init__(
        self,
        folders,
        nfeatures,
        overwrite_side_info=False,
        without_score=False,
        without_depth=True,
        train=False,
        shift_KPs_to_pixel_center=False,
        patch_radius: int = 5,
        adjust_only_second_keypoint: bool = False,
        total_split=-1,
        current_split=-1,
    ):
        self.nfeatures = nfeatures  # ensure fixed number of features, -1 keeps original feature count
        self.overwrite_side_info = (
            overwrite_side_info  # if true, provide no side information to the neural guidance network
        )
        self.without_score = without_score
        self.without_depth = without_depth
        self.train = train
        self.shift_KPs_to_pixel_center = shift_KPs_to_pixel_center
        self.patch_radius = patch_radius
        self.adjust_only_second_keypoint = adjust_only_second_keypoint

        # collect precalculated correspondences of all provided datasets
        self.files = []
        for folder in folders:
            self.files += [os.path.join(folder, f) for f in os.listdir(folder)]

        if total_split > 0 and current_split > 0:
            start_iter = len(self.files) * (current_split - 1) // total_split
            end_iter = len(self.files) * current_split // total_split
            self.files = self.files[start_iter:end_iter]

        self.minset = 5  # minimal set size for essential matrices

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # load precalculated correspondences using pickle
        data = None
        with open(self.files[idx], "rb") as f:
            data = pickle.load(f)
        assert data is not None, f"Error: Correspondences {self.files[idx]} could not be loaded."

        # correspondence coordinates and matching ratios (side information)
        pts1, pts2, ratios = data[0], data[1], data[2]  # N x 2, N x 2, N

        # image sizes
        im_size1, im_size2 = torch.from_numpy(np.asarray(data[3])), torch.from_numpy(np.asarray(data[4]))
        # image calibration parameters
        K1, K2 = torch.from_numpy(data[5]), torch.from_numpy(data[6])
        # ground truth pose
        gt_R, gt_t = torch.from_numpy(data[7]), torch.from_numpy(data[8])

        # hack: try the inverse
        # gt_R = gt_R.transpose(1, 0)
        # gt_t = -gt_R.mm(gt_t)

        patch1, patch2 = torch.from_numpy(data[9]), torch.from_numpy(data[10])
        if not self.without_score:
            scorepatch1, scorepatch2 = torch.from_numpy(data[11]), torch.from_numpy(data[12])
            descriptor1, descriptor2 = torch.from_numpy(data[13]), torch.from_numpy(data[14])
        else:
            scorepatch1 = None
            scorepatch2 = None
            descriptor1, descriptor2 = torch.from_numpy(data[11]), torch.from_numpy(data[12])

        if not self.without_depth:
            depth1, depth2 = torch.from_numpy(data[-2]), torch.from_numpy(data[-1])
        else:
            depth1 = None
            depth2 = None

        if pts1.shape[0] < self.minset and self.train:
            print(
                "WARNING! Not enough correspondences left. \
                    Only %d correspondences among would be left, so I instead sample another one."
                % (int(pts1.shape[1]))
            )
            return self.__getitem__(np.random.randint(0, len(self.files)))

        return prepare_sparse_patch_dataset_data(
            pts1=pts1,
            pts2=pts2,
            patch1=patch1,
            patch2=patch2,
            cell_size=1,
            descriptor1=descriptor1,
            descriptor2=descriptor2,
            gt_pts2=None,
            gt_pts2_mask=None,
            patch1_warped_to_patch2=None,
            patch1_warped_to_patch2_mask=None,
            scorepatch1=scorepatch1,
            scorepatch2=scorepatch2,
            depth1=depth1,
            depth2=depth2,
            ratios=ratios,
            gt_t=gt_t,
            gt_R=gt_R,
            K1=K1,
            K2=K2,
            im_size1=im_size1,
            im_size2=im_size2,
            nfeatures=self.nfeatures,
            with_score=not self.without_score,
            with_depth=not self.without_depth,
            shift_KPs_to_pixel_center=self.shift_KPs_to_pixel_center,
            overwrite_side_info=self.overwrite_side_info,
            patch_radius=self.patch_radius,
            adjust_only_second_keypoint=self.adjust_only_second_keypoint,
        )


# The following function is based on code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def prepare_sparse_patch_dataset_data(
    pts1: np.ndarray,
    pts2: np.ndarray,
    patch1: torch.Tensor,
    patch2: torch.Tensor,
    cell_size: int,
    descriptor1: Optional[torch.Tensor],
    descriptor2: Optional[torch.Tensor],
    gt_pts2: Optional[torch.Tensor],
    gt_pts2_mask: Optional[torch.Tensor],
    patch1_warped_to_patch2: Optional[torch.Tensor],
    patch1_warped_to_patch2_mask: Optional[torch.Tensor],
    scorepatch1: Optional[torch.Tensor],
    scorepatch2: Optional[torch.Tensor],
    depth1: Optional[torch.Tensor],
    depth2: Optional[torch.Tensor],
    ratios: np.ndarray,
    gt_t: np.ndarray,
    gt_R: np.ndarray,
    K1: torch.Tensor,
    K2: torch.Tensor,
    im_size1: np.ndarray,
    im_size2: np.ndarray,
    nfeatures: int,
    with_score: bool,
    with_depth: bool,
    shift_KPs_to_pixel_center: bool,
    overwrite_side_info: bool,
    patch_radius: int,
    adjust_only_second_keypoint: bool,
    end_point_error: Optional[torch.Tensor] = None,
    smp_error: Optional[torch.Tensor] = None,
    image_name_1: Optional[str] = None,
    image_name_2: Optional[str] = None,
):
    if patch1.shape[2] != patch1.shape[3] or patch1.shape[2] != patch2.shape[2] or patch1.shape[3] != patch2.shape[3]:
        raise ValueError("Patch dimensions are not equal.")

    current_patch_radius = (patch1.shape[2] - 1) // 2
    if patch_radius > current_patch_radius:
        raise ValueError("Requested patch radius is too large for the available patches.")

    if patch_radius < current_patch_radius:
        # Reduce size of patches
        start = current_patch_radius - patch_radius
        end = current_patch_radius + patch_radius + 1
        patch1 = patch1[:, :, start:end, start:end]
        patch2 = patch2[:, :, start:end, start:end]
        if patch1_warped_to_patch2 is not None:
            patch1_warped_to_patch2 = patch1_warped_to_patch2[:, :, start:end, start:end]
            if patch1_warped_to_patch2_mask is not None:
                patch1_warped_to_patch2_mask = patch1_warped_to_patch2_mask[:, :, start:end, start:end]
                # Set mask at points that warp outside the reduced target patch to zero
                patch1_warped_to_patch2_mask[patch1_warped_to_patch2[:, :1, :, :] < start] = 0
                patch1_warped_to_patch2_mask[patch1_warped_to_patch2[:, :1, :, :] > end] = 0
                patch1_warped_to_patch2_mask[patch1_warped_to_patch2[:, 1:2, :, :] < start] = 0
                patch1_warped_to_patch2_mask[patch1_warped_to_patch2[:, 1:2, :, :] > end] = 0
            # Adapt the origin of the warped points
            patch1_warped_to_patch2 = patch1_warped_to_patch2 - start
        if with_score:
            scorepatch1 = scorepatch1[:, :, start:end, start:end]
            scorepatch2 = scorepatch2[:, :, start:end, start:end]

    if overwrite_side_info:
        ratios = np.zeros(pts1.shape[0], dtype=np.float32)

    if pts1.shape[0] > 0:
        if shift_KPs_to_pixel_center:
            if not adjust_only_second_keypoint:
                # Only shift pts1 to pixel center, if both keypoints are updated
                pts1 = np.floor(pts1) + 0.5 * cell_size
            pts2 = np.floor(pts2) + 0.5 * cell_size

        if K1 is not None and K2 is not None:
            # for essential matrices, normalize image coordinate using the calibration parameters
            pts1 = cv2.undistortPoints(pts1[None], K1.numpy(), None).squeeze(1)
            pts2 = cv2.undistortPoints(pts2[None], K2.numpy(), None).squeeze(1)
            if gt_pts2 is not None:
                gt_pts2 = cv2.undistortPoints(gt_pts2[None], K2.numpy(), None).squeeze(1)

    # stack image coordinates and side information into one tensor
    correspondences = np.concatenate((pts1, pts2, ratios[..., None]), axis=-1)
    # correspondences = np.transpose(correspondences)
    correspondences = torch.from_numpy(correspondences)

    indices = None
    if nfeatures > 0 and pts1.shape[0] > 0:
        # ensure that there are exactly n features in the data tensor
        if correspondences.size(0) > nfeatures:
            rnd = torch.randperm(correspondences.size(0))
            correspondences = correspondences[rnd]
            correspondences = correspondences[0:nfeatures]
            indices = rnd[0:nfeatures]
        else:
            indices = torch.arange(0, correspondences.size(0))

        if correspondences.size(0) < nfeatures:
            prepared_data = correspondences
            for _ in range(0, math.ceil(nfeatures / correspondences.size(0) - 1)):
                rnd = torch.randperm(correspondences.size(0))
                prepared_data = torch.cat((prepared_data, correspondences[rnd]), dim=0)
                indices = torch.cat((indices, rnd))
            correspondences = prepared_data[0:nfeatures]
            indices = indices[0:nfeatures]

    if indices is not None:
        patch1, patch2 = patch1[indices], patch2[indices]
        if descriptor1 is not None:
            descriptor1 = descriptor1[indices]
        if descriptor2 is not None:
            descriptor2 = descriptor2[indices]
        if gt_pts2 is not None:
            gt_pts2 = gt_pts2[indices]
        if gt_pts2_mask is not None:
            gt_pts2_mask = gt_pts2_mask[indices]
        if patch1_warped_to_patch2 is not None:
            patch1_warped_to_patch2 = patch1_warped_to_patch2[indices]
        if patch1_warped_to_patch2_mask is not None:
            patch1_warped_to_patch2_mask = patch1_warped_to_patch2_mask[indices]
        if with_score:
            scorepatch1 = scorepatch1[indices]
            scorepatch2 = scorepatch2[indices]
        if with_depth:
            depth1 = depth1[indices]
            depth2 = depth2[indices]
        if end_point_error is not None:
            end_point_error = end_point_error[indices]
        if smp_error is not None:
            smp_error = smp_error[indices]

    if gt_t is not None and gt_R is not None:
        # construct the ground truth essential matrix from the ground truth relative pose
        gt_E = torch.zeros((3, 3))
        gt_E[0, 1] = -float(gt_t[2, 0])
        gt_E[0, 2] = float(gt_t[1, 0])
        gt_E[1, 0] = float(gt_t[2, 0])
        gt_E[1, 2] = -float(gt_t[0, 0])
        gt_E[2, 0] = -float(gt_t[1, 0])
        gt_E[2, 1] = float(gt_t[0, 0])
        gt_E = gt_E.mm(gt_R)
    else:
        gt_E = None

    prepared_data = {
        "correspondences": correspondences,
        "im_size1": im_size1,
        "im_size2": im_size2,
        "patch1": patch1,
        "patch2": patch2,
    }
    if descriptor1 is not None:
        prepared_data["descriptor1"] = descriptor1
    if descriptor2 is not None:
        prepared_data["descriptor2"] = descriptor2
    if gt_E is not None:
        prepared_data["gt_E"] = gt_E
    if gt_R is not None:
        prepared_data["gt_R"] = gt_R
    if gt_t is not None:
        prepared_data["gt_t"] = gt_t
    if K1 is not None:
        prepared_data["K1"] = K1
    if K2 is not None:
        prepared_data["K2"] = K2
    if gt_pts2 is not None:
        prepared_data["gt_pts2"] = gt_pts2
    if gt_pts2_mask is not None:
        prepared_data["gt_pts2_mask"] = gt_pts2_mask
    if patch1_warped_to_patch2 is not None:
        prepared_data["patch1_warped_to_patch2"] = patch1_warped_to_patch2
    if patch1_warped_to_patch2_mask is not None:
        prepared_data["patch1_warped_to_patch2_mask"] = patch1_warped_to_patch2_mask
    if with_score:
        prepared_data["scorepatch1"] = scorepatch1
        prepared_data["scorepatch2"] = scorepatch2
    if with_depth:
        prepared_data["depth1"] = depth1
        prepared_data["depth2"] = depth2
    if end_point_error is not None:
        prepared_data["end_point_error"] = end_point_error
    if smp_error is not None:
        prepared_data["smp_error"] = smp_error
    if image_name_1 is not None:
        prepared_data["image_name_1"] = image_name_1
    if image_name_2 is not None:
        prepared_data["image_name_2"] = image_name_2

    return prepared_data
