# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from data_utils import (
    boolean_string,
    extract_patches_from_map,
    get_normalized_coordinates,
    image_grid,
    nearest_neighbor_match,
    sample_distortions,
    sampson_dist,
    warp_source_points_to_target,
)
from mega_depth_gluefactory import MegaDepthGlueFactoryDataset
from torch.utils.data import DataLoader, Dataset


def parse_args():
    """Parse arguments for creation of matches dataset"""
    parser = argparse.ArgumentParser(
        description="Creation of matches dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--local_feature",
        dest="local_feature",
        type=str,
        help="The type of local feature to use",
        default="",
    )
    parser.add_argument(
        "--keypoints_per_image",
        dest="keypoints_per_image",
        type=int,
        help="Use only the k best interest points.",
        default=4096,
    )
    parser.add_argument(
        "--output_directory",
        dest="output_directory",
        type=str,
        help="Output location for the dataset .hdf5-file",
        default="../processed_data/train_val",
    )
    parser.add_argument(
        "--dataset_mode",
        dest="dataset_mode",
        type=str,
        help="Dataset mode to use. Can be 'train_val', to include training and validation data, 'val' or 'eval',"
        + " to include validation or evaluation data, or 'all', to include training, validation, and evaluation data.",
        default="eval",
    )
    parser.add_argument(
        "--patch_radius",
        dest="patch_radius",
        type=int,
        help="Radius for the patches cut from images (patch_w=patch_h=2*R+1)",
        default=5,
    )
    parser.add_argument(
        "--min_patch_radius_of_additional_patches",
        dest="min_patch_radius_of_additional_patches",
        type=int,
        help="This is the minimum patch radius for additional patches (score, descriptors, intermediate embeddings).",
        default=2,
    )
    parser.add_argument(
        "--skip_image_pairs_with_less_than_8_matches_for_training_split",
        dest="skip_image_pairs_with_less_than_8_matches_for_training_split",
        help="If set to True, matches from training data with large Sampson distance are not written to the dataset.",
        type=boolean_string,
        default="True",
    )
    parser.add_argument(
        "--extract_score_patches",
        dest="extract_score_patches",
        help="If set to True, score patches are extracted at the keypoint positions.",
        type=boolean_string,
        default="False",
    )
    parser.add_argument(
        "--use_torch_hub_xfeat",
        dest="use_torch_hub_xfeat",
        help="If set to True, the XFeat model from torch hub is used for feature extraction.",
        type=boolean_string,
        default="True",
    )
    parser.add_argument(
        "--matching_approach",
        dest="matching_approach",
        type=str,
        help="Matching approach. Options: mnn (mutual nearest neighbor), dsm (double soft max), lightglue",
        default="eval",
    )
    parser.add_argument(
        "--apply_xfeat_refinement",
        dest="apply_xfeat_refinement",
        help="If set to True, matched keypoints are refined with the XFeat approach.",
        type=boolean_string,
        default="True",
    )
    parser.add_argument(
        "--process_with_gluefactory",
        dest="process_with_gluefactory",
        help="If set to True, feature extraction and matching is done with GlueFactory. "
        + "This option is available for SuperPoint (+NN, +LG), ALIKED (+LG), SIFT (+NN), DISK (+LG)",
        type=boolean_string,
        default="False",
    )
    parser.add_argument(
        "--sample_matches",
        dest="sample_matches",
        help="If set to 'uniformly' or 'normally', matching points are randomly sampled and distorted "
        + " with distortion vectors sampled from a uniform, respectively normal, distribution.",
        type=str,
        default="no",
    )
    parser.add_argument(
        "--normal_distortion_std",
        dest="normal_distortion_std",
        type=float,
        help="The standard deviation of the normal distribution used, if sample_matches is set to 'normally'.",
        default=0.3,
    )
    args = parser.parse_args()
    return args


def get_dataset_and_dataloader(
    split_info_type: str,
    use_every_nth_image: int,
) -> tuple[Dataset, DataLoader]:
    dataset = MegaDepthGlueFactoryDataset(
        split_info_type=split_info_type,
    )
    if use_every_nth_image > 1:
        subsampled_samples = []
        for index in range(0, len(dataset.samples), use_every_nth_image):
            subsampled_samples.append(dataset.samples[index])
        dataset.samples = subsampled_samples
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
    )
    return dataset, dataloader


def main():
    """Script for creation of dataset."""
    args = parse_args()

    if args.use_torch_hub_xfeat:
        keypoint_net = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=args.keypoints_per_image,
        ).cuda()
    elif args.process_with_gluefactory:
        if args.local_feature == "superpoint" and args.matching_approach == "mnn":
            modelconf = {
                "name": "two_view_pipeline",
                "extractor": {
                    "name": "superpoint_densescore",
                    "max_num_keypoints": args.keypoints_per_image,
                    "force_num_keypoints": False,
                    "detection_threshold": 0.00015,
                    "nms_radius": 4,
                    "remove_borders": 4,
                    "trainable": False,
                },
                "matcher": {"name": "matchers.nearest_neighbor_matcher"},
                "ground_truth": {"name": "matchers.depth_matcher", "th_positive": 3, "th_negative": 5, "th_epi": 5},
                "allow_no_extract": True,
            }
        elif args.local_feature == "superpoint" and args.matching_approach == "lightglue":
            modelconf = {
                "name": "two_view_pipeline",
                "extractor": {
                    "name": "superpoint_densescore",
                    "max_num_keypoints": args.keypoints_per_image,
                    "force_num_keypoints": False,
                    "detection_threshold": 0.0,
                    "nms_radius": 3,
                    "remove_borders": 3,
                    "trainable": False,
                },
                "matcher": {
                    "name": "matchers.lightglue_pretrained",
                    "features": "superpoint",
                    "depth_confidence": -1,
                    "width_confidence": -1,
                    "filter_threshold": 0.1,
                    "trainable": False,
                },
                "ground_truth": {"name": "matchers.depth_matcher", "th_positive": 3, "th_negative": 5, "th_epi": 5},
                "allow_no_extract": True,
            }
        elif args.local_feature == "aliked" and args.matching_approach == "lightglue":
            modelconf = {
                "name": "two_view_pipeline",
                "extractor": {
                    "name": "extractors.aliked",
                    "max_num_keypoints": args.keypoints_per_image,
                    "detection_threshold": 0.0,
                },
                "matcher": {
                    "name": "matchers.lightglue_pretrained",
                    "features": "aliked",
                    "depth_confidence": -1,
                    "width_confidence": -1,
                    "filter_threshold": 0.1,
                    "trainable": False,
                },
                "ground_truth": {"name": "matchers.depth_matcher", "th_positive": 3, "th_negative": 5, "th_epi": 5},
                "allow_no_extract": True,
            }
        elif args.local_feature == "disk" and args.matching_approach == "lightglue":
            modelconf = {
                "name": "two_view_pipeline",
                "extractor": {
                    "name": "extractors.disk_kornia",
                    "max_num_keypoints": args.keypoints_per_image,
                    "detection_threshold": 0.0,
                },
                "matcher": {
                    "name": "matchers.lightglue_pretrained",
                    "features": "disk",
                    "depth_confidence": -1,
                    "width_confidence": -1,
                    "filter_threshold": 0.1,
                    "trainable": False,
                },
                "ground_truth": {"name": "matchers.depth_matcher", "th_positive": 3, "th_negative": 5, "th_epi": 5},
                "allow_no_extract": True,
            }
        elif args.local_feature == "sift" and args.matching_approach == "mnn":
            modelconf = {
                "name": "two_view_pipeline",
                "extractor": {
                    "name": "extractors.sift",
                    "detector": "pycolmap_cuda",
                    "max_num_keypoints": args.keypoints_per_image,
                    "detection_threshold": 0.00666666,
                    "nms_radius": -1,
                    "pycolmap_options": {"first_octave": -1},
                },
                "matcher": {
                    "name": "matchers.nearest_neighbor_matcher",
                },
                "ground_truth": {"name": "matchers.depth_matcher", "th_positive": 3, "th_negative": 5, "th_epi": 5},
                "allow_no_extract": True,
            }
        else:
            raise ValueError("Unknown combination of feature type and matching approach for GlueFactory processing.")

        try:
            from gluefactory.models import get_model as get_model_gf
            from gluefactory.utils.tensor import batch_to_device as batch_to_device_gf
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("Please install the gluefactory package to process_with_gluefactory.")

        modelconf = OmegaConf.create(modelconf)

        try:
            keypoint_net = get_model_gf(modelconf.name)(modelconf).to("cuda")
        except RuntimeError:
            error_msg = "Cannot find superpoint_densescore in GlueFactory. "
            error_msg += (
                "Copy https://github.com/KimSinjeong/keypt2subpx/blob/master/dataprocess/superpoint_densescore.py"
            )
            error_msg += "to the folder gluefactory/models/extractors/ in the glue-factory repository."
            raise RuntimeError(error_msg)
    elif args.sample_matches in ["uniformly", "normally"]:
        keypoint_net = None
    else:
        raise ValueError(
            "Select either use_torch_hub_xfeat, process_with_gluefactory or set sample_matches to uniformly or normally"
        )
    if keypoint_net is not None:
        keypoint_net.eval()

    Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    number_of_keypoints_per_image_str = args.keypoints_per_image
    if number_of_keypoints_per_image_str <= 0:
        number_of_keypoints_per_image_str = "all"
    number_of_keypoints_per_image_str = str(number_of_keypoints_per_image_str)

    matching_approach = args.matching_approach
    if args.apply_xfeat_refinement:
        matching_approach += "_xfeat_refined"

    extraction_approach = args.local_feature
    if args.sample_matches in ["uniformly", "normally"]:
        extraction_approach = "sampled"
        matching_approach = args.sample_matches
        if args.sample_matches == "normally":
            normal_distortion_std_str = f"{args.normal_distortion_std * args.patch_radius:.2f}".replace(".", "p")
            matching_approach += "-" + normal_distortion_std_str

    hdf5_file_path = os.path.join(
        args.output_directory,
        "MegaDepthGlueFactory"
        + "_"
        + extraction_approach
        + "_"
        + matching_approach
        + "_"
        + number_of_keypoints_per_image_str
        + ".hdf5",
    )

    if os.path.exists(hdf5_file_path):
        print(f"Dataset {hdf5_file_path} already exists. Exiting.")
        sys.exit(0)

    print("Creating dataset at: " + hdf5_file_path)
    hdf5_file = h5py.File(hdf5_file_path, "w")

    with torch.no_grad():
        if args.dataset_mode == "train_val":
            splits = ["train", "val"]
        elif args.dataset_mode == "val":
            splits = ["val"]
        elif args.dataset_mode == "eval":
            splits = ["eval"]
        elif args.dataset_mode == "all":
            splits = ["train", "val", "eval"]
        else:
            raise ValueError("Unknown dataset mode: " + args.dataset_mode)
        for split in splits:
            if split == "train":
                split_info_type = "training"
                use_every_nth_image = 1
            elif split == "val":
                split_info_type = "validation"
                use_every_nth_image = 1
            elif split == "eval":
                split_info_type = "evaluation"
                use_every_nth_image = 10

            print("Adding data for split: " + split)
            dataset_group = hdf5_file.create_group(split)

            _, dataloader = get_dataset_and_dataloader(
                split_info_type,
                use_every_nth_image,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for index, dataset_samples in enumerate(dataloader):
                if index % 500 == 0:
                    print("Processing sample: " + str(index))

                # The Keypt2Subpx repo processes padded images when using GlueFactory processing.
                # If we don't use GlueFactory, we remove the padding.
                actual_source_image_width = dataset_samples["source_image"].shape[-1]
                actual_source_image_height = dataset_samples["source_image"].shape[-2]

                if (
                    "source_image_width" in dataset_samples
                    and "source_image_height" in dataset_samples
                    and (
                        dataset_samples["source_image_width"] != actual_source_image_width
                        or dataset_samples["source_image_height"] != actual_source_image_height
                    )
                    and not args.process_with_gluefactory
                ):
                    dataset_samples["source_image"] = dataset_samples["source_image"][
                        :, :, : dataset_samples["source_image_height"], : dataset_samples["source_image_width"]
                    ]
                    if "source_depth_map" in dataset_samples:
                        dataset_samples["source_depth_map"] = dataset_samples["source_depth_map"][
                            :, :, : dataset_samples["source_image_height"], : dataset_samples["source_image_width"]
                        ]
                actual_target_image_width = dataset_samples["target_image"].shape[-1]
                actual_target_image_height = dataset_samples["target_image"].shape[-2]
                if (
                    "source_image_width" in dataset_samples
                    and "source_image_height" in dataset_samples
                    and (
                        dataset_samples["target_image_width"] != actual_target_image_width
                        or dataset_samples["target_image_height"] != actual_target_image_height
                    )
                    and not args.process_with_gluefactory
                ):
                    dataset_samples["target_image"] = dataset_samples["target_image"][
                        :, :, : dataset_samples["target_image_height"], : dataset_samples["target_image_width"]
                    ]
                    if "target_depth_map" in dataset_samples:
                        dataset_samples["target_depth_map"] = dataset_samples["target_depth_map"][
                            :, :, : dataset_samples["target_image_height"], : dataset_samples["target_image_width"]
                        ]

                if "target_camera_pose" in dataset_samples and "source_camera_pose" in dataset_samples:
                    dataset_samples["gt_source_to_target_transform"] = torch.matmul(
                        dataset_samples["target_camera_pose"],
                        torch.linalg.inv(dataset_samples["source_camera_pose"]),
                    )

                tensors_to_move_to_device = [
                    "source_image",
                    "target_image",
                    "source_depth_map",
                    "target_depth_map",
                    "source_camera_matrix",
                    "target_camera_matrix",
                    "gt_source_to_target_transform",
                    "source_to_target_flow_map",
                ]
                for tensor_name in tensors_to_move_to_device:
                    if tensor_name in dataset_samples:
                        dataset_samples[tensor_name] = dataset_samples[tensor_name].to(
                            dtype=torch.float32, device=device, non_blocking=True
                        )

                # save original images before they get normalized during inference
                target_image = dataset_samples["target_image"].clone()
                source_image = dataset_samples["source_image"].clone()
                assert (target_image[0] < (1.0 + 1e-5)).all() and (
                    target_image > -(1.0 + 1e-5)
                ).all(), "Target image out of range"
                assert (source_image[0] < (1.0 + 1e-5)).all() and (
                    source_image > -(1.0 + 1e-5)
                ).all(), "Target image out of range"

                assert source_image.shape[0] == 1, "Batchsize has to be one!"

                # Feature extraction
                selected_source_scales = None

                if args.sample_matches not in ["uniformly", "normally"]:
                    if args.use_torch_hub_xfeat:
                        if args.local_feature == "xfeat_orig":
                            source_output = keypoint_net.detectAndCompute(
                                dataset_samples["source_image"], top_k=args.keypoints_per_image
                            )[0]
                            selected_source_keypoints, _, selected_source_descriptors = (
                                source_output["keypoints"].cpu().numpy(),
                                source_output["scores"].unsqueeze(dim=1).cpu().numpy(),
                                source_output["descriptors"].cpu().numpy(),
                            )

                            target_output = keypoint_net.detectAndCompute(
                                dataset_samples["target_image"], top_k=args.keypoints_per_image
                            )[0]
                            selected_target_keypoints, _, selected_target_descriptors = (
                                target_output["keypoints"].cpu().numpy(),
                                target_output["scores"].unsqueeze(dim=1).cpu().numpy(),
                                target_output["descriptors"].cpu().numpy(),
                            )
                        elif args.local_feature == "xfeat_star":
                            source_output = keypoint_net.detectAndComputeDense(
                                dataset_samples["source_image"], top_k=args.keypoints_per_image
                            )
                            selected_source_keypoints, selected_source_descriptors, selected_source_scales = (
                                source_output["keypoints"].squeeze(dim=0).cpu().numpy(),
                                source_output["descriptors"].squeeze(dim=0).cpu().numpy(),
                                source_output["scales"].squeeze(dim=0).cpu().numpy(),
                            )
                            target_output = keypoint_net.detectAndComputeDense(
                                dataset_samples["target_image"], top_k=args.keypoints_per_image
                            )
                            selected_target_keypoints, selected_target_descriptors = (
                                target_output["keypoints"].squeeze(dim=0).cpu().numpy(),
                                target_output["descriptors"].squeeze(dim=0).cpu().numpy(),
                            )
                        else:
                            raise ValueError(
                                "Option use_torch_hub_xfeat is only available for xfeat_orig and xfeat_star."
                            )

                # Matching
                if args.matching_approach == "mnn" and not args.process_with_gluefactory:
                    gf_matching_output = nearest_neighbor_match(
                        torch.from_numpy(selected_source_descriptors).unsqueeze(0),
                        torch.from_numpy(selected_target_descriptors).unsqueeze(0),
                    )
                    matches = gf_matching_output["matches0"][0]
                    matches_mask = matches != -1
                    matched_source_keypoints = selected_source_keypoints[matches_mask]
                    matched_target_keypoints = selected_target_keypoints[matches[matches_mask]]
                    matched_source_descriptors = selected_source_descriptors[matches_mask]
                    matched_target_descriptors = selected_target_descriptors[matches[matches_mask]]
                    if selected_source_scales is not None:
                        matched_source_scales = selected_source_scales[matches_mask]
                elif args.process_with_gluefactory:
                    source_image_size = torch.tensor(
                        [
                            dataset_samples["source_image_width"],
                            dataset_samples["source_image_height"],
                        ]
                    ).unsqueeze(0)
                    target_image_size = torch.tensor(
                        [
                            dataset_samples["target_image_width"],
                            dataset_samples["target_image_height"],
                        ]
                    ).unsqueeze(0)
                    data_gf = {
                        "view0": {
                            "image": dataset_samples["source_image"],
                            "image_size": source_image_size,
                        },
                        "view1": {
                            "image": dataset_samples["target_image"],
                            "image_size": target_image_size,
                        },
                    }
                    pred = keypoint_net(batch_to_device_gf(data_gf, "cuda", non_blocking=True))

                    # After feature extraction, we can now remove the image padding
                    dataset_samples["source_image"] = dataset_samples["source_image"][
                        :, :, : dataset_samples["source_image_height"], : dataset_samples["source_image_width"]
                    ]
                    source_image = dataset_samples["source_image"].clone()
                    if "source_depth_map" in dataset_samples:
                        dataset_samples["source_depth_map"] = dataset_samples["source_depth_map"][
                            :, :, : dataset_samples["source_image_height"], : dataset_samples["source_image_width"]
                        ]
                    dataset_samples["target_image"] = dataset_samples["target_image"][
                        :, :, : dataset_samples["target_image_height"], : dataset_samples["target_image_width"]
                    ]
                    target_image = dataset_samples["target_image"].clone()
                    if "target_depth_map" in dataset_samples:
                        dataset_samples["target_depth_map"] = dataset_samples["target_depth_map"][
                            :, :, : dataset_samples["target_image_height"], : dataset_samples["target_image_width"]
                        ]

                    matches1 = pred["matches0"][0]
                    mask1 = matches1 != -1

                    if not mask1.any():
                        # We need at least one match to continue
                        # -> This sample will result in an outlier during val/eval; and is skipped during train
                        mask1[0] = True

                    matched_source_keypoints = pred["keypoints0"][0][mask1].cpu().numpy()
                    matched_target_keypoints = pred["keypoints1"][0][matches1[mask1]].cpu().numpy()
                    matched_source_descriptors = pred["descriptors0"][0][mask1].cpu().numpy()
                    matched_target_descriptors = pred["descriptors1"][0][matches1[mask1]].cpu().numpy()

                    if args.extract_score_patches:
                        if "score_map0" in pred and "score_map1" in pred:
                            source_score_map = pred["score_map0"][
                                :, :, : dataset_samples["source_image_height"], : dataset_samples["source_image_width"]
                            ]
                            target_score_map = pred["score_map1"][
                                :, :, : dataset_samples["target_image_height"], : dataset_samples["target_image_width"]
                            ]
                        elif "keypoint_scores0" in pred and "keypoint_scores1" in pred:
                            source_score_map = pred["keypoint_scores0"].unsqueeze(dim=0)[
                                :, :, : dataset_samples["source_image_height"], : dataset_samples["source_image_width"]
                            ]
                            target_score_map = pred["keypoint_scores1"].unsqueeze(dim=0)[
                                :, :, : dataset_samples["target_image_height"], : dataset_samples["target_image_width"]
                            ]
                        else:
                            raise RuntimeError(
                                "Unknown score map location for extract_score_patches in GlueFactory output."
                            )
                elif args.sample_matches in ["uniformly", "normally"]:
                    source_image_height, source_image_width = dataset_samples["source_image"].shape[-2:]

                    # Create a grid of source points
                    source_point_grid = image_grid(
                        batch_size=1,
                        grid_height=source_image_height,
                        grid_width=source_image_width,
                        dtype=torch.float32,
                        device=device,
                        ones=False,
                        normalized=False,
                    )

                    # Obtain normalized coordinates
                    source_point_grid_norm = get_normalized_coordinates(
                        source_point_grid, source_image_width, source_image_height
                    )

                    # Initialize the source points mask
                    source_points_mask = torch.ones(
                        (
                            1,
                            source_image_height,
                            source_image_width,
                        ),
                        device=device,
                        dtype=torch.bool,
                    )

                    # Identify corresponding points in the target image
                    source_points_warped_to_target, _, source_points_mask = warp_source_points_to_target(
                        data=dataset_samples,
                        source_uv=source_point_grid,
                        source_uv_norm=source_point_grid_norm,
                        keypoint_mask=source_points_mask,
                        origin_is_center_of_first_pixel=True,
                    )

                    # Abort if there are not enough valid points
                    valid_points_mask = source_points_mask.squeeze(dim=0)
                    num_valid_points = valid_points_mask.sum().item()
                    if num_valid_points < 8:
                        continue

                    # Remove points without valid correspondences
                    source_point_grid = source_point_grid[:, :, valid_points_mask]
                    source_points_warped_to_target = source_points_warped_to_target[:, :, valid_points_mask]
                    # source_point_grid and source_points_warped_to_target have shape [1, 2, num_valid_points]

                    # Reshape to [num_valid_points, 2]
                    source_point_grid = source_point_grid.squeeze(dim=0).permute(1, 0)
                    source_points_warped_to_target = source_points_warped_to_target.squeeze(dim=0).permute(1, 0)

                    # Randomly select num_keypoints
                    num_keypoints = min(args.keypoints_per_image, num_valid_points)
                    selected_indexes = torch.randperm(num_valid_points)[:num_keypoints]

                    matched_source_keypoints = source_point_grid[selected_indexes]
                    matched_target_keypoints = source_points_warped_to_target[selected_indexes]

                    # Distort the keypoints
                    source_keypoints_x_distortion = sample_distortions(
                        sample_matches=args.sample_matches,
                        num_keypoints=num_keypoints,
                        max_distortion=args.patch_radius,
                        normal_distortion_std=args.normal_distortion_std,
                        device=device,
                    )
                    source_keypoints_y_distortion = sample_distortions(
                        sample_matches=args.sample_matches,
                        num_keypoints=num_keypoints,
                        max_distortion=args.patch_radius,
                        normal_distortion_std=args.normal_distortion_std,
                        device=device,
                    )
                    target_keypoints_x_distortion = sample_distortions(
                        sample_matches=args.sample_matches,
                        num_keypoints=num_keypoints,
                        max_distortion=args.patch_radius,
                        normal_distortion_std=args.normal_distortion_std,
                        device=device,
                    )
                    target_keypoints_y_distortion = sample_distortions(
                        sample_matches=args.sample_matches,
                        num_keypoints=num_keypoints,
                        max_distortion=args.patch_radius,
                        normal_distortion_std=args.normal_distortion_std,
                        device=device,
                    )

                    # Apply the distortions
                    matched_source_keypoints[:, 0] += source_keypoints_x_distortion
                    matched_source_keypoints[:, 1] += source_keypoints_y_distortion
                    matched_target_keypoints[:, 0] += target_keypoints_x_distortion
                    matched_target_keypoints[:, 1] += target_keypoints_y_distortion

                    # Move to cpu
                    matched_source_keypoints = matched_source_keypoints.cpu().numpy()
                    matched_target_keypoints = matched_target_keypoints.cpu().numpy()
                    matched_source_descriptors = None
                    matched_target_descriptors = None

                if (
                    len(matched_source_keypoints.shape) == 1
                    or matched_source_keypoints.shape[0] == 0
                    or len(matched_target_keypoints.shape) == 1
                    or matched_target_keypoints.shape[0] == 0
                ):
                    # We need at least one match to continue
                    # -> This sample will result in an outlier during val/eval; and is skipped during train
                    matched_source_keypoints = selected_source_keypoints[0].reshape(1, -1)
                    matched_target_keypoints = selected_target_keypoints[0].reshape(1, -1)
                    matched_source_descriptors = selected_source_descriptors[0].reshape(1, -1)
                    matched_target_descriptors = selected_target_descriptors[0].reshape(1, -1)

                if split == "train":
                    num_keypoints = matched_source_keypoints.shape[0]
                    if args.skip_image_pairs_with_less_than_8_matches_for_training_split and num_keypoints < 8:
                        print("Not enough inliers: " + str(num_keypoints) + " skipping sample of the training dataset")
                        continue

                if args.apply_xfeat_refinement:
                    start_time = time.time()
                    for _ in range(1):
                        offsets = keypoint_net.net.fine_matcher(
                            torch.cat(
                                [
                                    torch.tensor(matched_source_descriptors, device="cuda"),
                                    torch.tensor(matched_target_descriptors, device="cuda"),
                                ],
                                dim=-1,
                            )
                        )
                        offsets = keypoint_net.subpix_softmax2d(offsets.view(-1, 8, 8)).cpu().numpy()
                        if selected_source_scales is not None:
                            matched_source_keypoints += offsets * (matched_source_scales[:, None])
                        else:
                            matched_source_keypoints += offsets
                    end_time = time.time()
                    elapsed_ms = (end_time - start_time) * 1000
                    print(f"XFeat refinement took {elapsed_ms:.2f} ms")
                    print(f"Average XFeat refinement time per iteration: {elapsed_ms / 100000:.6f} ms")

                matched_source_keypoints_tensor = torch.tensor(matched_source_keypoints, device=device)
                matched_target_keypoints_tensor = torch.tensor(matched_target_keypoints, device=device)

                # intrinsics
                if "source_camera_matrix" in dataset_samples:
                    source_camera_matrix = dataset_samples["source_camera_matrix"][0]
                else:
                    source_camera_matrix = None
                if "target_camera_matrix" in dataset_samples:
                    target_camera_matrix = dataset_samples["target_camera_matrix"][0]
                else:
                    target_camera_matrix = None

                if "gt_source_to_target_transform" in dataset_samples:
                    # relative camera pose ground truth
                    source_to_target_transform = dataset_samples["gt_source_to_target_transform"][0].cpu().numpy()
                    source_to_target_rotmat = source_to_target_transform[0:3, 0:3]
                    source_to_target_translation = source_to_target_transform[0:3, 3:]

                    smp_error = sampson_dist(
                        gt_rot=torch.from_numpy(source_to_target_rotmat).to(device),
                        gt_t=torch.from_numpy(source_to_target_translation).to(device),
                        k1=source_camera_matrix,
                        k2=target_camera_matrix,
                        x1=matched_source_keypoints_tensor,
                        x2=matched_target_keypoints_tensor,
                    )
                else:
                    source_to_target_rotmat = None
                    source_to_target_translation = None
                    smp_error = None

                _, _, source_image_height, source_image_width = source_image.shape
                _, _, target_image_height, target_image_width = target_image.shape

                # patches
                source_patches = extract_patches_from_map(
                    source_image[0], matched_source_keypoints_tensor, args.patch_radius
                )
                target_patches = extract_patches_from_map(
                    target_image[0], matched_target_keypoints_tensor, args.patch_radius
                )

                # Create per-patch ground truth warping
                # First, create a grid of point coordinates on the source image
                source_point_grid = image_grid(
                    batch_size=1,
                    grid_height=source_image_height,
                    grid_width=source_image_width,
                    dtype=torch.float32,
                    device=device,
                    ones=False,
                    normalized=False,
                )
                source_point_grid_norm = get_normalized_coordinates(
                    source_point_grid, source_image_width, source_image_height
                )
                # Here, coordinates are grid coordinates on the image,
                # e.g. the top left pixel is [-1,-1]. Accordingly, we want align_corners=True (in grid_sample)
                source_points_mask = torch.ones(
                    (source_image.shape[0], source_image_height, source_image_width), device=device, dtype=torch.bool
                )
                if "source_to_target_flow_map" in dataset_samples:
                    if dataset_samples["source_to_target_flow_map"].shape[1] == 3:
                        # If flow map mask is available, we only consider points where the flow map is valid
                        gt_flow_availability_mask = dataset_samples["source_to_target_flow_map"][:, 2, :, :].unsqueeze(
                            dim=1
                        )
                        gt_flow_availability_mask_sampled = (
                            (
                                torch.nn.functional.grid_sample(
                                    gt_flow_availability_mask,
                                    source_point_grid_norm,
                                    mode="nearest",
                                    padding_mode="zeros",
                                    align_corners=True,
                                )
                            )
                            .gt(0.999)
                            .squeeze(dim=1)
                        )
                        source_points_mask = source_points_mask & gt_flow_availability_mask_sampled

                # matched_source_keypoints_tensor has shape [num_matches, 2]
                # Reshape to [2, num_matches]
                matched_source_keypoints_tensor_reshaped = matched_source_keypoints_tensor.permute(1, 0)
                # Reshape to [1, 2, num_matches, 1]
                matched_source_keypoints_tensor_reshaped = matched_source_keypoints_tensor_reshaped.unsqueeze(
                    0
                ).unsqueeze(-1)

                source_patches = source_patches.cpu().numpy().astype(np.float32)
                target_patches = target_patches.cpu().numpy().astype(np.float32)

                data_dict = {
                    "source_keypoints": matched_source_keypoints,
                    "target_keypoints": matched_target_keypoints,
                    "source_image_size": (source_image_height, source_image_width, 3),
                    "target_image_size": (target_image_height, target_image_width, 3),
                    "source_patches": source_patches,
                    "target_patches": target_patches,
                }

                if "source_image_name" in dataset_samples and "target_image_name" in dataset_samples:
                    data_dict["source_image_name"] = dataset_samples["source_image_name"][0]
                    data_dict["target_image_name"] = dataset_samples["target_image_name"][0]
                if matched_source_descriptors is not None and matched_target_descriptors is not None:
                    data_dict["source_descriptors"] = matched_source_descriptors
                    data_dict["target_descriptors"] = matched_target_descriptors
                if source_camera_matrix is not None and target_camera_matrix is not None:
                    source_camera_matrix = source_camera_matrix.cpu().numpy().astype(np.float32)
                    target_camera_matrix = target_camera_matrix.cpu().numpy().astype(np.float32)
                    data_dict["source_camera_matrix"] = source_camera_matrix
                    data_dict["target_camera_matrix"] = target_camera_matrix
                if source_to_target_rotmat is not None and source_to_target_translation is not None:
                    data_dict["source_to_target_rotmat"] = source_to_target_rotmat
                    data_dict["source_to_target_translation"] = source_to_target_translation
                if smp_error is not None:
                    smp_error = smp_error.cpu().numpy().astype(np.float32)
                    data_dict["smp_error"] = smp_error

                if args.extract_score_patches:
                    map_scale = source_score_map.shape[-1] / source_image.shape[-1]
                    map_patch_radius = int(args.patch_radius * map_scale)
                    if map_patch_radius < args.min_patch_radius_of_additional_patches:
                        map_patch_radius = args.min_patch_radius_of_additional_patches
                    map_scale_target_keypoints = matched_target_keypoints_tensor * map_scale
                    map_scale_source_keypoints = matched_source_keypoints_tensor * map_scale

                    source_score_patches = extract_patches_from_map(
                        source_score_map[0],
                        map_scale_source_keypoints,
                        map_patch_radius,
                    )
                    target_score_patches = extract_patches_from_map(
                        target_score_map[0],
                        map_scale_target_keypoints,
                        map_patch_radius,
                    )
                    source_score_patches = source_score_patches.cpu().numpy()
                    target_score_patches = target_score_patches.cpu().numpy()

                    data_dict["source_score_patches"] = source_score_patches
                    data_dict["target_score_patches"] = target_score_patches

                sample_group = dataset_group.create_group(str(index).zfill(10))
                for key in data_dict:
                    sample_group.create_dataset(key, data=data_dict[key])
    hdf5_file.close()


if __name__ == "__main__":
    main()
