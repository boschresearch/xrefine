# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch
from torch.utils.data import Dataset


class MegaDepthGlueFactoryDataset(Dataset):
    """
    MegaDepth dataset class that uses GlueFactory loader.

    Parameters
    ----------
    split_info_type : str
        Type of split to use. Can be "training", "validation" or "evaluation".
    """

    def __init__(
        self,
        split_info_type: str,
    ):
        super().__init__()

        try:
            from gluefactory.datasets import get_dataset as gf_get_dataset
            from omegaconf import OmegaConf
        except ImportError:
            error_msg = "Cannot find glue-factory, please clone the repo from "
            error_msg += "https://github.com/cvg/glue-factory"
            error_msg += " to ./submodules to use the MegaDepthGlueFactoryDataset."
            raise ImportError(error_msg)
        conf_dict = {
            "name": "megadepth",
            "preprocessing": {"resize": 1024, "side": "long", "square_pad": True},
            "train_split": "train_scenes_clean.txt",
            "train_num_per_scene": 300,
            "val_split": "valid_scenes_clean.txt",
            "val_pairs": "valid_pairs.txt",
            "test_split": "test_scenes_clean.txt",
            "min_overlap": 0.1,
            "max_overlap": 0.7,
            "num_overlap_bins": 3,
            "read_depth": True,
            "read_image": True,
            "grayscale": False,
        }
        dataconf = OmegaConf.create(conf_dict)
        if split_info_type == "training":
            variant = "train"
        elif split_info_type == "validation":
            variant = "val"
        elif split_info_type == "evaluation":
            variant = "test"
        else:
            raise ValueError(f"Unknown split_info_type for MegaDepthGlueFactory: {split_info_type}")
        self.dataset = gf_get_dataset(dataconf.name)(dataconf).get_dataset(variant)
        self.samples = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_idx = self.samples[idx]
        gf_dataset_samples = self.dataset[sample_idx]

        source_image_width, source_image_height = gf_dataset_samples["view0"]["image_size"]  # W, H
        target_image_width, target_image_height = gf_dataset_samples["view1"]["image_size"]  # W, H

        dataset_samples = {}
        dataset_samples["index"] = sample_idx
        dataset_samples["source_image"] = gf_dataset_samples["view0"]["image"]
        dataset_samples["target_image"] = gf_dataset_samples["view1"]["image"]
        dataset_samples["source_depth_map"] = gf_dataset_samples["view0"]["depth"].unsqueeze(0)
        dataset_samples["target_depth_map"] = gf_dataset_samples["view1"]["depth"].unsqueeze(0)
        dataset_samples["source_camera_matrix"] = gf_dataset_samples["view0"]["camera"].calibration_matrix()
        dataset_samples["target_camera_matrix"] = gf_dataset_samples["view1"]["camera"].calibration_matrix()
        dataset_samples["source_camera_model"] = "pinhole"
        dataset_samples["target_camera_model"] = "pinhole"
        dataset_samples["source_image_width"] = source_image_width
        dataset_samples["source_image_height"] = source_image_height
        dataset_samples["target_image_width"] = target_image_width
        dataset_samples["target_image_height"] = target_image_height

        source_to_target_rotmat = gf_dataset_samples["T_0to1"].R.cpu().numpy()
        source_to_target_translation = gf_dataset_samples["T_0to1"].t.unsqueeze(-1).cpu().numpy()

        gt_source_to_target_transform = np.zeros((4, 4))
        gt_source_to_target_transform[0:3, 0:3] = source_to_target_rotmat
        gt_source_to_target_transform[0:3, 3:] = source_to_target_translation
        gt_source_to_target_transform[3, 3] = 1
        gt_source_to_target_transform = torch.from_numpy(gt_source_to_target_transform)
        gt_source_to_target_transform = gt_source_to_target_transform

        dataset_samples["target_camera_pose"] = gt_source_to_target_transform
        dataset_samples["source_camera_pose"] = torch.eye(4, dtype=torch.float64)
        target_file_extension = "." + gf_dataset_samples["name"].split(".")[-1]
        source_file_extension = "." + gf_dataset_samples["name"].split(".")[1].split("_")[0]
        scene = gf_dataset_samples["name"].split("/")[0]
        image_names = gf_dataset_samples["name"].split("/")[1]
        dataset_samples["source_image_name"] = (
            scene + "/images/" + image_names.split(source_file_extension)[0] + source_file_extension
        )
        dataset_samples["target_image_name"] = (
            scene
            + "/images/"
            + image_names.split(source_file_extension)[1].split(target_file_extension)[0][1:]
            + target_file_extension
        )

        # Values between 0.0 and 1.0
        dataset_samples["view_selection_score"] = gf_dataset_samples["overlap_0to1"]

        return dataset_samples
