# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import time
from pathlib import Path
from typing import Dict

import numpy as np
from pixsfm.refine_colmap import PixSfM
from pixsfm.util.hloc import Map_NameKeypoints


def refine_pixsfm(matches: np.array, image_list: list[str], image_dir: str, match_scores=None):
    """
    Refine the matches using the PixSFM refinement process.

    Args:
        matches (np.array, N x 4): The initial matches to be refined. The first two columns are the KP coordinates
                                    of the first image, the remaining two columns are the KP coordinates of the second image.
        image_list (list[str]): List of image names (should contain exactly two for pairwise matching).
        image_dir (str): Directory containing the images.
        match_scores (optional): Scores for the matches. Not yet implemented.

    Returns:
        refined_matches (np.array, N x 4): The refined matches.
    """
    assert len(image_list) == 2, "Function currently supports only a single pair of images"

    image1, image2 = image_list

    # Build keypoints dictionary
    keypoints_dict: Dict[str, np.ndarray] = {
        image1: matches[:, :2],  # KP coords in first image
        image2: matches[:, 2:],  # KP coords in second image
    }

    # Convert keypoints to Map_NameKeypoints format
    keypoints = Map_NameKeypoints()
    for name, kpts in keypoints_dict.items():
        keypoints[name] = kpts

    # PixSfM expects matches as index pairs, not coordinates, so we use np.arange
    # convert to numpy.uint64 for compatibility
    num_matches = matches.shape[0]
    match_indices = (
        np.vstack([np.arange(num_matches), np.arange(num_matches)]).astype(np.uint64).transpose()
    )  # shape: N x 2

    # Matches_scores tuple: (List[np.ndarray], List[Optional[np.ndarray]])
    matches_scores = ([match_indices], [None])  # We pass None for scores as they're not used

    # List of image pairs
    pairs = [(image1, image2)]

    # Get name of dataset because the will have to be resized differently within PixSfM
    if "megadepth" in image_dir:
        dataset = "megadepth"
    elif "scannet1500" in image_dir:
        dataset = "scannet1500"
    elif "KITTI" in image_dir:
        dataset = "KITTI"
    else:
        raise ValueError("Unknown dataset. Please provide a valid dataset (megadepth, scannet1500, KITTI).")

    # Run refinement
    pixsfm_refinement = PixSfM(dataset=dataset)

    start_time = time.time()
    refined_keypoints, ka_data, feature_manager, pixsfm_internal_refinement_time_wo_s2dnet = pixsfm_refinement.run_ka(
        keypoints=keypoints,
        image_dir=Path(image_dir),
        pairs=pairs,
        matches_scores=matches_scores,
        cache_path=None,
        feature_manager=None,
    )
    pixsfm_internal_refinement_time = time.time() - start_time

    # Get refined keypoints using match indices
    refined_kp1 = refined_keypoints[image1][match_indices[:, 0]]
    refined_kp2 = refined_keypoints[image2][match_indices[:, 1]]
    refined_matches = np.hstack([refined_kp1, refined_kp2])  # shape: N x 4

    return refined_matches, pixsfm_internal_refinement_time, pixsfm_internal_refinement_time_wo_s2dnet
