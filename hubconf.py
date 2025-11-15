# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

dependencies = ["torch"]
import torch

from model import XRefine as XRefine_


def XRefine(
    pretrained: bool = True,
    detector: str = "general",
    variant: str = "small",
    adjust_only_second_keypoint: bool = False,
    image_values_are_normalized: bool = True,
) -> XRefine_:
    """Class factory for the XRefine model.

    Args:
        pretrained (bool):
            Whether to load pretrained weights.
            Defaults to True.
        detector (str):
            Whether to use the general model or a detector specific variant.
            Options: 'general', 'aliked', 'dedode', 'dedodev2', 'disk', 'r2d2, 'sift', 'splg', 'spnn', 'xfeat',
            'xfeat-star'.
            Defaults to 'general'.
        variant (str):
            Model variant, either 'small' or 'large'.
            Defaults to 'small'.
        adjust_only_second_keypoint (bool):
            Whether to adjust only the second keypoint.
            If True, only the second keypoint will be adjusted, keeping the reference keypoint fixed.
            This option is useful for optimization of keypoint tracks, e.g. in a SfM.
            Defaults to False.
        image_values_are_normalized (bool):
            Whether the input images are normalized to [0, 1] range.
            If True, the input images should be in the range [0, 1].
            If False, the input images should be in the range [0, 255].
            Defaults to True.

    Returns:
        XRefine_ model instance.

    Raises:
        AssertionError: If the variant or detector is not recognized.
        AssertionError: If the detector is not 'general' and variant is 'large'.
        AssertionError: If the detector is not 'general' and adjust_only_second_keypoint is True.
    """
    assert variant in ["small", "large"], f"Unknown variant: {variant}"
    assert detector in [
        "general",
        "aliked",
        "dedode",
        "dedodev2",
        "disk",
        "r2d2",
        "sift",
        "splg",
        "spnn",
        "xfeat",
        "xfeat-star",
    ], f"Unknown detector: {detector}"
    assert detector == "general" or variant != "large", "Large variant is only supported for the 'general' case."
    assert (
        detector == "general" or not adjust_only_second_keypoint
    ), "Adjusting only the second keypoint is only supported for the 'general' case."

    model = XRefine_(variant, adjust_only_second_keypoint, image_values_are_normalized)

    weights = None
    if pretrained:
        weights_file_name = "xrefine_" + variant
        if detector != "general":
            weights_file_name += "_specific"
        weights_file_name += "_" + detector
        if adjust_only_second_keypoint:
            weights_file_name += "_adjust_only_second_keypoint"
        weights_file_name += ".pth"
        weights_file_path = "pretrained/" + weights_file_name

        # For local loading of weights, uncomment the following lines.
        # weights = torch.load(weights_file_path, weights_only=False)
        # print("Loaded model from", weights_file_path)

        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/boschresearch/xrefine/blob/main/" + weights_file_path,
            map_location=torch.device("cpu"),
        )
        model.net.load_state_dict(weights["model"])

    return model
