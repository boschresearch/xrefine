# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.

from typing import Optional

import torch

from common_modules import Attention, Encoder, MatchScoreHead, ScoreMapHead, SpatialArgmax2d, SpatialSoftArgmax2d
from dataprocess.data_utils import extract_patches_from_map
from utils import color_normalization


# The following class is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class AttnTuner(torch.nn.Module):
    def __init__(
        self,
        output_dim: int = 256,
        use_score: bool = True,
        color_normalization_strategy: str = "orig",
        spatial_argmax_type: str = "soft",
        no_delta_scaling: bool = False,
    ):
        super(AttnTuner, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.feat_axis = 1
        self.use_score = use_score
        self.color_normalization_strategy = color_normalization_strategy
        self.use_color = color_normalization_strategy == "simple_color"
        self.spatial_argmax_type = spatial_argmax_type
        self.no_delta_scaling = no_delta_scaling

        input_channels = 1
        if self.use_color:
            input_channels += 2
        if self.use_score:
            input_channels += 1

        c1, c2, c3 = 16, 64, output_dim
        # patch size -> 11x11
        self.conv1a = torch.nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=0)
        # patch size -> 9x9
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=0)
        # patch size -> 7x7
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=0)
        # patch size -> 5x5

        if self.spatial_argmax_type == "soft":
            self.spatial_argmax = SpatialSoftArgmax2d()
        elif self.spatial_argmax_type == "soft_with_temperature":
            self.spatial_argmax = SpatialSoftArgmax2d(temperature=0.01)
        elif self.spatial_argmax_type == "hard":
            self.spatial_argmax = SpatialArgmax2d()
        else:
            raise ValueError("Unknown spatial argmax type: {}".format(self.spatial_argmax_type))

    def forward(self, patch: torch.Tensor, scorepatch: torch.Tensor, desc: torch.Tensor):
        batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width = patch.shape
        _, _, desc_dim = desc.shape
        assert orig_patch_height == orig_patch_width, "Patch shape must be square"

        # Reshape patch from [batch_size, num_patches, patch_channels, patch_height, patch_width]
        # to [batch_size * num_patches, patch_channels, patch_height, patch_width]
        patch = patch.view(batch_size * num_patches, patch_channels, orig_patch_height, orig_patch_width)
        if self.use_score:
            # Reshape scorepatch from [batch_size, num_patches, 1, patch_height, patch_width]
            # to [batch_size * num_patches, 1, patch_height, patch_width]
            scorepatch = scorepatch.view(batch_size * num_patches, 1, orig_patch_height, orig_patch_width)
        # Reshape desc from [batch_size, num_patches, desc_dim]
        # to [batch_size * num_patches, desc_dim, 1, 1]
        desc = desc.view(batch_size * num_patches, desc_dim, 1, 1)

        patch = color_normalization(patch, self.feat_axis, self.color_normalization_strategy)

        if self.use_score:
            x = torch.cat([patch, scorepatch], self.feat_axis)
        else:
            x = patch

        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.conv3(x)

        _, _, processed_patch_height, processed_patch_width = x.shape
        assert processed_patch_height == processed_patch_width, "Patch shape must stay square"

        # Normalize predicted descriptors
        x = torch.nn.functional.normalize(x, p=2, dim=self.feat_axis)

        cos_sim = (x * desc).sum(dim=self.feat_axis)  # Cosine similarity (in [-1, 1])
        cos_sim = cos_sim.unsqueeze(dim=1)

        cos_sim = cos_sim.view(batch_size, num_patches, processed_patch_height, processed_patch_width)
        max_coord = self.spatial_argmax(cos_sim)

        # max_coord values are in [0, processed_patch_height-1] (0.5 margin at each border cannot be reached)
        coord = max_coord - (processed_patch_height - 1) / 2.0
        # coord values are in [-(processed_patch_height-1)/2, (processed_patch_height-1)/2]; 0 at center of pixel center

        if not self.no_delta_scaling:
            # reference coordinate to (smaller) patch center;
            coord = (orig_patch_height // 2) / (processed_patch_height // 2) * coord  # scaling: see paper chapter 4

        x = x.view(batch_size, num_patches, -1, processed_patch_height, processed_patch_width)
        cos_sim = cos_sim.unsqueeze(dim=2)

        return {
            "delta": coord,  # batch_size x num_patches x 2,
            "similarity": cos_sim,  # batch_size x num_patches x 1 x processed_patch_height x processed_patch_width
            "descr": x,  # batch_size x num_patches x desc_dim x processed_patch_height x processed_patch_width
        }


# The following class is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class JointAttnTuner(torch.nn.Module):
    def __init__(
        self,
        desc_dim: int,
        use_score: bool,
        color_normalization_strategy: str,
        spatial_argmax_type: str,
        attn_with_desc: bool,
        attn_with_avg_desc: bool,
        attn_with_patch: bool,
        num_attention_blocks: int,
        positional_encoding_type: str,
        attn_layer_norm: bool,
        attn_skip_connection: bool,
        with_match_score: bool,
        directly_infer_score_map: bool,
        patch_radius: int,
        skip_encoder: bool,
        encoder_variant: str,
        adjust_only_second_keypoint: bool,
    ):
        super(JointAttnTuner, self).__init__()
        self.feat_axis = 1
        self.normalized_coordinates = False
        self.desc_dim = desc_dim
        self.use_score = use_score
        self.color_normalization_strategy = color_normalization_strategy
        self.use_color = color_normalization_strategy == "simple_color"
        self.spatial_argmax_type = spatial_argmax_type
        self.attn_with_desc = attn_with_desc
        self.attn_with_avg_desc = attn_with_avg_desc
        self.attn_with_patch = attn_with_patch
        self.num_attention_blocks = num_attention_blocks
        self.with_match_score = with_match_score
        self.directly_infer_score_map = directly_infer_score_map
        self.skip_encoder = skip_encoder
        self.adjust_only_second_keypoint = adjust_only_second_keypoint

        patch_size = patch_radius * 2 + 1
        if not self.skip_encoder:
            self.encoder = Encoder(self.desc_dim, use_score, self.use_color, encoder_variant)
            processed_patch_size = patch_size - self.encoder.patch_size_reduction
        else:
            processed_patch_size = patch_size

        if self.spatial_argmax_type == "soft":
            self.spatial_argmax = SpatialSoftArgmax2d()
        elif self.spatial_argmax_type == "soft_with_temperature":
            self.spatial_argmax = SpatialSoftArgmax2d(temperature=0.01)
        elif self.spatial_argmax_type == "hard":
            self.spatial_argmax = SpatialArgmax2d()
        else:
            raise ValueError("Unknown spatial argmax type: {}".format(self.spatial_argmax_type))

        # match score head for binary classification correct / incorrect match
        if self.with_match_score:
            self.match_score_head = MatchScoreHead(
                patch_size=processed_patch_size, in_channels=2 * self.desc_dim + 2, reduced_channels=8
            )

        if self.directly_infer_score_map:
            self.score_map_head = ScoreMapHead(in_channels=self.desc_dim)

        if self.attn_with_patch:
            self.cross_attn_patch_blocks = torch.nn.ModuleList(
                [
                    Attention(
                        feature_dim=self.desc_dim,
                        num_attention_heads=4,
                        positional_encoding_type=positional_encoding_type,
                        skip_connection=attn_skip_connection,
                        apply_layer_norm=attn_layer_norm,
                        seq_len_query=processed_patch_size * processed_patch_size,
                        seq_len_key_value=processed_patch_size * processed_patch_size,
                    )
                    for _ in range(num_attention_blocks)
                ]
            )

        if self.attn_with_desc or self.attn_with_avg_desc:
            if self.attn_with_desc:
                seq_len_key_value = 2
            else:
                seq_len_key_value = 1

            self.cross_attn_desc_blocks = torch.nn.ModuleList(
                [
                    Attention(
                        feature_dim=self.desc_dim,
                        num_attention_heads=4,
                        positional_encoding_type=positional_encoding_type,
                        skip_connection=attn_skip_connection,
                        apply_layer_norm=attn_layer_norm,
                        seq_len_query=processed_patch_size * processed_patch_size,
                        seq_len_key_value=seq_len_key_value,
                    )
                    for _ in range(num_attention_blocks)
                ]
            )

    def forward(
        self,
        patch1: torch.Tensor,  # shape [batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width]
        patch2: torch.Tensor,  # shape [batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width]
        scorepatch1: Optional[torch.Tensor],
        scorepatch2: Optional[torch.Tensor],
        desc1: Optional[torch.Tensor],  # shape [batch_size, num_patches, desc_dim]
        desc2: Optional[torch.Tensor],  # shape [batch_size, num_patches, desc_dim]
    ):
        # data preparation
        batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width = patch1.shape
        assert orig_patch_height == orig_patch_width, "Patch shape must be square"

        patch1 = patch1.view(batch_size * num_patches, patch_channels, orig_patch_height, orig_patch_width)
        patch2 = patch2.view(batch_size * num_patches, patch_channels, orig_patch_height, orig_patch_width)

        if not self.directly_infer_score_map or self.attn_with_desc or self.attn_with_avg_desc:
            desc1 = desc1.view(batch_size * num_patches, self.desc_dim, 1, 1)
            desc2 = desc2.view(batch_size * num_patches, self.desc_dim, 1, 1)
            avg_desc = (desc1 + desc2) / 2.0

        if not self.skip_encoder:
            if self.use_score:
                scorepatch1 = scorepatch1.view(batch_size * num_patches, 1, orig_patch_height, orig_patch_width)
                scorepatch2 = scorepatch2.view(batch_size * num_patches, 1, orig_patch_height, orig_patch_width)

            patch1 = color_normalization(patch1, self.feat_axis, self.color_normalization_strategy)
            patch2 = color_normalization(patch2, self.feat_axis, self.color_normalization_strategy)

            x1 = torch.cat([patch1, scorepatch1], self.feat_axis) if self.use_score else patch1
            x2 = torch.cat([patch2, scorepatch2], self.feat_axis) if self.use_score else patch2

            # Shared Encoder
            x1 = self.encoder(x1)
            x2 = self.encoder(x2)
        else:
            x1 = patch1
            x2 = patch2

        if not self.directly_infer_score_map:
            # Normalize predicted descriptors
            x1 = torch.nn.functional.normalize(x1, p=2, dim=self.feat_axis)
            x2 = torch.nn.functional.normalize(x2, p=2, dim=self.feat_axis)

        _, _, processed_patch_height, processed_patch_width = x1.shape
        assert processed_patch_height == processed_patch_width, "Patch shape must stay square"

        embedding_1 = x1.view(batch_size, num_patches, -1, processed_patch_height, processed_patch_width)
        embedding_2 = x2.view(batch_size, num_patches, -1, processed_patch_height, processed_patch_width)

        # Apply optional attention
        if self.attn_with_patch or self.attn_with_desc or self.attn_with_avg_desc:
            seq_len = processed_patch_height * processed_patch_width
            # reshape x1 and x2 from [batch_size * num_patches, desc_dim, processed_patch_height, processed_patch_width]
            #   to [batch_size * num_patches, desc_dim, seq_len]
            x1 = x1.view(batch_size * num_patches, self.desc_dim, seq_len)
            x2 = x2.view(batch_size * num_patches, self.desc_dim, seq_len)

            if self.attn_with_desc:
                attended_desc = torch.cat([desc1, desc2], dim=2)
                # attended_desc has shape [batch_size * num_patches, desc_dim, 2, 1]
                # reshape attended_desc to [batch_size * num_patches, desc_dim, 2]
                attended_desc = attended_desc.view(batch_size * num_patches, self.desc_dim, 2)
            elif self.attn_with_avg_desc:
                attended_desc = avg_desc
                # attended_desc has shape [batch_size * num_patches, desc_dim, 1, 1]
                # reshape attended_desc to [batch_size * num_patches, desc_dim, 1]
                attended_desc = attended_desc.view(batch_size * num_patches, self.desc_dim, 1)

            for block_index in range(self.num_attention_blocks):
                # optional cross-attention between x1 and x2
                if self.attn_with_patch:
                    x1_updated = self.cross_attn_patch_blocks[block_index](query=x1, key_value=x2)
                    x2_updated = self.cross_attn_patch_blocks[block_index](query=x2, key_value=x1)
                    x1 = x1_updated
                    x2 = x2_updated

                # optional cross-attention between x1, x2 and descriptors
                if self.attn_with_desc or self.attn_with_avg_desc:
                    x1 = self.cross_attn_desc_blocks[block_index](query=x1, key_value=attended_desc)
                    x2 = self.cross_attn_desc_blocks[block_index](query=x2, key_value=attended_desc)

            x1 = x1.view(batch_size * num_patches, self.desc_dim, processed_patch_height, processed_patch_width)
            x2 = x2.view(batch_size * num_patches, self.desc_dim, processed_patch_height, processed_patch_width)

        embedding_after_attention_1 = x1.view(
            batch_size, num_patches, -1, processed_patch_height, processed_patch_width
        )
        embedding_after_attention_2 = x2.view(
            batch_size, num_patches, -1, processed_patch_height, processed_patch_width
        )

        if self.directly_infer_score_map:
            cos_sim2 = self.score_map_head(x2)  # Score (in [-1, 1])
            if not self.adjust_only_second_keypoint or self.with_match_score:
                cos_sim1 = self.score_map_head(x1)  # Score (in [-1, 1])
        else:
            cos_sim2 = (x2 * avg_desc).sum(dim=self.feat_axis)  # Cosine similarity (in [-1, 1])
            cos_sim2 = cos_sim2.unsqueeze(dim=1)
            if not self.adjust_only_second_keypoint or self.with_match_score:
                cos_sim1 = (x1 * avg_desc).sum(dim=self.feat_axis)  # Cosine similarity (in [-1, 1])
                cos_sim1 = cos_sim1.unsqueeze(dim=1)

        if self.with_match_score:
            # concatenate extracted patch features and cosine similarity
            inp = torch.cat([x1, cos_sim1, x2, cos_sim2], dim=1)
            match_score = self.match_score_head(inp)  # shape [batch_size * num_patches, 1]
            match_score = match_score.view(batch_size, num_patches)
        else:
            match_score = None

        # get pixel shift as maximum of similarity
        cos_sim2 = cos_sim2.view(batch_size, num_patches, processed_patch_height, processed_patch_width)
        max_coord2 = self.spatial_argmax(cos_sim2)  # values between 0 and processed_patch_height-1
        # shift and rescale coordinates
        coord2 = max_coord2 - (processed_patch_height - 1) / 2.0
        # coord values are in [-(processed_patch_height-1)/2, (processed_patch_height-1)/2]; 0 at center of pixel center
        coord2 = (orig_patch_height // 2) / (processed_patch_height // 2) * coord2  # scaling: see paper chapter 4
        cos_sim2 = cos_sim2.unsqueeze(dim=2)

        if not self.adjust_only_second_keypoint:
            cos_sim1 = cos_sim1.view(batch_size, num_patches, processed_patch_height, processed_patch_width)
            max_coord1 = self.spatial_argmax(cos_sim1)
            coord1 = max_coord1 - (processed_patch_height - 1) / 2.0
            coord1 = (orig_patch_height // 2) / (processed_patch_height // 2) * coord1
            cos_sim1 = cos_sim1.unsqueeze(dim=2)
        else:
            cos_sim1 = torch.zeros_like(cos_sim2)
            coord1 = torch.zeros_like(coord2)

        x1 = x1.view(batch_size, num_patches, -1, processed_patch_height, processed_patch_width)
        x2 = x2.view(batch_size, num_patches, -1, processed_patch_height, processed_patch_width)

        output = {
            "delta1": coord1,
            "similarity1": cos_sim1,
            "descr1": x1,
            "embedding1": embedding_1,
            "embedding_after_attention1": embedding_after_attention_1,
            "delta2": coord2,
            "similarity2": cos_sim2,
            "descr2": x2,
            "embedding2": embedding_2,
            "embedding_after_attention2": embedding_after_attention_2,
            "match_score": match_score,
        }

        return output


# The following class is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class SimpleJointAttnTuner(torch.nn.Module):
    def __init__(
        self,
        desc_dim: int,
        color_normalization_strategy: str,
        spatial_argmax_type: str,
        attn_with_patch: bool,
        num_attention_blocks: int,
        positional_encoding_type: str,
        attn_layer_norm: bool,
        attn_skip_connection: bool,
        patch_radius: int,
        encoder_variant: str,
        adjust_only_second_keypoint: bool,
        image_values_are_normalized: bool,
    ):
        """
        This is a simplified version of JointAttnTuner with less experimental options.
        """
        super(SimpleJointAttnTuner, self).__init__()
        self.feat_axis = 1
        self.normalized_coordinates = False
        self.desc_dim = desc_dim
        self.color_normalization_strategy = color_normalization_strategy
        self.use_color = color_normalization_strategy == "simple_color"
        self.spatial_argmax_type = spatial_argmax_type
        self.attn_with_patch = attn_with_patch
        self.num_attention_blocks = num_attention_blocks
        self.adjust_only_second_keypoint = adjust_only_second_keypoint
        self.image_values_are_normalized = image_values_are_normalized

        patch_size = patch_radius * 2 + 1
        self.encoder = Encoder(self.desc_dim, False, self.use_color, encoder_variant)
        processed_patch_size = patch_size - self.encoder.patch_size_reduction

        if self.spatial_argmax_type == "soft":
            self.spatial_argmax = SpatialSoftArgmax2d()
        elif self.spatial_argmax_type == "soft_with_temperature":
            self.spatial_argmax = SpatialSoftArgmax2d(temperature=0.01)
        elif self.spatial_argmax_type == "hard":
            self.spatial_argmax = SpatialArgmax2d()
        else:
            raise ValueError("Unknown spatial argmax type: {}".format(self.spatial_argmax_type))

        self.score_map_head = ScoreMapHead(in_channels=self.desc_dim)

        if self.attn_with_patch:
            self.cross_attn_patch_blocks = torch.nn.ModuleList(
                [
                    Attention(
                        feature_dim=self.desc_dim,
                        num_attention_heads=4,
                        positional_encoding_type=positional_encoding_type,
                        skip_connection=attn_skip_connection,
                        apply_layer_norm=attn_layer_norm,
                        seq_len_query=processed_patch_size * processed_patch_size,
                        seq_len_key_value=processed_patch_size * processed_patch_size,
                    )
                    for _ in range(num_attention_blocks)
                ]
            )

    def forward(
        self,
        patch1: torch.Tensor,  # shape [batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width]
        patch2: torch.Tensor,  # shape [batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width]
    ):
        if not self.image_values_are_normalized:
            # Bring patches to [0, 1]
            patch1 = patch1 / 255.0
            patch2 = patch2 / 255.0

        # data preparation
        batch_size, num_patches, patch_channels, orig_patch_height, orig_patch_width = patch1.shape

        patch1 = patch1.view(batch_size * num_patches, patch_channels, orig_patch_height, orig_patch_width)
        patch2 = patch2.view(batch_size * num_patches, patch_channels, orig_patch_height, orig_patch_width)

        patch1 = color_normalization(patch1, self.feat_axis, self.color_normalization_strategy)
        patch2 = color_normalization(patch2, self.feat_axis, self.color_normalization_strategy)

        # Shared Encoder
        x1 = self.encoder(patch1)
        x2 = self.encoder(patch2)

        _, _, processed_patch_height, processed_patch_width = x1.shape

        # Apply optional attention
        if self.attn_with_patch:
            seq_len = processed_patch_height * processed_patch_width
            # reshape x1 and x2 from [batch_size * num_patches, desc_dim, processed_patch_height, processed_patch_width]
            #   to [batch_size * num_patches, desc_dim, seq_len]
            x1 = x1.view(batch_size * num_patches, self.desc_dim, seq_len)
            x2 = x2.view(batch_size * num_patches, self.desc_dim, seq_len)

            for block_index in range(self.num_attention_blocks):
                # optional cross-attention between x1 and x2
                if self.attn_with_patch:
                    x1_updated = self.cross_attn_patch_blocks[block_index](query=x1, key_value=x2)
                    x2_updated = self.cross_attn_patch_blocks[block_index](query=x2, key_value=x1)
                    x1 = x1_updated
                    x2 = x2_updated

            x1 = x1.view(batch_size * num_patches, self.desc_dim, processed_patch_height, processed_patch_width)
            x2 = x2.view(batch_size * num_patches, self.desc_dim, processed_patch_height, processed_patch_width)

        cos_sim2 = self.score_map_head(x2)  # Score (in [-1, 1])
        if not self.adjust_only_second_keypoint:
            cos_sim1 = self.score_map_head(x1)  # Score (in [-1, 1])

        # get pixel shift as maximum of similarity
        cos_sim2 = cos_sim2.view(batch_size, num_patches, processed_patch_height, processed_patch_width)
        max_coord2 = self.spatial_argmax(cos_sim2)  # values between 0 and processed_patch_height-1
        # shift and rescale coordinates
        coord2 = max_coord2 - (processed_patch_height - 1) / 2.0
        # coord values are in [-(processed_patch_height-1)/2, (processed_patch_height-1)/2]; 0 at center of pixel center
        coord2 = (orig_patch_height // 2) / (processed_patch_height // 2) * coord2  # scaling: see paper chapter 4

        if not self.adjust_only_second_keypoint:
            cos_sim1 = cos_sim1.view(batch_size, num_patches, processed_patch_height, processed_patch_width)
            max_coord1 = self.spatial_argmax(cos_sim1)
            coord1 = max_coord1 - (processed_patch_height - 1) / 2.0
            coord1 = (orig_patch_height // 2) / (processed_patch_height // 2) * coord1
        else:
            coord1 = torch.zeros_like(coord2)

        return coord1, coord2


class XRefine(torch.nn.Module):
    def __init__(
        self,
        variant: str = "small",
        adjust_only_second_keypoint: bool = False,
        image_values_are_normalized: bool = True,
    ):
        super(XRefine, self).__init__()

        self.patch_radius = 5
        self.adjust_only_second_keypoint = adjust_only_second_keypoint

        if variant == "small":
            num_attention_blocks = 1
        elif variant == "large":
            num_attention_blocks = 3
        else:
            raise ValueError("Unknown encoder variant: {}".format(variant))

        self.net = SimpleJointAttnTuner(
            desc_dim=64,
            color_normalization_strategy="orig",
            spatial_argmax_type="soft",
            attn_with_patch=True,
            num_attention_blocks=num_attention_blocks,
            positional_encoding_type="learnable",
            attn_layer_norm=False,
            attn_skip_connection=True,
            patch_radius=self.patch_radius,
            encoder_variant=variant,
            adjust_only_second_keypoint=self.adjust_only_second_keypoint,
            image_values_are_normalized=image_values_are_normalized,
        )

    def forward(self, keypoints1: torch.Tensor, keypoints2: torch.Tensor, image1: torch.Tensor, image2: torch.Tensor):
        """Refines keypoint coordinates using attention-based refinement.

        Args:
            keypoints1 (torch.Tensor):
                Matched keypoint coordinates for the first image, shape [num_keypoints, 2].
            keypoints2 (torch.Tensor):
                Matched keypoint coordinates for the second image, shape [num_keypoints, 2].
            image1 (torch.Tensor):
                Image tensor for the first image, shape [channels, height, width].
            image2 (torch.Tensor):
                Image tensor for the second image, shape [channels, height, width].
            image_values_are_normalized (bool):
                If True, image values are assumed to be in [0, 1]. If False, they are assumed to be in [0, 255].
        """
        if len(image1.shape) == 4:
            assert image1.shape[0] == 1, "Only batch size of 1 is supported"
            image1 = image1.squeeze(0)
        if len(image2.shape) == 4:
            assert image2.shape[0] == 1, "Only batch size of 1 is supported"
            image2 = image2.squeeze(0)
        assert (
            len(keypoints1.shape) == 2
            and keypoints1.shape[1] == 2
            and len(keypoints2.shape) == 2
            and keypoints2.shape[1] == 2
        ), "Keypoints must have shape [num_keypoints, 2]"

        with torch.no_grad():
            patches1 = extract_patches_from_map(
                image=image1, keypoints=keypoints1, patch_radius=self.patch_radius
            ).unsqueeze(dim=0)
            patches2 = extract_patches_from_map(
                image=image2, keypoints=keypoints2, patch_radius=self.patch_radius
            ).unsqueeze(dim=0)
            # patches shape is [1, num_patches, patch_channels, patch_height, patch_width]

            # XRefine expects keypoints to be located at the pixel center
            if not self.adjust_only_second_keypoint:
                # Only shift pts1 to pixel center, if both keypoints are updated
                keypoints1 = torch.floor(keypoints1) + 0.5
            keypoints2 = torch.floor(keypoints2) + 0.5

            coord1, coord2 = self.net(patches1, patches2)

            coord1 = coord1.squeeze(dim=0)
            coord2 = coord2.squeeze(dim=0)

        return keypoints1 + coord1, keypoints2 + coord2
