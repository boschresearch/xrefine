# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.

from typing import Optional

import torch

from utils import create_meshgrid, retrieve_sub_pixel_precision


# The following class is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class SpatialSoftArgmax2d(torch.nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        temperature (float): input tensor is divided by the temperature before the softmax.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature = temperature
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}".format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape

        x = input.view(batch_size, channels, height * width)

        # compute softmax with max substraction trick
        exp_x = torch.exp((x - torch.max(x, dim=-1, keepdim=True)[0]) / self.temperature)
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, False)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = torch.sum((pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x = torch.sum((pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output = torch.cat([expected_x, expected_y], dim=-1)
        output = output.view(batch_size, channels, 2)  # BxNx2
        return output


class SpatialArgmax2d(torch.nn.Module):
    """
    Creates a module that computes the Spatial Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`
    """

    def __init__(self) -> None:
        super(SpatialArgmax2d, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}".format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))

        batch_size, num_patches, height, width = input.shape

        # Get the peaks
        input_flat = input.view(batch_size, num_patches, height * width)
        max_idx = torch.argmax(input_flat, dim=-1)
        max_x = max_idx % width
        max_y = max_idx // width
        max_coordinates = torch.stack((max_x, max_y), dim=-1)

        # Reshape the input from [batch_size, num_patches, height, width] to [batch_size x num_patches, height, width]
        input = input.view(batch_size * num_patches, height, width)

        # Reshape max_coordinates from [batch_size, num_patches, 2] to [batch_size x num_patches, 1, 2]
        max_coordinates = max_coordinates.view(batch_size * num_patches, 1, 2)

        sub_pixel_coordinates = retrieve_sub_pixel_precision(
            input,
            max_coordinates,
            origin_is_center_of_first_pixel=True,
            input_keypoints_format_is_y_x=False,
            add_border=True,
        )

        # Reshape the sub-pixel coordinates from [batch_size x num_patches, 1, 2] to [batch_size, num_patches, 2]
        sub_pixel_coordinates = sub_pixel_coordinates.view(batch_size, num_patches, 2)
        return sub_pixel_coordinates


class Encoder(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        use_score: bool,
        use_color: bool,
        variant: str = "small",
    ):
        super(Encoder, self).__init__()
        self.variant = variant
        input_channels = 1
        if use_color:
            input_channels += 2
        if use_score:
            input_channels += 1
        self.relu = torch.nn.ReLU(inplace=True)
        if self.variant == "large":
            c1, c2, c3 = 16, 64, output_dim
            # There are three conv2d without padding, so the patch size is reduced by 2*3=6
            self.patch_size_reduction = 6
            self.conv1a = torch.nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=0)
            self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
            self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=0)
            self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=0)
        elif self.variant == "small":
            c1, c2, c3 = 16, 64, output_dim
            # There are four conv2d without padding, so the patch size is reduced by 2*4=8
            self.patch_size_reduction = 8
            self.conv1a = torch.nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=0)
            self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=0)
            self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=0)
            self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=0)
        else:
            raise ValueError(f"Unknown encoder variant: {self.variant}. Supported variants are 'small' and 'large'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.conv3(x)
        return x


class ScoreMapHead(torch.nn.Module):
    """
    Score map head which takes extracted patch features and outputs a score map.
    """

    def __init__(self, in_channels: int):
        super(ScoreMapHead, self).__init__()
        self.in_channels = in_channels
        if self.in_channels > 1:
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv = torch.nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the output of the encoder, it has not yet been passed through the ReLU activation function
        if self.in_channels > 1:
            x = self.relu(x)
            x = self.conv(x)
        x = torch.tanh(x)  # Output in range [-1, 1]
        return x


class MatchScoreHead(torch.nn.Module):
    """
    Match score head which takes extracted patch features and cosine similarity as input and outputs a match score.
    Can be trained with binary cross entropy (BCE) loss.
    """

    def __init__(self, patch_size: int, in_channels: int = 64, reduced_channels: int = 8):
        super(MatchScoreHead, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=reduced_channels, kernel_size=1)
        # Output: batch_size x reduced_channels x patch_size x patch_size
        self.fc1 = torch.nn.Linear(patch_size * patch_size * reduced_channels, reduced_channels)
        # Fully connected layer (intermediate representation)
        self.fc2 = torch.nn.Linear(reduced_channels, 1)  # Binary classification output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1x1(x))  # Reduce channel dimensions
        x = x.view(x.shape[0], -1)  # Flatten (batch_size, patch_size * patch_size * reduced_channels)
        x = torch.nn.functional.relu(self.fc1(x))  # Intermediate representation (could be removed for efficiency)
        x = torch.sigmoid(self.fc2(x))  # Binary classification output
        return x


class Attention(torch.nn.Module):
    """
    Multihead attention wrapper.
    """

    def __init__(
        self,
        feature_dim: int,
        num_attention_heads: int,
        positional_encoding_type: str,
        skip_connection: bool,
        apply_layer_norm: bool,
        seq_len_query: Optional[int] = None,
        seq_len_key_value: Optional[int] = None,
    ):
        super(Attention, self).__init__()
        self.positional_encoding_type = positional_encoding_type
        self.skip_connection = skip_connection
        self.apply_layer_norm = apply_layer_norm

        self.multihead_attn = torch.nn.MultiheadAttention(feature_dim, num_attention_heads, batch_first=True, bias=True)

        if self.positional_encoding_type == "sinusoidal":
            self.sinusoidal_encodings = {}
        elif self.positional_encoding_type == "learnable":
            if seq_len_query is None:
                raise ValueError("seq_len_query must be provided when using learnable positional encodings.")
            self.learned_positional_encoding_query = torch.nn.Parameter(torch.randn(1, seq_len_query, feature_dim))
            self.register_parameter("learned_positional_encoding_query", self.learned_positional_encoding_query)

            if seq_len_key_value is None:
                raise ValueError("seq_len_key_value must be provided when using learnable positional encodings.")
            self.learned_positional_encoding_kv = torch.nn.Parameter(torch.randn(1, seq_len_key_value, feature_dim))
            self.register_parameter("learned_positional_encoding_kv", self.learned_positional_encoding_kv)
        elif self.positional_encoding_type in ["none", "None"]:
            pass
        else:
            raise ValueError(f"Unknown positional encoding type: {self.positional_encoding_type}")

        if self.apply_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(feature_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query: torch.Tensor
            query token, shape [batch_size, feature_dim, h*w (=seq_len)]
        key_value: torch.Tensor
            key and value token, shape [batch_size, feature_dim, h*w (=seq_len)]

        Returns
        -------
        torch.Tensor [batch_size, feature_dim, num_agents(=seq_len)]
            updated query token
        """
        # Adjust input shape to match the expectation of the Attention layer,
        # which is [batch_size, sequence_length, feature_dim]
        query = query.transpose(-2, -1)
        query_pos = query.clone()

        key_value = key_value.transpose(-2, -1)
        key_value_pos = key_value.clone()

        if self.positional_encoding_type:
            if self.positional_encoding_type == "learnable":
                positional_encoding_query = self.learned_positional_encoding_query
                positional_encoding_kv = self.learned_positional_encoding_kv
            else:
                raise NotImplementedError(
                    "Positional encoding {} is not implemented.".format(self.positional_encoding_type)
                )

            query_pos += positional_encoding_query
            key_value_pos += positional_encoding_kv

        # Compute attention
        # requires input shape [batch_size, sequence_length, feature_dim]
        updated_query, _ = self.multihead_attn(query=query_pos, key=key_value_pos, value=key_value)

        if self.skip_connection:
            updated_query += query

        if self.apply_layer_norm:
            # Normalize over the feature dimension
            updated_query = self.layer_norm(updated_query)

        return updated_query.transpose(-2, -1)


class BasicLayer(torch.nn.Module):
    """
    This code is copied from: https://github.com/verlab/accelerated_features
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias
            ),
            torch.nn.BatchNorm2d(out_channels, affine=False),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)
