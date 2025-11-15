# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.
# This file uses code modified from https://github.com/cvg/glue-factory, which is under the Apache 2.0 license.

from typing import Optional, Tuple

import torch


def boolean_string(input_string: str) -> bool:
    """
    Function that defines the type 'boolean_string',
    which can be used in a ArgumentParser, to enable bools as input type.

    Args:
        s (str): The input string that should be converted to a bool, has to be in ["False", "false", "True", "true"].

    Returns:
        bool: The corresponding bool to the given input string.

    Raises:
        ValueError: If input string is not in ["False", "false", "True", "true"].
    """
    if input_string not in {"False", "false", "True", "true"}:
        raise ValueError("Not a valid boolean string")
    return input_string in ["True", "true"]


# The following function is from KP2D (https://github.com/TRI-ML/KP2D)
# Copyright (c) 2019 Toyota Research Institute (TRI)., licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def meshgrid(
    batch_size: int,
    grid_height: int,
    grid_width: int,
    dtype: torch.dtype,
    device: torch.device,
    normalized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create mesh-grid given batch size, height and width dimensions.

    Parameters
    ----------
    batch_size: int
        Batch size
    grid_height: int
        Grid Height
    grid_width: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = torch.linspace(-1, 1, grid_width, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, grid_height, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, grid_width - 1, grid_width, device=device, dtype=dtype)
        ys = torch.linspace(0, grid_height - 1, grid_height, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs], indexing="ij")
    return xs.repeat([batch_size, 1, 1]), ys.repeat([batch_size, 1, 1])


# The following function is from KP2D (https://github.com/TRI-ML/KP2D)
# Copyright (c) 2019 Toyota Research Institute (TRI)., licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def image_grid(
    batch_size: int,
    grid_height: int,
    grid_width: int,
    dtype: torch.dtype,
    device: torch.device,
    ones: bool = True,
    normalized: bool = False,
) -> torch.Tensor:
    """Create an image mesh grid with shape B3HW given image shape BHW.

    Parameters
    ----------
    batch_size: int
        Batch size
    grid_height: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(batch_size, grid_height, grid_width, dtype, device, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid


def get_normalized_coordinates(
    coordinates: torch.Tensor, image_width: int, image_height: int, permute_for_grid_sample: bool = True
) -> torch.Tensor:
    """
    Get normalized coordinates between -1 and 1, e.g. for pytorch grid_sample.

    Args:
        coordinates (torch.tensor): (batch size, 2, height, width)
            The Image coordinates to be normalized.
        image_width (int):
            Image width
        image_height (int):
            Image height
        permute_for_grid_sample (bool, optional):
            Whether to permute the normalized coordinates for direct usage in the grid_sample operation.
            Defaults to True.

    Returns:
        torch.tensor: shape (batch size, height, width, 2) if permute_for_grid_sample=True
                      shape (batch size, 2, height, width) if permute_for_grid_sample=False

    """
    coordinates_norm = coordinates.clone()

    # For compatibility, we avoid using the slicing operator here on the left side of the equation
    coordinates_norm_tmp0 = (coordinates_norm[:, :1, :, :] / (float(image_width) / 2.0)) - 1.0
    coordinates_norm_tmp1 = (coordinates_norm[:, 1:, :, :] / (float(image_height) / 2.0)) - 1.0
    coordinates_norm = torch.cat([coordinates_norm_tmp0, coordinates_norm_tmp1], dim=1)

    if permute_for_grid_sample:
        coordinates_norm = coordinates_norm.permute(0, 2, 3, 1)

    return coordinates_norm


def get_depth_values_torch(
    points: torch.Tensor,
    depth_map: torch.Tensor,
    points_norm: Optional[torch.Tensor] = None,
    origin_is_center_of_first_pixel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get an array of depth values for the given points from a given depth map.

    Args:
        points (torch.Tensor (N,2)): Points.
        depth_map (torch.Tensor (H,W)): Depth map.
        points_norm (torch.Tensor (1,grid_height,grid_width,2)): Normalized points.

    Returns:
        torch.Tensor (N,): Corresponding depth values for the given points.
        torch.Tensor (N,): Corresponding depth value availability for the given points.
    """
    height, width = depth_map.shape
    if points_norm is None:
        points_norm = get_normalized_coordinates(points.t().unsqueeze(dim=0).unsqueeze(dim=3), width, height)
    else:
        points_norm = points_norm.view(1, points_norm.shape[1] * points_norm.shape[2], 1, 2)
    depth_values = torch.nn.functional.grid_sample(
        depth_map.unsqueeze(dim=0).unsqueeze(dim=1),
        points_norm,
        mode="nearest",  # we use nearest, to avoid interpolation with undefined depth values (represented by 0.0)
        align_corners=origin_is_center_of_first_pixel,
    )
    depth_values = depth_values.squeeze(dim=(0, 1, 3))
    depth_value_availability = depth_values > 0.0

    return depth_values, depth_value_availability


def pinhole_to_world_torch(
    camera_matrix: torch.Tensor, camera_pose: torch.Tensor, image_points: torch.Tensor, depth_values: torch.Tensor
) -> torch.Tensor:
    """Maps points in image coordinates to world coordinates.

    Args:
        camera_matrix (torch.Tensor (3,3)): Camera matrix to map camera coordinates to image coordinates.
        camera_pose (torch.Tensor (4,4)): The camera pose mapping points from world to camera coordinates.
        image_points (torch.Tensor (N,2)): The points in the image to be mapped to world coordinates.
        depth_values (torch.Tensor (N,)): Corresponding depth values for each image point.

    Returns:
        torch.Tensor (N,3): Corresponding world coordinates for the image points.
    """
    inverted_camera_matrix = torch.linalg.inv(camera_matrix)
    homogeneous_image_points = torch.column_stack(
        [image_points, torch.ones((image_points.shape[0], 1), device=image_points.device)]
    )
    warped_points = torch.transpose(
        inverted_camera_matrix.matmul(torch.transpose(homogeneous_image_points, 0, 1)), 0, 1
    )
    normalized_warped_points = warped_points[:, :] / warped_points[:, 2:]
    relative_world_points = normalized_warped_points * depth_values.unsqueeze(1)
    relative_world_points = torch.column_stack(
        [relative_world_points, torch.ones((relative_world_points.shape[0], 1), device=relative_world_points.device)]
    )
    return torch.transpose(torch.linalg.inv(camera_pose).matmul(torch.transpose(relative_world_points, 0, 1)), 0, 1)[
        :, :3
    ]


def world_to_pinhole_torch(
    camera_matrix: torch.Tensor, camera_pose: torch.Tensor, world_points: torch.Tensor
) -> torch.Tensor:
    """Maps points in world coordinates to image coordinates.

    Args:
        camera_matrix (torch.Tensor (3,3)): Camera matrix to map camera coordinates to image coordinates.
        camera_pose (torch.Tensor (4,4)): The camera pose mapping points from world to camera coordinates.
        world_points (torch.Tensor (N,3)): The points in world coordinates to be mapped to image coordinates.

    Returns:
       torch.Tensor (N,2): Corresponding image coordinates for the world points.
    """
    homogeneous_world_points = torch.column_stack(
        [world_points, torch.ones((world_points.shape[0], 1), device=world_points.device)]
    )
    warped_world_points = camera_pose[:3, :].matmul(torch.transpose(homogeneous_world_points, 0, 1))
    image_points = torch.transpose(camera_matrix.matmul(warped_world_points), 0, 1)
    normalized_image_points = image_points[:, :2] / image_points[:, 2:]
    return normalized_image_points


def project_from_image_to_world_torch(
    camera_model: str,
    camera_matrix: torch.Tensor,
    camera_pose: torch.Tensor,
    image_points: torch.Tensor,
    depth_values: torch.Tensor,
) -> torch.Tensor:
    """
    Maps points in image coordinates to world coordinates.

    Args:
        camera_model (str):
            Camera model identifier.
        camera_matrix (torch.Tensor (3,3)):
            Camera matrix to map camera coordinates to image coordinates.
        camera_pose (torch.Tensor (4,4)):
            The camera pose mapping points from world to camera coordinates.
        image_points (torch.Tensor (N,2)):
            The points in the image to be mapped to world coordinates.
        depth_values (torch.Tensor (N,)):
            Corresponding depth values for each image point.

    Returns:
        torch.Tensor (N,3): Corresponding world coordinates for the image points.

    Raises
    ------
    NotImplementedError
        If the method has not been implemented for the camera model of this camera object.
    """
    if camera_model == "pinhole":
        return pinhole_to_world_torch(
            camera_matrix=camera_matrix,
            camera_pose=camera_pose,
            image_points=image_points,
            depth_values=depth_values,
        )
    else:
        raise NotImplementedError(
            "Camera model {} is not implemented for project_from_image_to_world_torch()".format(camera_model)
        )


def project_from_world_to_image_torch(
    camera_model: str,
    camera_matrix: torch.Tensor,
    camera_pose: torch.Tensor,
    world_points: torch.Tensor,
) -> torch.Tensor:
    """
    Maps points in world coordinates to image coordinates.

    Args:
        camera_model (str):
            Camera model identifier.
        camera_matrix (torch.Tensor (3,3)):
            Camera matrix to map camera coordinates to image coordinates.
        camera_pose (torch.Tensor (4,4)):
            The camera pose mapping points from world to camera coordinates.
        world_points (torch.Tensor (N,3)):
            The points in world coordinates to be mapped to image coordinates.

    Returns:
        torch.Tensor (N,2): Corresponding image coordinates for the world points.

    Raises
    ------
    NotImplementedError
        If the method has not been implemented for the camera model of this camera object.
    """
    if camera_model == "pinhole":
        return world_to_pinhole_torch(camera_matrix=camera_matrix, camera_pose=camera_pose, world_points=world_points)
    else:
        raise NotImplementedError(
            "Camera model {} is not implemented for project_from_world_to_image_torch()".format(camera_model)
        )


def project_points_from_a_to_b_torch(
    camera_model_a: str,
    camera_matrix_a: torch.Tensor,
    camera_pose_a: torch.Tensor,
    camera_model_b: str,
    camera_matrix_b: torch.Tensor,
    camera_pose_b: torch.Tensor,
    image_points_a: torch.Tensor,
    depth_map_a: torch.Tensor,
    image_points_a_norm: Optional[torch.Tensor] = None,
    image_a_width: Optional[int] = None,
    image_a_height: Optional[int] = None,
    origin_is_center_of_first_pixel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Projects points from one image to the other.

    Args:
        camera_model_a (str): camera model identifier of image a.
        camera_matrix_a (torch.Tensor (3,3)): camera matrix of image a.
        camera_pose_a (torch.Tensor (4, 4)): Pose of camera a.
        camera_model_b (str): camera model identifier of image b.
        camera_matrix_b (torch.Tensor (3,3)): camera matrix of image b.
        camera_pose_b (torch.Tensor (4, 4)): Pose of camera b.
        image_points_a (torch.Tensor (N,2)): Points in image a.
        depth_map_a (torch.Tensor (H, W)): Depth map of image a.
        image_points_a_norm (torch.Tensor (1,grid_height,grid_width,2)): Normalized points.
        image_a_width (int): Width of image a.
        image_a_height (int): Height of image a.

    Returns:
        torch.Tensor (N,2): The projected points in image b.
        torch.Tensor (N): Boolean vector providing info about the useability of the projected points,
        i.e. was a depth value available for the corresponding keypoint from image a
        and is the projected point visible in image a.
    """
    depth_values, projected_point_useablity = get_depth_values_torch(
        points=image_points_a.detach(),
        depth_map=depth_map_a.squeeze(dim=0),
        points_norm=image_points_a_norm.detach() if image_points_a_norm is not None else None,
        origin_is_center_of_first_pixel=origin_is_center_of_first_pixel,
    )
    world_points_a = project_from_image_to_world_torch(
        camera_model=camera_model_a,
        camera_matrix=camera_matrix_a,
        camera_pose=camera_pose_a,
        image_points=image_points_a,
        depth_values=depth_values,
    )
    image_points_b = project_from_world_to_image_torch(
        camera_model=camera_model_b,
        camera_matrix=camera_matrix_b,
        camera_pose=camera_pose_b,
        world_points=world_points_a,
    )
    if image_a_width is not None and image_a_height is not None:
        projected_point_useablity = projected_point_useablity.unsqueeze(dim=1)
        projected_point_useablity[image_points_b[:, 0] < 0.0] = False
        projected_point_useablity[image_points_b[:, 1] < 0.0] = False
        projected_point_useablity[image_points_b[:, 0] > image_a_width] = False
        projected_point_useablity[image_points_b[:, 1] > image_a_height] = False
        projected_point_useablity = projected_point_useablity.squeeze(dim=1)
    return image_points_b, projected_point_useablity


def warp_source_points_to_target(
    data: dict,
    source_uv: torch.Tensor,
    source_uv_norm: torch.Tensor,
    keypoint_mask: Optional[torch.Tensor] = None,
    origin_is_center_of_first_pixel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Warps source points to target points using different methods based on the input data.

    Args:
        data (dict):
            Dictionary containing various data required for warping.
        source_uv (torch.Tensor, torch.float) [batch_size, 2, grid_height, grid_width]:
            Source keypoint coordinates.
        source_uv_norm (torch.Tensor, torch.float) [batch_size, grid_height, grid_width, 2]:
            Normalized source keypoint coordinates.
        keypoint_mask (Optional[torch.Tensor], optional, torch.bool) [batch_size, grid_height, grid_width]:
            Mask for keypoints.
            Defaults to None.
        origin_is_center_of_first_pixel (bool, optional):
            Flag indicating if the origin is the center of the first pixel.
            Defaults to False.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - source_uv_warped (torch.Tensor):
                Warped source keypoint coordinates.
            - source_uv_warped_norm (torch.Tensor):
                Normalized warped source keypoints coordinates.
            - keypoint_mask (Optional[torch.Tensor]):
                Updated keypoint mask.
    """
    _, _, image_height, image_width = data["target_image"].shape
    # Batch size of source_uv can be different from that of the images (if this function is called per batch index)
    batch_size, _, _, _ = source_uv.shape

    gt_source_to_target_transform = data["gt_source_to_target_transform"]
    source_depth_map = data["source_depth_map"]
    target_depth_map = data["target_depth_map"]

    source_camera_model = data["source_camera_model"]
    source_camera_matrix = data["source_camera_matrix"]
    target_camera_model = data["target_camera_model"]
    target_camera_matrix = data["target_camera_matrix"]

    source_uv_warped_list = []
    if keypoint_mask is not None:
        source_visibility_list = []
    for image_index in range(batch_size):
        source_uv_i = source_uv[image_index].view(2, -1).t()
        source_uv_norm_i = source_uv_norm[image_index].unsqueeze(dim=0)
        source_uv_warped_i, source_visibility_i = project_points_from_a_to_b_torch(
            camera_model_a=source_camera_model[image_index],
            camera_matrix_a=source_camera_matrix[image_index],
            camera_pose_a=torch.eye(4, device=source_camera_matrix[image_index].device),
            camera_model_b=target_camera_model[image_index],
            camera_matrix_b=target_camera_matrix[image_index],
            camera_pose_b=gt_source_to_target_transform[image_index],
            image_points_a=source_uv_i,
            image_points_a_norm=source_uv_norm_i,
            depth_map_a=source_depth_map[image_index],
            image_a_width=image_width,
            image_a_height=image_height,
            origin_is_center_of_first_pixel=origin_is_center_of_first_pixel,
        )

        # for real image pairs, we have to account for the fact that due to the perspective change, points get
        # occluded, i.e. two points in the source image can be projected to the same point in the target image.
        # In this case, only the one with the smaller depth should be kept.
        # We here check this using cycle consistency:
        # we project the points back to the source image and check if they are close to the original points.
        # If they are not, we mask them out.
        consistency_threshold = 3.0
        source_uv_warped_norm_i_reshaped = source_uv_warped_i.t().view(source_uv[image_index].shape).unsqueeze(dim=0)
        source_uv_warped_norm_i = get_normalized_coordinates(
            source_uv_warped_norm_i_reshaped, image_width, image_height
        )

        # shape [num_points, 2]
        source_uv_warped_i_back, source_visibility_i_back = project_points_from_a_to_b_torch(
            camera_model_a=target_camera_model[image_index],
            camera_matrix_a=target_camera_matrix[image_index],
            camera_pose_a=gt_source_to_target_transform[image_index],
            camera_model_b=source_camera_model[image_index],
            camera_matrix_b=source_camera_matrix[image_index],
            camera_pose_b=torch.eye(4, device=source_camera_matrix[image_index].device),
            image_points_a=source_uv_warped_i,
            image_points_a_norm=source_uv_warped_norm_i,
            depth_map_a=target_depth_map[image_index],
            image_a_width=image_width,
            image_a_height=image_height,
            origin_is_center_of_first_pixel=origin_is_center_of_first_pixel,
        )
        # shape [num_points]
        consistency_mask_source = (
            torch.linalg.norm(source_uv_i - source_uv_warped_i_back, dim=1) < consistency_threshold
        )
        source_visibility_i *= consistency_mask_source * source_visibility_i_back

        source_uv_warped_list.append(source_uv_warped_i.t().view(source_uv[image_index].shape))
        if keypoint_mask is not None:
            source_visibility_list.append(source_visibility_i.view(keypoint_mask[image_index].shape))
    source_uv_warped = torch.stack(source_uv_warped_list, dim=0)
    source_uv_warped_norm = get_normalized_coordinates(source_uv_warped, image_width, image_height)
    if keypoint_mask is not None:
        source_visibility = torch.stack(source_visibility_list, dim=0)
        keypoint_mask = keypoint_mask & source_visibility

    return source_uv_warped, source_uv_warped_norm, keypoint_mask


# The following function is based on code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def sampson_dist(gt_rot, gt_t, k1, k2, x1, x2):
    ones = torch.ones(x1.size()[0], 1, dtype=x1.dtype, device=x1.device)
    x1 = torch.cat((x1, ones), 1)
    x2 = torch.cat((x2, ones), 1)

    gt_essential_mat = torch.zeros((3, 3), device=x1.device, dtype=x1.dtype)
    gt_essential_mat[0, 1] = -float(gt_t[2, 0])
    gt_essential_mat[0, 2] = float(gt_t[1, 0])
    gt_essential_mat[1, 0] = float(gt_t[2, 0])
    gt_essential_mat[1, 2] = -float(gt_t[0, 0])
    gt_essential_mat[2, 0] = -float(gt_t[1, 0])
    gt_essential_mat[2, 1] = float(gt_t[0, 0])

    gt_essential_mat = gt_essential_mat.mm(gt_rot)

    # fundamental matrix from essential matrix
    fudamental_mat = k2.inverse().transpose(0, 1).mm(gt_essential_mat).mm(k1.inverse())
    nominator = (torch.diag(x2 @ fudamental_mat @ x1.t())) ** 2
    fx1 = torch.mm(fudamental_mat, x1.t())
    fx2 = torch.mm(fudamental_mat.t(), x2.t())
    denom = fx1[0] ** 2 + fx1[1] ** 2 + fx2[0] ** 2 + fx2[1] ** 2

    errors = nominator / denom
    return errors


# The following function is based on code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def extract_patches_from_map(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    patch_radius: int,
):
    image_padded = torch.nn.functional.pad(
        image,
        [patch_radius] * 4,
        mode="constant",
        value=0.0,
    )
    # In the original image, keypoint coordinates represent the center of the patch that should be extracted.
    # After padding, however, the same coordinates represent the top left corner of the patch that should be extracted.
    patches = extract_patches(
        tensor=image_padded,
        top_left_corners=keypoints.to(device=image.device, dtype=torch.int32),
        ps=2 * patch_radius + 1,
    )
    return patches


# The following function is based on code from Glue Factory (https://github.com/cvg/glue-factory)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def extract_patches(
    tensor: torch.Tensor,
    top_left_corners: torch.Tensor,
    ps: int,
) -> torch.Tensor:
    c, h, w = tensor.shape
    top_left_corners_at_pixel_position = top_left_corners.long()
    top_left_corners_at_pixel_position[:, 0] = top_left_corners_at_pixel_position[:, 0].clamp(min=0, max=w - ps)
    top_left_corners_at_pixel_position[:, 1] = top_left_corners_at_pixel_position[:, 1].clamp(min=0, max=h - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(top_left_corners_at_pixel_position) + top_left_corners_at_pixel_position[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1)


# The following function is based on code from Glue Factory (https://github.com/cvg/glue-factory)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# The function has been modified for Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def nearest_neighbor_match(descriptors0, descriptors1):
    # The matcher part is borrowed and modified from GlueFactory (https://github.com/cvg/glue-factory)
    # Nearest neighbor matcher for normalized descriptors.
    # Optionally apply the mutual check and threshold the distance or ratio.
    @torch.no_grad()
    def find_nn(sim, ratio_thresh, distance_thresh):
        sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
        dist_nn = 2 * (1 - sim_nn)
        mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
        if ratio_thresh:
            mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
        if distance_thresh:
            mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
        matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
        return matches

    def mutual_check(m0, m1):
        inds0 = torch.arange(m0.shape[-1], device=m0.device)
        inds1 = torch.arange(m1.shape[-1], device=m1.device)
        loop0 = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
        loop1 = torch.gather(m0, -1, torch.where(m1 > -1, m1, m1.new_tensor(0)))
        m0_new = torch.where((m0 > -1) & (inds0 == loop0), m0, m0.new_tensor(-1))
        m1_new = torch.where((m1 > -1) & (inds1 == loop1), m1, m1.new_tensor(-1))
        return m0_new, m1_new

    sim = torch.einsum("bnd,bmd->bnm", descriptors0, descriptors1)
    matches0 = find_nn(sim, None, None)
    matches1 = find_nn(sim.transpose(1, 2), None, None)
    matches0, matches1 = mutual_check(matches0, matches1)
    b, m, n = sim.shape
    la = sim.new_zeros(b, m + 1, n + 1)
    la[:, :-1, :-1] = torch.nn.functional.log_softmax(sim, -1) + torch.nn.functional.log_softmax(sim, -2)
    mscores0 = (matches0 > -1).float()
    mscores1 = (matches1 > -1).float()
    return {
        "matches0": matches0,
        "matches1": matches1,
        "matching_scores0": mscores0,
        "matching_scores1": mscores1,
        "similarity": sim,
        "log_assignment": la,
    }


def sample_distortions(
    sample_matches: str, num_keypoints: int, max_distortion: float, normal_distortion_std: float, device: str
):
    if sample_matches == "uniformly":
        # Sample distortions uniformly in the range [-1, 1)
        distortions = torch.rand(num_keypoints, device=device) * 2 - 1
    elif sample_matches == "normally":
        # Sample distortions from a normal distribution
        # mean = 0, std = normal_distortion_std -> 99.7% of values are in the range [-3 x std, +3 x std]
        distortions = torch.normal(0, normal_distortion_std, size=(num_keypoints,), device=device)
        # Resample distortions with absolute values larger than 1
        while torch.sum(distortions >= 1.0).item() > 0:
            distortions = torch.where(
                distortions.abs() >= 1,
                torch.normal(0, normal_distortion_std, size=(num_keypoints,), device=device),
                distortions,
            )
    return distortions * max_distortion
