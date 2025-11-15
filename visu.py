# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import random

import cv2 as cv
import numpy as np
import torch
from sklearn.decomposition import PCA


def descriptors_as_rgb(desc):
    desc = desc.permute(1, 2, 0)  # code works on channel last
    pca = PCA(n_components=3)
    desc_pca = pca.fit_transform(desc.reshape(desc.shape[0] * desc.shape[1], -1).detach().cpu().numpy())
    desc_pca = (desc_pca - desc_pca.min()) / (desc_pca.max() - desc_pca.min())
    desc_pca = desc_pca.reshape(desc.shape[0], desc.shape[1], 3)
    return torch.from_numpy(desc_pca).to(device=desc.device).permute(2, 0, 1)


def pad_with_zeros(x, pad_amount):
    x = torch.nn.functional.pad(
        x,
        [pad_amount] * 4,
        mode="constant",
        value=0.0,
    )
    return x


def visualize_patches(
    similarity1,
    similarity2,
    descr1,
    descr2,
    delta1,
    delta2,
    device,
    img_patch1=None,
    img_patch2=None,
    gt_delta2=None,
):
    visu_scale = 16  # scale everything up to show subpixel deltas
    pencil_size = 2
    batch_idx = 0  # only visualize for one image pair of the batch
    feat_idx = random.randint(1, img_patch1.shape[1] - 1)
    pad_amount = int((img_patch1.shape[-1] - similarity1.shape[-1]) / 2)

    sim1 = (similarity1[batch_idx, feat_idx, :, :, :] + 1.0) * 0.5  # range [-1,1] to [0,1]
    sim2 = (similarity2[batch_idx, feat_idx, :, :, :] + 1.0) * 0.5  # range [-1,1] to [0,1]
    sim1 = pad_with_zeros(sim1, pad_amount)
    sim2 = pad_with_zeros(sim2, pad_amount)
    similarity_concat = torch.cat((sim1, sim2), dim=-1).detach()
    similarity_concat = torch.cat((similarity_concat, similarity_concat, similarity_concat), dim=0)  # 3, H, W*2

    descr1_rgb = descriptors_as_rgb(descr1[batch_idx, feat_idx, :, :, :])
    descr1_rgb = pad_with_zeros(descr1_rgb, pad_amount)
    descr2_rgb = descriptors_as_rgb(descr2[batch_idx, feat_idx, :, :, :])
    descr2_rgb = pad_with_zeros(descr2_rgb, pad_amount)
    descr_rgb_concat = torch.cat((descr1_rgb, descr2_rgb), dim=-1)  # 3, H, W*2

    if img_patch1 is not None and img_patch2 is not None:
        single_patch_1 = img_patch1[batch_idx, feat_idx, :, :, :]
        single_patch_2 = img_patch2[batch_idx, feat_idx, :, :, :]
        img_patch_concat = torch.cat((single_patch_1, single_patch_2), dim=-1)

        # for the original megadepth dataset, patches have just 1 channel
        if img_patch_concat.shape[0] == 1:
            img_patch_concat = torch.cat((img_patch_concat, img_patch_concat, img_patch_concat), dim=0)
    else:
        img_patch_concat = torch.zeros_like(similarity_concat, device=device)

    descrpatch_rgb_concat = torch.zeros_like(similarity_concat, device=device)

    all_concat = torch.cat((img_patch_concat, descrpatch_rgb_concat, similarity_concat, descr_rgb_concat), dim=-2)
    all_concat = torch.nn.functional.interpolate(
        all_concat.unsqueeze(0), scale_factor=visu_scale, mode="bilinear", align_corners=True
    ).squeeze()
    all_concat = all_concat.detach().cpu().permute(1, 2, 0) * 255  # .squeeze()
    all_concat = np.ascontiguousarray(all_concat.type(torch.uint8).cpu().numpy())

    p_size = single_patch_1.shape[-1] * visu_scale
    center_coord = int((p_size) / 2)
    center = np.array([center_coord, center_coord]).astype(np.int32)
    d1 = np.round(delta1[batch_idx, feat_idx].squeeze().detach().cpu().numpy() * visu_scale).astype(np.int32)
    d2 = np.round(delta2[batch_idx, feat_idx].squeeze().detach().cpu().numpy() * visu_scale).astype(np.int32)

    # draw kp and refined kp in patch 1
    cv.circle(all_concat, center, pencil_size, (255, 0, 0), -2)
    cv.circle(all_concat, center + d1, pencil_size, (0, 0, 255), -2)
    # draw kp and refined kp in patch 2
    cv.circle(all_concat, center + np.array([p_size, 0]), pencil_size, (255, 0, 0), -2)
    cv.circle(all_concat, center + np.array([p_size, 0]) + d2, pencil_size, (0, 0, 255), -2)
    if gt_delta2 is not None:
        gt_d2 = np.round(gt_delta2[batch_idx, feat_idx].squeeze().detach().cpu().numpy() * visu_scale).astype(np.int32)
        cv.circle(all_concat, center + np.array([p_size, 0]) + gt_d2, pencil_size, (0, 255, 0), -2)

    return torch.from_numpy(all_concat).permute(2, 0, 1)
