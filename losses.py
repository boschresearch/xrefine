# This source code is from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# This source code is licensed under the Apache-2.0 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import torch


# Below functions
def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    # print(f"x: {x.size()[0]}, {x.size()[1]}, {x.dtype}, {x.device}")
    if len(x.size()) == 2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size()) == 3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo


def _sampson_dist(F, X, Y, if_homo=False):
    if not if_homo:
        X = _homo(X)
        Y = _homo(Y)
    if len(X.size()) == 2:
        nominator = (torch.diag(Y @ F @ X.t())) ** 2
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    else:
        nominator = (torch.diagonal(Y @ F @ X.transpose(1, 2), dim1=1, dim2=2)) ** 2
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2

    errors = nominator / denom
    return errors


def compute_smp_loss(inp, gt_E, K1s, K2s, train_thr, match_score=None):
    smp_dist = _sampson_dist(gt_E, inp[..., :2], inp[..., 2:4])

    smp_dist_mean = smp_dist.mean()
    smp_dist_01 = smp_dist[smp_dist < 0.1].mean()
    smp_dist_03 = smp_dist[smp_dist < 0.3].mean()
    loss_data = {"smp_dist": smp_dist_mean, "smp_dist_01": smp_dist_01, "smp_dist_03": smp_dist_03}
    sampling_data = {"smp_dist": smp_dist}

    if match_score is not None:
        kappa = smp_dist_mean
        # match_score is encouraged to be large if smp_dist is small and vise versa,
        # because the smaller the match_score the larger the value that is added to the loss from the substraction part
        smp_dist_loss = smp_dist * match_score - kappa * torch.log(match_score + 1e-8)
    else:
        smp_dist_loss = smp_dist

    threshold = train_thr / ((K1s[:, 0, 0] + K1s[:, 1, 1] + K2s[:, 0, 0] + K2s[:, 1, 1]) / 4)
    threshold = threshold.view(-1, 1)
    thresholdsq = threshold**2.0
    # squared msac score with essential error
    smp_dist_loss = torch.where(smp_dist < thresholdsq, smp_dist_loss / thresholdsq, torch.ones_like(smp_dist))

    total_loss = smp_dist_loss.mean()

    loss_data["total_loss"] = total_loss

    return loss_data, sampling_data
