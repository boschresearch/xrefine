# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.

import time
from pathlib import Path

import cv2
import numpy as np
import pygcransac
import torch

from logger import SimpleLogger
from losses import compute_smp_loss
from model import AttnTuner, JointAttnTuner
from settings import get_config, get_logger, print_usage
from utils import compute_pose_error, create_log_dir, desc_dim_dict, evaluate, get_dataset, pose_auc, print_progress

try:
    from pixsfm_refinement import refine_pixsfm
except ImportError:
    print("PixSfM refinement not available. To use it, switch to pixsfm environment.")
    refine_pixsfm = None

CUDA_LAUNCH_BLOCKING = 2, 3
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# The following function is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def pose_validation_imagespace(
    valid_loader,
    model,
    device,
    opt,
    writer=None,
    step=-1,
    without_refinement=False,
    with_pixsfm_refinement=False,
    image_dir=None,
):
    # This validation works in image space instead of normalized coordinate space (as in the keypt2subpx repo).
    if model is not None:
        model.eval()
    err_ts, err_Rs = [], []
    loss_vals = []
    inlier_cnt = []
    outlier_cnt = []

    print("Warning: Loss values are not calculated in this mode.")

    processed_data_in_percent = 0
    refinement_times = []
    pixsfm_internal_refinement_times = []
    start_time = time.time()
    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            processed_data_in_percent = print_progress(
                idx,
                1,
                len(valid_loader),
                processed_data_in_percent,
                start_time,
            )

            corr = data["correspondences"].to(device)
            ret = None
            if len(corr.shape) > 1 and corr.shape[1] > 4:
                gt_E = data["gt_E"].to(device)
                gt_R, gt_t = data["gt_R"].to(device), data["gt_t"].to(device)
                K1s, K2s = data["K1"].to(device), data["K2"].to(device)
                im_size1, im_size2 = data["im_size1"], data["im_size2"]
                patch1, patch2 = data["patch1"].to(device), data["patch2"].to(device)
                descriptor1 = data["descriptor1"].to(device) if "descriptor1" in data else None
                descriptor2 = data["descriptor2"].to(device) if "descriptor2" in data else None
                scorepatch1 = data["scorepatch1"].to(device) if "scorepatch1" in data else None
                scorepatch2 = data["scorepatch2"].to(device) if "scorepatch2" in data else None
                image_list = [data["image_name_1"][0], data["image_name_2"][0]]

                kpts0 = torch.cat(
                    [
                        corr.squeeze()[:, :2],
                        torch.ones_like(corr.squeeze()[:, 0:1]),
                    ],
                    dim=-1,
                )
                kpts1 = torch.cat(
                    [
                        corr.squeeze()[:, 2:4],
                        torch.ones_like(corr.squeeze()[:, 0:1]),
                    ],
                    dim=-1,
                )
                # Let's change mkpts' normalized points to image coordinates
                mkpts0 = kpts0 @ K1s.transpose(-1, -2).squeeze(0)
                mkpts1 = kpts1 @ K2s.transpose(-1, -2).squeeze(0)

                mkpts0 = mkpts0[:, :2] / mkpts0[:, 2:]
                mkpts1 = mkpts1[:, :2] / mkpts1[:, 2:]

                corr = torch.cat([mkpts0, mkpts1], dim=-1).unsqueeze(0)  # B x N x 4
                start_time_refinement = time.time()

                if opt.vanilla or without_refinement:  # without our model
                    updated_correspondence = corr[..., :4]  # B x N x 4
                elif with_pixsfm_refinement:
                    corr = corr.squeeze(0)
                    (
                        updated_correspondence,
                        pixsfm_internal_refinement_time,
                        pixsfm_internal_refinement_time_wo_s2dnet,
                    ) = refine_pixsfm(
                        matches=np.array(corr[..., :4].to("cpu")),
                        image_list=image_list,
                        image_dir=image_dir,
                        match_scores=None,
                    )
                    updated_correspondence = torch.tensor(updated_correspondence).to(device).float().unsqueeze(0)

                else:  # with our model
                    eval_result = evaluate(
                        model=model,
                        device=device,
                        processing_mode=opt.processing_mode,
                        adjust_only_second_keypoint=opt.adjust_only_second_keypoint,
                        cell_size=opt.cell_size,
                        correspondence=corr[..., :4],
                        patch1=patch1,
                        patch2=patch2,
                        scorepatch1=scorepatch1,
                        scorepatch2=scorepatch2,
                        descriptor1=descriptor1,
                        descriptor2=descriptor2,
                        K1s=None,
                        K2s=None,
                        min_match_score_threshold=opt.min_match_score_threshold,
                    )
                    updated_correspondence = eval_result["updated_correspondence"]

                refinement_time = time.time() - start_time_refinement
                refinement_times.append(refinement_time)
                if with_pixsfm_refinement:
                    pixsfm_internal_refinement_times.append(pixsfm_internal_refinement_time)

                mkpts = updated_correspondence
                # Change mkpts image coordinaes to normalized points
                mkpts0 = mkpts[0, :, :2]
                mkpts1 = mkpts[0, :, 2:]

                kpts0 = K1s[:, :2, :2].inverse() @ mkpts0.unsqueeze(-1)
                kpts1 = K2s[:, :2, :2].inverse() @ mkpts1.unsqueeze(-1)
                mkpts = mkpts.squeeze(0).cpu().detach().numpy()

                im_size1 = im_size1.detach().numpy()[0]
                im_size2 = im_size2.detach().numpy()[0]
                K1_np = K1s.detach().cpu().numpy()[0]
                K2_np = K2s.detach().cpu().numpy()[0]

                E, mask = pygcransac.findEssentialMatrix(
                    np.ascontiguousarray(mkpts),
                    K1_np,
                    K2_np,
                    im_size1[0],
                    im_size1[1],
                    im_size2[0],
                    im_size2[1],
                    probabilities=[],
                    threshold=opt.ransac_thr,
                    conf=0.99999,  # RANSAC confidence
                    max_iters=1000,
                    min_iters=1000,
                    sampler=0,
                )

                kpts0 = kpts0[:, :2].cpu().detach().numpy()
                kpts1 = kpts1[:, :2].cpu().detach().numpy()

                mask = np.expand_dims(mask, axis=-1).astype(np.uint8)

                best_num_inliers = 0
                if E is not None:
                    for _E in np.split(E, len(E) / 3):
                        n, R, t, mask_ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
                        if n > best_num_inliers:
                            best_num_inliers = n
                            ret = (R, t[:, 0], mask.ravel() > 0)

                loss_data, _ = compute_smp_loss(updated_correspondence, gt_E, K1s, K2s, opt.train_thr)
                loss = loss_data["total_loss"]
                tmp = loss.cpu().detach().item()
                if np.isnan(tmp):
                    print(f"Pair #{idx}: Loss is nan!!")
                else:
                    loss_vals.append(tmp)

            if ret is None:
                print(f"Pair #{idx}: Not enough points for E estimation or just E estimation failed")
                err_t, err_R = np.inf, np.inf
                inlier_cnt.append(0)
                outlier_cnt.append(corr.shape[1])
            else:
                R, t, inliers = ret
                T_0to1 = torch.cat([gt_R.squeeze(), gt_t.squeeze().unsqueeze(-1)], dim=-1).cpu().numpy()
                err_t, err_R = compute_pose_error(T_0to1, R, t)
                n = np.sum(inliers)
                outlier_cnt.append(len(mkpts) - n)
                inlier_cnt.append(n)

            err_ts.append(err_t)
            err_Rs.append(err_R)

    # Write the evaluation results to disk.
    out_eval = {"error_t": err_ts, "error_R": err_Rs, "inliers": inlier_cnt, "outliers": outlier_cnt}

    pose_errors = []
    for idx in range(len(out_eval["error_t"])):
        pose_error = np.maximum(out_eval["error_t"][idx], out_eval["error_R"][idx])
        if pose_error == np.inf:
            pose_error = 180
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.0 * yy for yy in aucs]
    in_rate = 100.0 * np.sum(inlier_cnt) / (np.sum(inlier_cnt) + np.sum(outlier_cnt))
    err_mean = np.mean(pose_errors)
    err_median = np.median(pose_errors)
    loss = np.mean(loss_vals)

    if writer is not None:
        writer.add_scalar("val/loss", loss, step)
        writer.add_scalar("val/inlier_ratio", np.sum(inlier_cnt) / (np.sum(inlier_cnt) + np.sum(outlier_cnt)), step)
        writer.add_scalar("val/AUC@5", aucs[0], step)
        writer.add_scalar("val/AUC@10", aucs[1], step)
        writer.add_scalar("val/AUC@20", aucs[2], step)

    print(f"Evaluation Results (mean over {len(err_ts)} pairs):")
    print("AUC@5\t AUC@10\t AUC@20\t InRat.\t Mean\t Median\t Avg loss\t")
    print(
        f"{aucs[0]:.2f}\t {aucs[1]:.2f}\t {aucs[2]:.2f}\t "
        f"{in_rate:.2f}\t {err_mean:.2f}\t {err_median:.2f}\t {loss:.8f}"
    )
    print("Average refinement time: ", np.mean(refinement_times))
    if with_pixsfm_refinement:
        print("Average pixsfm internal refinement time: ", np.mean(pixsfm_internal_refinement_times))

    average_refinement_time = (
        np.mean(pixsfm_internal_refinement_times) if with_pixsfm_refinement else np.mean(refinement_times)
    )
    return aucs[0], aucs[1], aucs[2], in_rate, err_mean, err_median, loss, average_refinement_time


# The following function is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def main(opt):
    logger = get_logger()

    # construct folder that should contain pre-calculated correspondences
    test_data = opt.datasets.split(",")  # support multiple training datasets used jointly
    logger.info("Using datasets: [" + ", ".join(test_data) + "]")

    desc_dim = opt.input_channels
    if desc_dim < 0:
        desc_dim = desc_dim_dict[opt.detector]

    model = None
    if not opt.vanilla and not opt.pixsfm_refinement:
        # create or load model
        if opt.processing_mode == "joint_processing":
            model = JointAttnTuner(
                desc_dim=desc_dim,
                use_score=opt.use_score,
                color_normalization_strategy=opt.color_normalization_strategy,
                spatial_argmax_type=opt.spatial_argmax_type,
                attn_with_desc=opt.attn_with_desc,
                attn_with_avg_desc=opt.attn_with_avg_desc,
                attn_with_patch=opt.attn_with_patch,
                num_attention_blocks=opt.num_attention_blocks,
                positional_encoding_type=opt.positional_encoding_type,
                attn_layer_norm=opt.attn_layer_norm,
                attn_skip_connection=opt.attn_skip_connection,
                with_match_score=opt.with_match_score,
                directly_infer_score_map=opt.directly_infer_score_map,
                patch_radius=opt.patch_radius,
                skip_encoder=opt.patch_type != "image",
                encoder_variant=opt.encoder_variant,
                adjust_only_second_keypoint=opt.adjust_only_second_keypoint,
            )
        else:
            model = AttnTuner(
                output_dim=desc_dim,
                use_score=opt.use_score,
                color_normalization_strategy=opt.color_normalization_strategy,
                spatial_argmax_type=opt.spatial_argmax_type,
                no_delta_scaling=opt.no_delta_scaling,
            )
        if len(opt.model) > 0:
            model.load_state_dict(torch.load(opt.model, weights_only=False)["model"])
            logger.info(f"Loaded model from {opt.model}")
        model = model.cuda()
        model.eval()

    # ----------------------------------------
    logger.info(f"Starting experiment {opt.experiment}")
    output_dir = Path(__file__).parent / "results" / opt.experiment

    if output_dir.exists():
        print(
            "\033[91mResults directory {} for the experiment {} already exists. Skipping evaluation.\033[0m".format(
                output_dir, opt.experiment
            )
        )
    else:
        if opt.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        if opt.current_split > 1 and opt.total_split > 1:
            print(f"Working with split: {opt.current_split} / {opt.total_split}", flush=True)

        if opt.test:
            print("Test case...")
            split_info_type = "eval"
        else:
            print("Valid case...")
            split_info_type = "val"

        dataset = get_dataset(
            opt=opt,
            dataset_names=test_data,
            split_info_type=split_info_type,
            with_score=opt.use_score,
            logger=logger,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            num_workers=2,
            batch_size=1,
        )

        output_dir.mkdir(exist_ok=True, parents=True)
        create_log_dir(output_dir, opt, training=False)
        filelogger = SimpleLogger(Path(opt.log_path) / "log_evaluation.txt", decimals=4)
        filelogger.set_metrics(["AUC5"] + ["AUC10", "AUC20", "InRat.", "Mean", "Median", "Loss", "RefinementTime"])

        all_metrics = None
        for i in range(opt.total_run):
            logger.info(f"Test run {i+1}/{opt.total_run}")
            metrics = pose_validation_imagespace(
                dataloader,
                model,
                opt.device,
                opt,
                without_refinement=opt.vanilla,
                with_pixsfm_refinement=opt.pixsfm_refinement,
                image_dir=opt.image_dir,
            )
            filelogger.add_results(metrics)
            if all_metrics is None:
                all_metrics = [[m] for m in metrics]
            else:
                for j in range(len(all_metrics)):
                    all_metrics[j].append(metrics[j])

        averaged_metrics = [round(np.mean(m), 2) for m in all_metrics]

        # Create a summary log file
        filelogger = SimpleLogger(output_dir / "averaged_results.txt", decimals=2)
        filelogger.set_metrics(["AUC5"] + ["AUC10", "AUC20", "InRat.", "Mean", "Median", "Loss", "RefinementTime"])
        filelogger.add_results(averaged_metrics)
        print("----------------------------------------")
        print(f"Final Evaluation Results (average over {opt.total_run} runs):")
        print("AUC@5\t AUC@10\t AUC@20\t InRat.\t Mean\t Median\t Avg loss\t")
        print(
            f"{averaged_metrics[0]:.2f}\t {averaged_metrics[1]:.2f}\t {averaged_metrics[2]:.2f}\t "
            f"{averaged_metrics[3]:.2f}\t {averaged_metrics[4]:.2f}\t {averaged_metrics[5]:.2f}\t "
            f"{averaged_metrics[6]:.8f}"
        )


if __name__ == "__main__":
    # parse command line arguments
    # If we have unparsed arguments, print usage and exit
    opt, unparsed = get_config()
    if len(unparsed) > 0:
        print("Unparsed arguments: ", unparsed)
        print_usage()
        exit(1)
    main(opt)
