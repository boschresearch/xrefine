# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This file uses code modified from https://github.com/KimSinjeong/keypt2subpx, which is under the Apache 2.0 license.

import os
import time
from pathlib import Path
from test import pose_validation_imagespace

import numpy as np
import torch
from tensorboardX import SummaryWriter

from logger import SimpleLogger
from losses import compute_smp_loss
from model import AttnTuner, JointAttnTuner
from settings import get_config, get_logger, print_usage
from utils import create_log_dir, desc_dim_dict, evaluate, get_dataset, get_git_hash, print_progress, set_seeds
from visu import visualize_patches

CUDA_LAUNCH_BLOCKING = 2, 3
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train_step(
    step: int,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    opt,
    data: dict,
    writer: SummaryWriter,
    device: torch.device,
):
    # Data preparation
    corr = data["correspondences"].to(device)
    patch1 = data["patch1"].to(device)
    patch2 = data["patch2"].to(device)
    descriptor1 = data["descriptor1"].to(device) if "descriptor1" in data else None
    descriptor2 = data["descriptor2"].to(device) if "descriptor2" in data else None
    gt_E = data["gt_E"].to(device) if "gt_E" in data else None
    K1s = data["K1"].to(device) if "K1" in data else None
    K2s = data["K2"].to(device) if "K2" in data else None
    gt_pts2 = data["gt_pts2"].to(device) if "gt_pts2" in data else None
    scorepatch1 = data["scorepatch1"].to(device) if "scorepatch1" in data else None
    scorepatch2 = data["scorepatch2"].to(device) if "scorepatch2" in data else None
    end_point_error = data["end_point_error"].to(device) if "end_point_error" in data else None

    # Obtain updated correspondences from the model
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
        K1s=K1s,
        K2s=K2s,
        min_match_score_threshold=0.0,  # Use all matches
    )
    updated_correspondence = eval_result["updated_correspondence"]
    match_score = eval_result["match_score"]

    # Calculate loss
    losses, loss_samples = compute_smp_loss(
        inp=updated_correspondence,
        gt_E=gt_E,
        K1s=K1s,
        K2s=K2s,
        train_thr=opt.train_thr,
        match_score=match_score if opt.learn_match_score_as_confidence else None,
    )

    if match_score is not None:
        if not opt.learn_match_score_as_confidence:
            # This is a prototypical implementation; to be modified / tested
            gt_correct_matches = end_point_error < 4.0
            bce_loss = torch.nn.BCELoss(reduction="none")
            match_score_loss = bce_loss(match_score, gt_correct_matches.float())
            losses["match_score_loss"] = match_score_loss.mean()
            losses["total_loss"] += losses["match_score_loss"]
            loss_samples["match_score_loss"] = match_score_loss

    total_loss = losses["total_loss"]

    # tesorboard logging
    gt_delta2 = gt_pts2 - updated_correspondence[..., 2:4] if gt_pts2 is not None else None
    if opt.patch_type == "image" and (step % opt.visu_intv) == 0:
        patch_visu = visualize_patches(
            eval_result["similarity1"],
            eval_result["similarity2"],
            eval_result["descr1"],
            eval_result["descr2"],
            eval_result["delta1"],
            eval_result["delta2"],
            device,
            patch1,
            patch2,
            gt_delta2,
        )
        writer.add_image("train/patches", patch_visu, step + 1)

    for key, val in losses.items():
        writer.add_scalar(f"train/{key}", val.item(), step + 1)
    for key, val in loss_samples.items():
        writer.add_histogram(f"train/{key}", val, step + 1)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return [total_loss.item()]


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    step: int,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    opt,
    writer: SummaryWriter,
    device: torch.device,
    total_loss_values_history: list,
    logger_train: list,
    epoch: int,
):
    processed_data_in_percent = 0
    start_time = time.time()
    model.train()
    for batch_index, train_data in enumerate(train_loader):
        step += 1
        cur_lr = optimizer.param_groups[0]["lr"]

        loss_vals = train_step(
            step,
            optimizer,
            model,
            opt,
            train_data,
            writer,
            device,
        )

        total_loss_values_history.append(loss_vals[0])
        if len(total_loss_values_history) > 100:
            total_loss_values_history.pop(0)

        status_info = ": Epoch {}, LR: {}, total_loss: {:.3f}".format(
            epoch,
            cur_lr,
            np.mean(total_loss_values_history),
        )
        processed_data_in_percent = print_progress(
            batch_index,
            opt.batchsize,
            len(train_loader),
            processed_data_in_percent,
            start_time,
            additional_status_info=status_info,
        )

        if step % 100 == 0:
            logger_train.add_results([cur_lr] + loss_vals)
    return step, total_loss_values_history


def validate(valid_loader, model, device, opt, writer, step, logger_valid) -> float:
    current_error = np.inf
    aucs5, aucs10, aucs20, _, _, _, _, _ = pose_validation_imagespace(valid_loader, model, device, opt, writer, step)
    logger_valid.add_results([aucs5, aucs10, aucs20])
    current_error = 100 - aucs5
    return current_error


# The following function is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def train(model, train_loader, valid_loader, opt, writer, logger, device):
    # Determine the initial learning rate
    initial_learning_rate = opt.learning_rate
    if opt.adapt_lr_to_batchsize_divisor > 0:
        initial_learning_rate = initial_learning_rate * (opt.batchsize / opt.adapt_lr_to_batchsize_divisor)

    # Initialize
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_learning_rate,
        weight_decay=opt.weight_decay,
    )
    if opt.lr_scheduler_factor > 0.0 and opt.lr_scheduler_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=opt.lr_scheduler_factor,
            patience=opt.lr_scheduler_patience,
        )
    else:
        scheduler = None

    resume = opt.resume
    if resume and not os.path.isfile(opt.model):
        logger.info("Warning: No checkpoint found at '{}'".format(opt.model))
        resume = False
    logger_train = SimpleLogger(os.path.join(opt.log_path, "log_train.txt"), decimals=4)
    logger_valid = SimpleLogger(os.path.join(opt.log_path, "log_valid.txt"), decimals=4)

    # Create model
    if resume:
        logger.info("==> Resuming from checkpoint: {}".format(opt.model))
        checkpoint = torch.load(opt.model, map_location=torch.device("cpu"))
        best_error = checkpoint["best_error"]
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        logger.info("==> Training is starting from scratch..")
        epoch = 0
        step = 0

        logger_train.set_metrics(["Learning Rate"] + ["Sampson Loss"])
        logger_valid.set_metrics(["AUC5", "AUC10", "AUC20"])
        best_error = np.inf

    # Training
    epoch = 0
    total_loss_values_history = []
    while epoch < opt.epochs:
        epoch += 1
        logger.info(f"Epoch {epoch}")

        step, total_loss_values_history = train_one_epoch(
            train_loader,
            step,
            optimizer,
            model,
            opt,
            writer,
            device,
            total_loss_values_history,
            logger_train,
            epoch,
        )

        # Validation and checkpointing
        current_error = validate(valid_loader, model, device, opt, writer, step, logger_valid)
        if scheduler is not None:
            scheduler.step(current_error)

        if current_error < best_error:
            best_error = current_error
            logger.info(f"Saving best model with error: {best_error:6.3f} at step {step}")
            torch.save(
                {
                    "epoch": epoch,
                    "step": step + 1,
                    "model": model.state_dict(),
                    "best_error": best_error,
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(opt.model_path, "best_model.pth"),
            )

        logger.info(f"Saving latest model at step {step}")
        torch.save(
            {
                "epoch": epoch,
                "step": step + 1,
                "model": model.state_dict(),
                "best_error": best_error,
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(opt.model_path, "latest_model.pth"),
        )
        logger.info(
            "Maximum amount of GPU memory used during the training so far: {:.0f}MB\n".format(
                torch.cuda.max_memory_reserved(device="cuda") / 1000000.0
            )
        )


# The following function is derived from code from Keypt2Subpx (https://github.com/KimSinjeong/keypt2subpx)
# Licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def main(opt):
    logger = get_logger()

    set_seeds(opt.training_seed)

    device = torch.device("cuda:0" if opt.device == "cuda" else "cpu")
    logger.info("Device: " + device.type)

    # construct folder that should contain pre-calculated correspondences
    dataset_names = opt.datasets.split(",")  # support multiple training datasets used jointly
    logger.info("Using datasets: " + ", ".join(dataset_names))

    trainset = get_dataset(
        opt=opt,
        dataset_names=dataset_names,
        split_info_type="train",
        with_score=opt.use_score,
        logger=logger,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        shuffle=True,
        num_workers=2,
        batch_size=opt.batchsize,
    )
    logger.info(f"Train image pairs: {len(trainset)}")

    valset = get_dataset(
        opt=opt,
        dataset_names=dataset_names,
        split_info_type="val",
        with_score=opt.use_score,
        logger=logger,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valset,
        shuffle=False,
        num_workers=2,
        batch_size=1,
    )
    logger.info(f"Val image pairs: {len(valset)}")

    desc_dim = opt.input_channels
    if desc_dim < 0:
        desc_dim = desc_dim_dict[opt.detector]

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
        logger.info(f"Loading model from: {opt.model}")
        model.load_state_dict(torch.load(opt.model)["model"])

    model = model.to(device)

    # ----------------------------------------
    logger.info(f"Starting experiment {opt.experiment}")
    output_dir = os.path.join(opt.output_dir, opt.experiment)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    print("Creating logs in: ", output_dir)
    create_log_dir(output_dir, opt, training=True)
    writer = SummaryWriter(log_dir=str(output_dir))

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    train(model, train_loader, valid_loader, opt, writer, logger, device)


if __name__ == "__main__":
    print("Script is executed at git commit: {}".format(get_git_hash()))
    # parse command line arguments
    # If we have unparsed arguments, print usage and exit
    opt, unparsed = get_config()
    if len(unparsed) > 0:
        print("Unparsed arguments: ", unparsed)
        print_usage()
        exit(1)
    main(opt)
