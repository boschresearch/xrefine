# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch

from model import SimpleJointAttnTuner
from settings import get_config, get_logger, print_usage
from utils import desc_dim_dict

CUDA_LAUNCH_BLOCKING = 2, 3
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def export_attn_tuner(
    output_dir: str,
    attn_tuner: torch.nn.Module,
    patch_tensor_a: torch.Tensor,
    patch_tensor_b: torch.Tensor,
    make_num_patches_dynamic: bool,
    opset_version: int,
):
    input_names = ["patch1", "patch2"]
    output_names = ["coord1", "coord2"]
    with torch.no_grad():
        if make_num_patches_dynamic:
            dynamic_axes = {
                "patch1": {1: "num_patches"},
                "patch2": {1: "num_patches"},
                "coord1": {1: "num_patches"},
                "coord2": {1: "num_patches"},
            }
        else:
            dynamic_axes = None

        try:
            torch.onnx.export(
                attn_tuner,  # Model being run
                (patch_tensor_a, patch_tensor_b),  # Model input (or a tuple for multiple inputs)
                output_dir,  # Where to save the model
                export_params=True,  # Store the trained parameter weights inside the model file
                opset_version=opset_version,  # The ONNX version to export the model to
                do_constant_folding=True,  # Whether to execute constant folding for optimization
                input_names=input_names,  # The model's input names
                output_names=output_names,  # The model's output names
                dynamic_axes=dynamic_axes,  # Variable length axes
                verbose=False,
            )
        except torch.onnx.errors.OnnxExporterError:
            print("Please install onn with 'pip install onnx' to export the model to onnx.")
            raise


def compare_models(
    attn_tuner: torch.nn.Module,
    onnx_model_path: str,
    patch_tensor_a: torch.Tensor,
    patch_tensor_b: torch.Tensor,
):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install onnxruntime with 'pip install onnxruntime' to export the model to onnx."
        )

    # Clone the input tensor
    onnx_input_tensor_a = patch_tensor_a.clone()
    onnx_input_tensor_b = patch_tensor_b.clone()

    # Set the PyTorch model to evaluation mode
    attn_tuner.eval()

    # Get PyTorch model output
    with torch.no_grad():
        pytorch_output = attn_tuner(patch_tensor_a, patch_tensor_b)

    # Prepare the ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Prepare the input for ONNX model
    onnx_input_np_a = onnx_input_tensor_a.cpu().numpy()
    onnx_input_np_b = onnx_input_tensor_b.cpu().numpy()
    ort_inputs = {
        ort_session.get_inputs()[0].name: onnx_input_np_a,
        ort_session.get_inputs()[1].name: onnx_input_np_b,
    }

    # Get ONNX model output
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare the outputs
    for i, output_name in enumerate(ort_session.get_outputs()):
        onnx_output = ort_outs[i]
        pytorch_output_tensor = pytorch_output[i] if isinstance(pytorch_output, (tuple, list)) else pytorch_output
        pytorch_output_numpy = pytorch_output_tensor.cpu().numpy()

        print(f"Comparing output: {output_name.name}")
        print(f"PyTorch output shape: {pytorch_output_numpy.shape}")
        print(f"ONNX output shape: {onnx_output.shape}")
        l2_norm = np.linalg.norm(pytorch_output_numpy - onnx_output)
        print(f"L2 Norm: {l2_norm}")
        cosine_similarity = np.dot(pytorch_output_numpy.flatten(), onnx_output.flatten()) / (
            np.linalg.norm(pytorch_output_numpy.flatten()) * np.linalg.norm(onnx_output.flatten())
        )
        print(f"Cosine Similarity: {cosine_similarity}")
        print("\n")


def main(opt):
    logger = get_logger()

    desc_dim = opt.input_channels
    if desc_dim < 0:
        desc_dim = desc_dim_dict[opt.detector]

    # create or load model
    if opt.processing_mode == "joint_processing":
        model = SimpleJointAttnTuner(
            desc_dim=desc_dim,
            color_normalization_strategy=opt.color_normalization_strategy,
            spatial_argmax_type=opt.spatial_argmax_type,
            attn_with_patch=opt.attn_with_patch,
            num_attention_blocks=opt.num_attention_blocks,
            positional_encoding_type=opt.positional_encoding_type,
            attn_layer_norm=opt.attn_layer_norm,
            attn_skip_connection=opt.attn_skip_connection,
            patch_radius=opt.patch_radius,
            encoder_variant=opt.encoder_variant,
            adjust_only_second_keypoint=opt.adjust_only_second_keypoint,
            image_values_are_normalized=False,
        )
    else:
        raise NotImplementedError(f"Exporting AttnTuner with processing mode {opt.processing_mode} is not implemented.")
    if len(opt.model) > 0:
        model.load_state_dict(torch.load(opt.model, weights_only=False)["model"])
        logger.info(f"Loaded model from {opt.model}")
    model = model.to(opt.device)
    model.eval()

    batch_size = 1
    num_patches = 100
    image_channels = 1
    patch_height = opt.patch_radius * 2 + 1
    patch_width = opt.patch_radius * 2 + 1
    output_file = "./xrefine_small_general_1_N_1_11_11.onnx"

    print("Refinement net ready. Generate random input sample.")
    input_sample_patch_a = (
        torch.rand(
            batch_size,
            num_patches,
            image_channels,
            patch_height,
            patch_width,
            device=opt.device,
        )
        * 255
    )
    input_sample_patch_b = (
        torch.rand(
            batch_size,
            num_patches,
            image_channels,
            patch_height,
            patch_width,
            device=opt.device,
        )
        * 255
    )

    print(f"Start exporting to {output_file}.")
    export_attn_tuner(
        output_file,
        model,
        input_sample_patch_a,
        input_sample_patch_b,
        make_num_patches_dynamic=True,
        opset_version=17,
    )

    # Compare the models
    compare_models(model, output_file, input_sample_patch_a, input_sample_patch_b)


if __name__ == "__main__":
    # parse command line arguments
    # If we have unparsed arguments, print usage and exit
    opt, unparsed = get_config()
    print(unparsed)
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    main(opt)
