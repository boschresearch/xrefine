import argparse
import logging


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser(
    description="Attention-Guided Keypoint Match Refinement.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("experiment", type=str)


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


common_arg = add_argument_group("Common")
common_arg.add_argument(
    "--output_dir",
    default="experiments/train",
    help="directory to store the experiments in",
)
common_arg.add_argument(
    "--data_dir",
    default="processed_data/train_val",
    help="Folder containing the datasets",
)
common_arg.add_argument(
    "--datasets",
    "-ds",
    default="megadepth",
    help="which datasets to use, separate multiple datasets by comma",
)
common_arg.add_argument(
    "--nfeatures",
    "-nf",
    type=int,
    default=2048,
    help="fixes number of features by clamping/replicating, "
    + "set to -1 for dynamic feature count but then batchsize (-bs) has to be set to 1",
)
common_arg.add_argument(
    "--sideinfo",
    action="store_true",
    help="Do not provide side information (matching ratios) to the network. "
    + "The network should be trained and tested consistently.",
)
common_arg.add_argument(
    "--detect_anomaly",
    "-da",
    action="store_true",
    help="Anomaly detection of PyTorch",
)
common_arg.add_argument(
    "--ransac_thr",
    "-rt",
    type=float,
    default=1.0,
    help="GCRANSAC inlier threshold. Recommended value is 1.0px",
)
common_arg.add_argument(
    "--train_thr",
    "-tt",
    type=float,
    default=1.5,
    help="Train inlier threshold. Recommended value is 1.5px",
)
common_arg.add_argument(
    "--model",
    "-m",
    default="",
    help="load a model to contuinue training or leave empty to create a new model",
)
common_arg.add_argument(
    "--detector",
    "-detc",
    choices=[
        "spnn",
        "splg",
        "dedode",
        "sift",
        "dedode",
        "dedodev2",
        "r2d2",
        "aliked",
        "disk",
        "xfeat",
        "xfeat-star",
        "xfeat_orig",
    ],
    default="spnn",
    help='Type of detector; "sp" means SuperPoint, "aliked" means ALIKED',
)
common_arg.add_argument(
    "--input_channels",
    type=int,
    default=-1,
    help="The input channels of the descriptor or of the patch type. \
          If set to -1, the number of channels is determined by the detector.",
)
common_arg.add_argument(
    "--device",
    choices=["cuda", "cpu"],
    default="cuda",
    help="Device to run on",
)
common_arg.add_argument(
    "--patch_radius",
    "-radius",
    type=int,
    default=5,
    help="Size of the patch around the keypoint",
)
common_arg.add_argument(
    "--patch_type",
    type=str,
    default="image",
    help="The type of map from which the input patch should come from.",
    choices=["image", "descriptor", "early_embedding", "late_embedding"],
)
common_arg.add_argument(
    "--cell_size",
    type=int,
    default=1,
    help="The length of an element of the used patch type in pixels, e.g. if it is an image patch the value is 1.",
)
common_arg.add_argument(
    "--no_delta_scaling",
    action="store_true",
    help="If set to True, the keypoint delta is not scaled to the original patch size.",
)
common_arg.add_argument(
    "--shift_KPs_to_pixel_center",
    action="store_true",
    help="Whether to shift the KPs to pixel center in the dataloader. \
          This should be activated during testing, if the refinement model has been trained with that option.",
)
common_arg.add_argument(
    "--color_normalization_strategy",
    choices=["orig", "simple_gray", "simple_color"],
    default="orig",
    help="The color normalization strategy to use.",
)
common_arg.add_argument(
    "--spatial_argmax_type",
    type=str,
    default="soft",
    help="The type of spatial argmax that is applied to the similarity map.",
    choices=["soft", "soft_with_temperature", "hard"],
)
common_arg.add_argument(
    "--processing_mode",
    type=str,
    default="independent_processing",
    help="Whether to process the patches independently or jointly in a single model \
         (attention requires joint processing).",
    choices=["independent_processing", "joint_processing"],
)
common_arg.add_argument(
    "--adjust_only_second_keypoint",
    action="store_true",
    help="If set to True, only the second keypoint is adjusted. Otherwise, both are adjusted.",
)
common_arg.add_argument(
    "--attn_with_desc",
    action="store_true",
    help="Whether to attend the extracted feature map to the concatenated descriptors.",
)
common_arg.add_argument(
    "--attn_with_avg_desc",
    action="store_true",
    help="Whether to attend the extracted feature map to the average descriptor (only applied if no attn_with_desc).",
)
common_arg.add_argument(
    "--attn_with_patch",
    action="store_true",
    help="Whether to include attention between extracted feature maps of both patches.",
)
common_arg.add_argument(
    "--num_attention_blocks",
    type=int,
    default=1,
    help="If attention is applied, this parameter defines the number of attention blocks.",
)
common_arg.add_argument(
    "--positional_encoding_type",
    type=str,
    default="learnable",
    choices=["none", "sinusoidal", "learnable"],
    help="Which type of positional encoding to use in the attention.",
)
common_arg.add_argument(
    "--attn_layer_norm",
    action="store_true",
    help="Whether to apply layer norm after the attention.",
)
common_arg.add_argument(
    "--attn_skip_connection",
    action="store_true",
    help="Whether to use a skip connection to apply the attention.",
)
common_arg.add_argument(
    "--with_match_score",
    action="store_true",
    help="Whether to include a match score head that classifies the patches to be a match or not.",
)
common_arg.add_argument(
    "--learn_match_score_as_confidence",
    action="store_true",
    help="If with_match_score is used, and this is set to True the match score is trained as a refinement confidence.",
)
common_arg.add_argument(
    "--directly_infer_score_map",
    action="store_true",
    help="With this, the score map is not computed as cosim with the average descriptor, but as direct network output.",
)
common_arg.add_argument(
    "--encoder_variant",
    choices=["small", "large"],
    default="small",
    help="The type of encoder (only applied for processing_mode joint_processing).",
)
common_arg.add_argument(
    "--use_score",
    action="store_true",
    help="Use score map patch of feature extractor as additional input to the refinement model.",
)
common_arg.add_argument(
    "--force_nfeatures",
    action="store_true",
    help="Enforce usage of nfeatures. If more are available, the number is reduced, if less are available, \
          some features are duplicated. This setting is only of interest for measuring the runtime.",
)


# Training parameters
train_arg = add_argument_group("Training")
train_arg.add_argument("--epochs", type=int, default=120, help="Terminate the training after this number of epochs")
train_arg.add_argument("--resume", action="store_true", help="Resume from existing experiment")
train_arg.add_argument("--training_seed", type=int, default=42, help="Seed for training")
train_arg.add_argument("--batchsize", "-bs", type=int, default=8, help="batch size")
train_arg.add_argument("--learning_rate", "-lr", type=float, default=0.0001, help="learning rate")
train_arg.add_argument(
    "--adapt_lr_to_batchsize_divisor",
    type=float,
    default=8,
    help="new_lr = lr * (batchsize / divisor); set to a value <= 0 to deactivate",
)
train_arg.add_argument(
    "--lr_scheduler_factor",
    type=float,
    default=1.0,
    help="factor applied at plateau; set to a value >= 1 to deactivate",
)
train_arg.add_argument("--lr_scheduler_patience", type=int, default=10, help="patience to determine plateau")
train_arg.add_argument("--weight_decay", "-wd", type=float, default=0.0, help="weight decay")
train_arg.add_argument("--visu_intv", type=int, default=2500, help="visualisation interval")
train_arg.add_argument(
    "--train_epe_threshold",
    type=float,
    default=-1.0,
    help="use only matches with an end-point-error below the given threshold; set to negative value to deactivate",
)

# Testing parameters
test_arg = add_argument_group("Testing")
test_arg.add_argument("--total_run", "-tr", type=int, default=10, help="Total number of runs on validation set")
test_arg.add_argument("--test", action="store_true", help="Testing mode")
test_arg.add_argument("--total_split", "-ts", type=int, default=1, help="total split size")
test_arg.add_argument("--current_split", "-cs", type=int, default=0, help="current split")
test_arg.add_argument("--vanilla", action="store_true", help="Run vanilla pose estimation pipeline (without K2S model)")
test_arg.add_argument(
    "--image_dir", type=str, default="", help="Main directory of images in dataset. Needed for PixSfM Refinement."
)
test_arg.add_argument(
    "--pixsfm_refinement",
    action="store_true",
    help="Run pixsfm refinement instead of K2S model. \
          This option needs a pixsfm installation and the pixsfm conda environment.",
)
test_arg.add_argument(
    "--min_match_score_threshold",
    type=float,
    default=0.0,
    help="If match score is available, use only matches with a score above this threshold.",
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()


def get_logger():
    formatter = logging.Formatter(fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger("xrefine")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
