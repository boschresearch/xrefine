#!/bin/bash

# User specific settings
path_to_dataprocessing="dataprocess"

# Generic settings
script_name="create_refinement_hdf5.py"
keypoints_per_image="4096"
patch_radius="5"
skip_image_pairs_with_less_than_8_matches_for_training_split="True"
sample_matches="no"

# Dataset specific settings
dataset="megadepth_gluefactory"
dataset_mode="train_val"
output_directory="../processed_data/$dataset_mode"

# Change working directory
cd $path_to_dataprocessing

# XFeat specific settings
extractor="xfeat_orig"
extract_score_patches="False"
use_torch_hub_xfeat="True"
matching_approach="mnn"
apply_xfeat_refinement="False"
process_with_gluefactory="False"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --skip_image_pairs_with_less_than_8_matches_for_training_split "$skip_image_pairs_with_less_than_8_matches_for_training_split" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"

# XFeat* specific settings
extractor="xfeat_star"
extract_score_patches="False"
use_torch_hub_xfeat="True"
matching_approach="mnn"
apply_xfeat_refinement="False"
process_with_gluefactory="False"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --skip_image_pairs_with_less_than_8_matches_for_training_split "$skip_image_pairs_with_less_than_8_matches_for_training_split" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"

# SIFT specific settings
extractor="sift"
extract_score_patches="False"
use_torch_hub_xfeat="False"
matching_approach="mnn"
apply_xfeat_refinement="False"
process_with_gluefactory="True"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"

# DISK specific settings
extractor="disk"
extract_score_patches="False"
use_torch_hub_xfeat="False"
matching_approach="lightglue"
apply_xfeat_refinement="False"
process_with_gluefactory="True"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"

# SuperPoint+MNN specific settings
extractor="superpoint"
extract_score_patches="True"
use_torch_hub_xfeat="False"
matching_approach="mnn"
apply_xfeat_refinement="False"
process_with_gluefactory="True"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"

# SuperPoint+LG specific settings
extractor="superpoint"
extract_score_patches="True"
use_torch_hub_xfeat="False"
matching_approach="lightglue"
apply_xfeat_refinement="False"
process_with_gluefactory="True"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"

# ALIKED specific settings
extractor="aliked"
extract_score_patches="True"
use_torch_hub_xfeat="False"
matching_approach="lightglue"
apply_xfeat_refinement="False"
process_with_gluefactory="True"

# Run the script with the specified parameters
python $script_name \
    --local_feature "$extractor" \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --matching_approach "$matching_approach" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches"