#!/bin/bash

data_dir="processed_data/only_eval"

dataset="MegaDepthGlueFactory"
dataset_identifier="mega"
keypoints_per_image="2048"

xrefine_args=(
    --input_channels 64    
    --processing_mode joint_processing
    --directly_infer_score_map
    --attn_with_patch
    --attn_skip_connection
    --positional_encoding_type learnable
    --num_attention_blocks 1
    --encoder_variant small
    --shift_KPs_to_pixel_center
)
model_type="small"

# XFeat+MNN
extractor="xfeat"
hdf5_file="${dataset}_xfeat_orig_mnn_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test
# XFeat*+MNN
extractor="xfeat-star"
hdf5_file="${dataset}_xfeat_star_mnn_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test
# SuperPoint+MNN
extractor="spnn"
hdf5_file="${dataset}_superpoint_mnn_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test
# SuperPoint+LG
extractor="splg"
hdf5_file="${dataset}_superpoint_lightglue_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test
# ALIKED+LG
extractor="aliked"
hdf5_file="${dataset}_aliked_lightglue_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test
# SIFT+MNN
extractor="sift"
hdf5_file="${dataset}_sift_mnn_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test
# DISK+LG
extractor="disk"
hdf5_file="${dataset}_disk_lightglue_"$keypoints_per_image".hdf5"
python test.py "$dataset_identifier"_"$extractor"_"$keypoints_per_image"_xrefine-"$model_type"-"$extractor" \
    --data_dir "$data_dir" \
    --datasets "$hdf5_file" \
    "${xrefine_args[@]}" \
    -m pretrained/xrefine_"$model_type"_specific_"$extractor".pth \
    --test