data_dir="processed_data/train_val"
dataset="MegaDepthGlueFactory_sampled_normally-1p50_4096.hdf5"

xrefine_model_name="xrefine-small-general"
output_directory="experiments/train"

xrefine_small_args=(
    --processing_mode joint_processing
    --input_channels 64    
    --directly_infer_score_map
    --attn_with_patch
    --attn_skip_connection
    --positional_encoding_type learnable
    --num_attention_blocks 1
    --encoder_variant small
    --shift_KPs_to_pixel_center
)

python train.py "$xrefine_model_name" \
    --data_dir "$data_dir" \
    --datasets "$dataset" \
    --output_dir "$output_directory" \
    "${xrefine_small_args[@]}"
