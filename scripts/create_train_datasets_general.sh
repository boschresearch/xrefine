# User specific settings
path_to_dataprocessing="dataprocess"

# Generic settings
script_name="create_refinement_hdf5.py"
keypoints_per_image="4096"
patch_radius="5"
skip_image_pairs_with_less_than_8_matches_for_training_split="True"

# Dataset specific settings
dataset="megadepth_gluefactory"
dataset_mode="train_val"
output_directory="../processed_data/$dataset_mode"

# Change working directory
cd $path_to_dataprocessing

# Normal sampling
sample_matches="normally"
extract_score_patches="False"
use_torch_hub_xfeat="False"
apply_xfeat_refinement="False"
process_with_gluefactory="False"

# Run the script with the specified parameters
python $script_name \
    --keypoints_per_image "$keypoints_per_image" \
    --output_directory "$output_directory" \
    --dataset_mode "$dataset_mode" \
    --patch_radius "$patch_radius" \
    --skip_image_pairs_with_less_than_8_matches_for_training_split "$skip_image_pairs_with_less_than_8_matches_for_training_split" \
    --extract_score_patches "$extract_score_patches" \
    --use_torch_hub_xfeat "$use_torch_hub_xfeat" \
    --apply_xfeat_refinement "$apply_xfeat_refinement" \
    --process_with_gluefactory "$process_with_gluefactory" \
    --sample_matches "$sample_matches" \
    --normal_distortion_std 0.3 # Will be multiplied by patch_radius in the script -> total std is 1.5