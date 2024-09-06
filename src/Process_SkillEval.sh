#!/bin/bash

# Assigning arguments to variables
TASK_TYPE=$1
INPUT_FOLDER=/input
output_dir=/output

# Check if the input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi
# Extracting the base directory from the input folder path
base_dir=$(dirname "$INPUT_FOLDER")

# Creating new directories for segmentation JSON and coordinates if they don't exist
new_dir="/seg_json"
coord_dir="/coord"
file="df4final.csv"

# Running inference for MP4 files if the segmentation JSON directory doesn't exist
if [ ! -d "$new_dir" ]; then
    poetry run python inference_mp4.py --target-dir "$INPUT_FOLDER" --out "$new_dir" --local-rank 0 --pred_score_thr 0.3 /app/src/weight/best_coco_bbox_mAP_50_epoch_10.pth 0
fi

# Running the script for coordinates if the coordinates directory doesn't exist
if [ ! -d "$coord_dir" ]; then
    poetry run python coords.py "$new_dir"
fi

# Running CNN and LGBM inference scripts
poetry run python inf_CNN.py "$coord_dir" "$file"
poetry run python inf_LGBM.py "$coord_dir" "$file" "$TASK_TYPE" "$output_dir"
