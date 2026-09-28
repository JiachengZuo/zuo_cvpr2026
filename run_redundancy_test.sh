#!/bin/bash
# ============================================================================
# Run Gaussian Redundancy Deactivation Test
# ============================================================================
# Based on test_method.md: classify Gaussians by repeat visibility +
# rendering contribution, deactivate redundant ones, observe PSNR changes.
#
# Usage: bash run_redundancy_test.sh
# ============================================================================

source /home/djhuai/anaconda3/bin/activate dggt

CKPT=/home/djhuai/zuo/mobicom/dggt_2/pretrained/model_latest_waymo.pt
DATA_DIR=data/nuscenes/processed_2Hz/mini/mini
OUTPUT_DIR=output_redundancy_test

echo "============================================"
echo "  Gaussian Redundancy Deactivation Test"
echo "============================================"
echo "  Checkpoint: $CKPT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo ""

python deactivate_redundant_gaussians.py \
    --image_dir "$DATA_DIR" \
    --scene_names 003 \
    --ckpt_path "$CKPT" \
    --output_path "$OUTPUT_DIR" \
    --mode 2 \
    --sequence_length 4 \
    --input_views 1 \
    --repeat_thresh 0.5 \
    --contrib_thresh 0.2

echo ""
echo "Done. Results saved to: $OUTPUT_DIR/redundancy_results.json"
