#!/usr/bin/bash
# ori bs 16 max words 384
# rxr bs 4 max words 1000

LLAMA_PATH="$1"
PRETRAINED_PATH="$2" # path to pre-trained checkpoint
CONFIG="$3"
OUTPUT_DIR="$4"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=4 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size 16 --max_words 384 \
 --epochs 20 --warmup_epochs 2 --blr 1e-4 --weight_decay 0.02 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 &>> "$OUTPUT_DIR"/output.log &