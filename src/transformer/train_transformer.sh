#!/bin/bash
python3.9 train_transformer.py \
    --model_name ${1} \
    --train_src_fn "../../data/aihub_koen/train.ko" \
    --train_tgt_fn "../../data/aihub_koen/train.en" \
    --valid_src_fn "../../data/aihub_koen/valid.ko" \
    --valid_tgt_fn "../../data/aihub_koen/valid.en" \
    --tokenizer "../../data/aihub_koen_32k" \
    --output_dir "../../checkpoints" \
    --n_epochs 1 \
    --batch_size_per_device 32 \
    --gradient_accumulation_steps 4 \
    --fp16
