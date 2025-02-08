#!/bin/bash
base_model=/path/to/llama3-8b-instruct

num_epochs=5
context_len=8192

python -u pl_train.py \
    --model_name_or_path ${base_model} \
    --flash_attn True \
    --data_name stage_1_mix_with_higpt \
    --tune_graph_mlp_adapter True \
    --tune_embed_tokens True \
    --bf16 True \
    --output_dir ./checkpoints/higpt-stage1-llama3-MIX-WITH-HiGPT-epoch${num_epochs}-${context_len} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_every_n_epochs 2 \
    --learning_rate 3e-5 \
    --logging_steps 1 \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --is_graph True \
    --gpus 0,1,2,3 \
    --resume checkpoints/higpt-stage1-llama3-MIX-WITH-HiGPT-epoch5-8192/lightning_logs/version_1/model_epoch=1-step=1264.ckpt