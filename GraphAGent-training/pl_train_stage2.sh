#!/bin/bash
base_model=/path/to/llama3-8b-instruct

# path to the checkpoint after stage 1 training (alignment)
pretrain_graph_mlp_adapter=checkpoints/freeze/llama3-8b-higpt-stage1-llama3-MIX-WITH-HiGPT-8192-version_1-epoch=3.bin

num_epochs=5
context_len=8192

python -u pl_train.py \
    --model_name_or_path ${base_model} \
    --data_name stage_2_dual_graph_imdb_few_shot_40 \
    --tune_graph_mlp_adapter True \
    --tune_embed_tokens True \
    --full_finetune True \
    --bf16 True \
    --output_dir ./checkpoints/higpt-stage2-llama3-IMDB_Few_Shot-epoch${num_epochs}-${context_len}-full_finetune \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --save_every_n_epochs 5 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --is_graph True \
    --gpus 0,1,2,3 \
    --pretrain_graph_mlp_adapter ${pretrain_graph_mlp_adapter} \
    --resume checkpoints/higpt-stage2-llama3-IMDB_Few_Shot-epoch30-8192-full_finetune/lightning_logs/version_1/model_epoch=29-step=120.ckpt