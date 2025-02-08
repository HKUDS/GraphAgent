#!/bin/bash

# 定义目录
SOURCE_DIR="/hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/checkpoints/higpt-stage2-llama3-new_batch_rw-epoch30-8192-full_finetune"
DEST_DIR="/hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/checkpoints/higpt-stage2-llama3-new_batch_rw-epoch30-8192-full_finetune/lightning_logs/version_1"
CHECK_INTERVAL=60  # 每隔 60 秒检查一次

# 创建一个已处理文件的列表
processed_files=()

# 检查文件是否在已处理列表中
is_processed() {
    local file=$1
    for processed_file in "${processed_files[@]}"; do
        if [[ "$processed_file" == "$file" ]]; then
            return 0
        fi
    done
    return 1
}

while true; do
    for file in "$SOURCE_DIR"/*.ckpt; do
        if [[ -f "$file"]]; then
            echo "检测到新文件：$(basename "$file")"
            
            # 将文件加入已处理列表
            processed_files+=("$(basename "$file")")
            
            # 等待10分钟
            sleep 600
            
            # 移动文件到目标目录
            mv "$file" "$DEST_DIR"
            echo "文件 $(basename "$file") 已移动到 $DEST_DIR"
        fi
    done
    
    # 等待 CHECK_INTERVAL 秒后再检查
    sleep "$CHECK_INTERVAL"
done