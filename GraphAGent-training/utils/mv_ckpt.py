import os
import time
import shutil

def monitor_directory(src_dir, dest_dir, interval=60):
    seen_files = set()
    
    while True:
        # 列出目录中的所有文件
        files = [f for f in os.listdir(src_dir) if f.endswith('.ckpt')]
        
        for file in files:
            file_path = os.path.join(src_dir, file)
            
            # 如果文件是新的（之前没见过），则记录当前时间
            if file_path not in seen_files:
                seen_files.add(file_path)
                print(f"Detected new .ckpt file: {file_path}")

        # 检查是否有文件已经存在超过10分钟
        current_time = time.time()
        for file_path in list(seen_files):
            if os.path.exists(file_path) and current_time - os.path.getctime(file_path) > 600:
                dest_path = os.path.join(dest_dir, os.path.basename(file_path))
                shutil.move(file_path, dest_path)
                print(f"Moved {file_path} to {dest_path}")
                seen_files.remove(file_path)

        # 等待指定的时间间隔后再次检查
        time.sleep(interval)

if __name__ == "__main__":
    src_directory = "/hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/checkpoints/higpt-stage2-llama3-new_batch_rw-epoch30-8192-full_finetune"
    dest_directory = "/hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/checkpoints/higpt-stage2-llama3-new_batch_rw-epoch30-8192-full_finetune/lightning_logs/version_1"
    monitor_directory(src_directory, dest_directory)