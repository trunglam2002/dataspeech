import os
import subprocess

# Đường dẫn tới folder chứa các file .parquet
parquet_folder = "processed2_parquet"
output_base_dir = "processed3_parquet"

# Tạo folder output nếu chưa có
os.makedirs(output_base_dir, exist_ok=True)
# Lặp qua từng file .parquet trong folder
for filename in os.listdir(parquet_folder):
    if filename.endswith(".parquet"):
        # Tách tên file để lấy thông tin speaker
        speaker_id, speaker_name = os.path.splitext(filename)[0].split("_")
        
        # Đường dẫn file input
        file_path = os.path.join(parquet_folder, filename)

        # Output folder riêng cho mỗi speaker
        output_dir = os.path.join(output_base_dir, f"{speaker_id}_{speaker_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Lệnh subprocess để chạy script
        command = [
            "python",
            "./scripts/run_prompt_creation.py",
            "--speaker_name", speaker_name,
            "--is_single_speaker",
            "--is_new_speaker_prompt",
            "--dataset_name", file_path,
            "--model_name_or_path", "mistralai/Mistral-7B-Instruct-v0.2",
            "--per_device_eval_batch_size", "32",
            "--attn_implementation", "sdpa",
            "--output_dir", output_dir,
            "--load_in_4bit",
            "--preprocessing_num_workers", "32",
            "--dataloader_num_workers", "32",
            "--save_only_parquet",
        ]

        # In ra lệnh đang chạy
        print(f"Running for speaker: {speaker_name} (file: {filename})")
        
        # Chạy lệnh
        subprocess.run(command, check=True)

print("Processing completed for all files.")
