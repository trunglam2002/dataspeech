import os
import glob
import pandas as pd
import numpy as np
import librosa
import tempfile
import soundfile as sf
import shutil

# --- Cấu hình các thư mục ---
# Folder chứa file Parquet gốc
input_folder = "data/test"      
# Folder chứa các file audio đã tách
audio_output_folder = "audio_outputs"
# Folder chứa các file Parquet đã được cập nhật (không còn cột audio, có thêm file_name)
updated_parquet_folder = "updated_parquet"

# --- Xử lý folder audio_outputs ---
# Nếu folder audio_outputs đã tồn tại, xóa toàn bộ nội dung để ghi đè
if os.path.exists(audio_output_folder):
    shutil.rmtree(audio_output_folder)
os.makedirs(audio_output_folder, exist_ok=True)
os.makedirs(updated_parquet_folder, exist_ok=True)

# --- Lấy danh sách file Parquet ---
parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))
print(f"Found {len(parquet_files)} parquet files in '{input_folder}'.")

# --- Xử lý từng file Parquet ---
for parquet_file in parquet_files:
    print(f"\n🔄 Processing file: {parquet_file}")
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"❌ Error reading file {parquet_file}: {e}")
        continue

    # Danh sách chứa đường dẫn file audio sau khi lưu
    file_paths = []

    # Duyệt qua từng dòng trong DataFrame
    for idx, row in df.iterrows():
        audio_info = row.get("audio", None)
        
        # Xây dựng tên file audio: dựa trên tên file Parquet (không đuôi) + chỉ số dòng
        base_name = os.path.splitext(os.path.basename(parquet_file))[0]
        audio_filename = f"{base_name}_row{idx}.wav"
        audio_file_path = os.path.join(audio_output_folder, audio_filename)
        
        audio_bytes = None
        # Nếu audio_info là dict, ưu tiên lấy key 'bytes'
        if isinstance(audio_info, dict):
            if "bytes" in audio_info and isinstance(audio_info["bytes"], (bytes, bytearray)):
                audio_bytes = audio_info["bytes"]
            elif "path" in audio_info and isinstance(audio_info["path"], str):
                # Nếu không có key 'bytes', thử đọc từ đường dẫn có sẵn
                audio_path = audio_info["path"]
                if os.path.exists(audio_path):
                    try:
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                    except Exception as e:
                        print(f"❌ Error reading audio from '{audio_path}': {e}")
        # Nếu audio_info là bytes
        elif isinstance(audio_info, (bytes, bytearray)):
            audio_bytes = audio_info
        # Nếu audio_info là đường dẫn (string)
        elif isinstance(audio_info, str):
            if os.path.exists(audio_info):
                try:
                    with open(audio_info, "rb") as f:
                        audio_bytes = f.read()
                except Exception as e:
                    print(f"❌ Error reading audio from '{audio_info}': {e}")

        # Lưu file audio nếu có dữ liệu
        if audio_bytes is not None:
            try:
                with open(audio_file_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"✅ Saved audio file: {audio_file_path}")
            except Exception as e:
                print(f"❌ Error saving audio file '{audio_file_path}': {e}")
                audio_file_path = ""
        else:
            print(f"⚠️ No audio data for row {idx} in file {parquet_file}.")
            audio_file_path = ""
        
        # Thêm đường dẫn file audio vào danh sách
        file_paths.append(audio_file_path)
    
    # Thêm cột 'file_name' chứa đường dẫn file audio đã lưu
    df["file_name"] = file_paths
    
    # --- Sắp xếp lại thứ tự các cột để 'file_name' lên đầu ---
    cols = df.columns.tolist()
    if "file_name" in cols:
        cols.insert(0, cols.pop(cols.index("file_name")))
        df = df[cols]
    
    # Lưu lại file Parquet đã cập nhật vào folder updated_parquet
    updated_file = os.path.join(updated_parquet_folder, os.path.basename(parquet_file))
    try:
        df.to_parquet(updated_file, index=False)
        print(f"✅ Updated parquet saved to: {updated_file}")
    except Exception as e:
        print(f"❌ Error saving updated parquet file '{updated_file}': {e}")
