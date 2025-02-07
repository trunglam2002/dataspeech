#!/usr/bin/env python3
import os
import pandas as pd
import argparse

def save_audio(row, output_dir, index):
    """
    Lưu dữ liệu audio của một hàng (row) ra file WAV.
    
    - Nếu cột 'path' có giá trị hợp lệ, ta dùng tên file dựa trên basename của đường dẫn đó.
    - Nếu không, sử dụng tên mặc định theo chỉ số.
    - Ưu tiên sử dụng dữ liệu ở cột 'audio' nếu nó là bytes; nếu không, thử đọc dữ liệu từ đường dẫn trong cột 'path'.
    """
    # Xác định tên file đầu ra
    if "path" in row and isinstance(row["path"], str) and row["path"]:
        base_name = os.path.basename(row["path"])
        name, ext = os.path.splitext(base_name)
        # Đảm bảo đuôi file là .wav
        if ext.lower() != ".wav":
            base_name = name + ".wav"
    else:
        base_name = f"audio_{index}.wav"
    
    output_file = os.path.join(output_dir, base_name)
    
    # Lấy dữ liệu audio từ cột "audio"
    audio_data = row.get("audio", None)
    
    # Nếu audio_data là bytes (hoặc bytearray) thì sử dụng luôn
    if isinstance(audio_data, (bytes, bytearray)):
        data = audio_data
    # Nếu audio_data là dict và chứa key 'bytes'
    elif isinstance(audio_data, dict) and "bytes" in audio_data:
        data = audio_data["bytes"]
    else:
        # Nếu không có dữ liệu hợp lệ trong cột 'audio', thử đọc file từ cột 'path'
        path = row.get("path", "")
        if isinstance(path, str) and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except Exception as e:
                print(f"Lỗi khi đọc file từ đường dẫn '{path}': {e}")
                return
        else:
            print(f"Không có dữ liệu audio hợp lệ cho hàng {index}.")
            return
    
    # Ghi dữ liệu ra file WAV
    try:
        with open(output_file, "wb") as f:
            f.write(data)
        print(f"Đã lưu file WAV: {output_file}")
    except Exception as e:
        print(f"Lỗi khi lưu file {output_file}: {e}")

def process_parquet_file(parquet_file, output_dir, num_rows=10):
    """
    Đọc file Parquet, lấy num_rows dòng đầu tiên và lưu dữ liệu audio ra file WAV.
    """
    # Đảm bảo thư mục output tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        df = pd.read_parquet(parquet_file)
        print(f"Đã đọc file Parquet: {parquet_file}")
    except Exception as e:
        print(f"Lỗi khi đọc file Parquet: {e}")
        return
    
    # Lấy num_rows dòng đầu tiên
    df_head = df.head(num_rows)
    
    # Duyệt qua từng hàng và lưu file audio
    for idx, row in df_head.iterrows():
        save_audio(row, output_dir, idx)

def main():
    parser = argparse.ArgumentParser(
        description="Đọc 10 dòng đầu của file Parquet chứa cột 'audio' (dạng bytes) và 'path', "
                    "lưu ra các file WAV vào thư mục chỉ định."
    )
    parser.add_argument("parquet_file", type=str, help="Đường dẫn đến file Parquet.")
    parser.add_argument("--output_dir", type=str, default="output_test/audio_outputs",
                        help="Thư mục lưu file WAV (mặc định: output_test/audio_outputs).")
    parser.add_argument("--num_rows", type=int, default=10,
                        help="Số dòng đầu cần xử lý (mặc định: 10).")
    args = parser.parse_args()
    
    process_parquet_file(args.parquet_file, args.output_dir, args.num_rows)

if __name__ == "__main__":
    main()
