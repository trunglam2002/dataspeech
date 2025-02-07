import os
import io
import pandas as pd
import torchaudio

from datasets import Dataset, Audio

# -----------------------------
# 1) ĐỌC prompts.txt
# -----------------------------
prompts_file = "vivos_data/vivos/train/prompts.txt"  # Bạn sửa lại đúng đường dẫn
transcript_dict = {}

with open(prompts_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Ví dụ: "VIVOSSPK01_R001 KHÁCH SẠN"
        # Tách bằng split(' ', 1) để lấy 2 phần: [0] = "VIVOSSPK01_R001", [1] = "KHÁCH SẠN"
        parts = line.split(" ", 1)
        if len(parts) == 2:
            wav_name, text = parts[0], parts[1]
            # Lưu vào dictionary. Key là tên file (chưa có .wav), value là transcript
            transcript_dict[wav_name] = text
        else:
            print(f"Dòng không đúng định dạng: {line}")

# -----------------------------
# 2) TẠO HÀM LẤY TOÀN BỘ WAV BYTES
# -----------------------------
def extract_audio_as_wav_bytes(file_path):
    """
    Đọc file WAV và trả về dữ liệu WAV đầy đủ (kể cả header) dạng bytes
    """
    waveform, sampling_rate = torchaudio.load(file_path)
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sampling_rate, format="wav")
    buffer.seek(0)
    return buffer.read()

# -----------------------------
# 3) DUYỆT THƯ MỤC WAV VÀ TẠO DANH SÁCH
# -----------------------------
# Khai báo các biến
speaker_id = "46"  # Số 13
name = "Oanh" # Tên Quỳnh

# Ghép vào base_folder và output_parquet
base_folder = f"vivos_data/vivos/train/waves/VIVOSSPK{speaker_id}"
output_parquet = f"dataset_parquet/{name}_voice_{speaker_id}.parquet"
audio_data = []

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            # Lấy tên file không có đuôi .wav, ví dụ "VIVOSSPK01_R001"
            file_stem = os.path.splitext(file)[0]  
            
            # Tìm transcription trong transcript_dict
            if file_stem in transcript_dict:
                transcription = transcript_dict[file_stem]
            else:
                transcription = ""  # hoặc None, nếu không tìm thấy

            # Tạo phiên bản hạ xuống chữ thường (nếu cần)
            transcription_normalised = transcription.lower()
            
            try:
                # Lấy toàn bộ WAV
                audio_bytes = extract_audio_as_wav_bytes(file_path)
                
                # Tạo dict "audio"
                audio_info = {
                    "bytes": audio_bytes,
                    "path": file_path
                }

                # Thêm vào danh sách
                audio_data.append({
                    "file_name": file,
                    "transcription": transcription,
                    "transcription_normalised": transcription_normalised,
                    "audio": audio_info
                })
            except Exception as e:
                print(f"Lỗi khi xử lý file {file_path}: {e}")

# -----------------------------
# 4) TẠO DataFrame -> HuggingFace Dataset
# -----------------------------
df = pd.DataFrame(audio_data)
print(df.head(5))  # Kiểm tra

hf_ds = Dataset.from_pandas(df)

# Cast cột "audio" sang Audio => để decode wav bytes tự động
hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=16000))

# -----------------------------
# 5) LƯU RA PARQUET
# -----------------------------
hf_ds.to_parquet(output_parquet)
print(f"Đã lưu Parquet tại: {output_parquet}")
