import os
import glob
import pandas as pd
import numpy as np
import librosa

# Danh sách nhãn tốc độ nói
levels = ["very slowly", "slowly", "slightly slowly", "moderate speed", "slightly fast", "fast", "very fast"]

# Thay vì dùng test_folder, ta chỉ cần:
# 1. processed_folder: Chứa các tệp .parquet cần tính speaking_rate
# 2. wav_folder: Chứa các file .wav tương ứng

processed_folder = "processed2_parquet"  # Đường dẫn đến thư mục processed
wav_folder = "audio_outputs"                 # Đường dẫn tới thư mục chứa file .wav


def get_audio_duration(file_name, wav_folder):
    """
    Tính thời lượng của file audio dựa trên tên file và thư mục chứa .wav
    """
    path = os.path.join(wav_folder, file_name)
    path = fix_filename(path)
    if os.path.exists(path):
        try:
            duration = librosa.get_duration(filename=path)
            return duration
        except Exception as e:
            print(f"Lỗi khi đọc file âm thanh '{path}': {e}")
    else:
        print(f"⚠️ File không tồn tại: {path}")
    return 0

def fix_filename(file_name: str) -> str:
    """
    Nếu file_name có dạng 'audio_outputs/audio_outputs/...'
    thì thay thế bằng 'audio_outputs/...'
    """
    # Bạn có thể điều chỉnh logic thay thế cụ thể hơn nếu cần.
    return file_name.replace("audio_outputs/audio_outputs/", "audio_outputs/")

def compute_speaking_rate(text, file_name, wav_folder):
    """
    Tính speaking rate = số từ trong text / duration của audio.
    Nếu duration <= 0, trả về 0.
    """
    num_words = len(text.split())
    duration = get_audio_duration(file_name, wav_folder)
    return num_words / duration if duration > 0 else 0


def main():
    # Tìm tất cả file .parquet trong processed_folder
    processed_files = glob.glob(os.path.join(processed_folder, "*.parquet"))
    print(f"Tìm thấy {len(processed_files)} file parquet trong thư mục: {processed_folder}")

    for p_file in processed_files:
        file_name = os.path.basename(p_file)
        print(f"\n🔄 Xử lý file: {file_name}")

        # Đọc file processed
        try:
            df_processed = pd.read_parquet(p_file)
        except Exception as e:
            print(f"Không thể đọc file processed {p_file}: {e}")
            continue

        # Đảm bảo cột 'text' không bị thiếu
        if "text" not in df_processed.columns:
            print(f"❌ File {p_file} không có cột 'text', bỏ qua.")
            continue

        df_processed["text"] = df_processed["text"].fillna("")

        # Đảm bảo cột 'file_name' tồn tại, vì ta cần tên file để ghép với wav_folder
        if "file_name" not in df_processed.columns:
            print(f"❌ File {p_file} không có cột 'file_name', bỏ qua.")
            continue

        # Tính computed_rate cho mỗi dòng
        df_processed["computed_rate"] = df_processed.apply(
            lambda row: compute_speaking_rate(row["text"], row["file_name"], wav_folder),
            axis=1
        )

        # Phân loại computed_rate thành 7 nhóm
        try:
            df_processed["speaking_rate"] = pd.qcut(df_processed["computed_rate"], q=len(levels), labels=levels)
        except Exception:
            # Nếu pd.qcut gặp lỗi (ví dụ: tất cả giá trị computed_rate giống nhau)
            min_val = df_processed["computed_rate"].min()
            max_val = df_processed["computed_rate"].max()

            if min_val == max_val:
                # Tất cả giá trị bằng nhau => gán luôn cùng một nhãn
                df_processed["speaking_rate"] = levels[0]
            else:
                bins = np.linspace(min_val, max_val, len(levels) + 1)
                df_processed["speaking_rate"] = pd.cut(
                    df_processed["computed_rate"], bins=bins, labels=levels, include_lowest=True
                )

        # Lưu file processed đã được cập nhật
        try:
            df_processed.to_parquet(p_file, index=False)
            print(f"✅ Đã cập nhật file: {p_file}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file {p_file}: {e}")

if __name__ == "__main__":
    main()
