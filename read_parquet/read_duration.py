import pandas as pd
import librosa
import tempfile
import soundfile as sf
import os

def test_read_audio_from_parquet(parquet_file):
    """
    Đọc file Parquet chứa cột 'audio' (mỗi dòng là dict có 'bytes' và 'path'),
    xử lý theo từng loại dữ liệu.

    Args:
        parquet_file (str): Đường dẫn đến file Parquet.
    """
    try:
        # Đọc dữ liệu từ file Parquet
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Lỗi khi đọc file Parquet: {parquet_file}\nError: {e}")
        return

    if 'audio' not in df.columns:
        print("File Parquet không có cột 'audio'.")
        return

    print(f"Đã load được {len(df)} dòng từ file: {parquet_file}")

    # Kiểm tra kiểu dữ liệu của cột 'audio'
    print("\n--- Kiểu dữ liệu của cột 'audio' ---")
    print(df.dtypes)

    # Duyệt qua từng dòng trong DataFrame
    for idx, row in df.iterrows():
        audio_info = row['audio']

        if not isinstance(audio_info, dict):  # Nếu không phải dict, bỏ qua
            print(f"\nRow {idx}: Dữ liệu không hợp lệ, bỏ qua!")
            continue

        audio_bytes = audio_info.get('bytes', None)
        audio_path = audio_info.get('path', None)

        if audio_bytes and isinstance(audio_bytes, (bytes, bytearray)):  
            # Xử lý nếu có dữ liệu nhị phân
            print(f"\nRow {idx}: Đọc dữ liệu nhị phân...")

            try:
                # Lưu dữ liệu nhị phân vào file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                # Đọc file audio từ file tạm
                audio_array, sr = librosa.load(tmp_path, sr=None)
                duration = librosa.get_duration(y=audio_array, sr=sr)

                print(f"  - Sample rate: {sr}")
                print(f"  - Duration: {duration:.2f} giây")
                print(f"  - Kích thước audio: {audio_array.shape}")

                # Xóa file tạm sau khi xử lý xong
                os.remove(tmp_path)

            except Exception as e:
                print(f"  Lỗi khi đọc dữ liệu âm thanh từ file WAV.\n  Error: {e}")

        elif audio_path and isinstance(audio_path, str):
            # Xử lý nếu có đường dẫn file
            print(f"\nRow {idx}: Đọc từ đường dẫn file: {audio_path}")

            if os.path.exists(audio_path):
                try:
                    audio_array, sr = librosa.load(audio_path, sr=None)
                    duration = librosa.get_duration(y=audio_array, sr=sr)

                    print(f"  - Sample rate: {sr}")
                    print(f"  - Duration: {duration:.2f} giây")
                    print(f"  - Kích thước audio: {audio_array.shape}")

                except Exception as e:
                    print(f"  Lỗi khi đọc file âm thanh từ đường dẫn.\n  Error: {e}")
            else:
                print(f"  ⚠️ File không tồn tại: {audio_path}")

        else:
            print(f"\nRow {idx}: Không có dữ liệu hợp lệ (không có 'bytes' hoặc 'path').")

if __name__ == '__main__':
    # Đường dẫn đến file Parquet cần test
    parquet_file_path = "data/test/data-00000-of-00024.parquet"
    
    test_read_audio_from_parquet(parquet_file_path)
