import os
import pandas as pd

def process_parquet_folders(folder_1, folder_2, output_folder):
    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Duyệt qua tất cả các file trong folder_1
    for file in os.listdir(folder_1):
        if file.endswith(".parquet"):  # Chỉ xử lý file parquet
            file_path_1 = os.path.join(folder_1, file)
            file_path_2 = os.path.join(folder_2, file)
            output_path = os.path.join(output_folder, file)

            # Kiểm tra nếu file tồn tại trong folder_2
            if os.path.exists(file_path_2):
                # Đọc dữ liệu từ cả hai file
                df1 = pd.read_parquet(file_path_1)
                df2 = pd.read_parquet(file_path_2)

                # Lấy cột 'file_name' từ df1 (nếu tồn tại)
                if 'file_name' in df1.columns:
                    file_name_col = df1[['file_name']]
                    # Ghép cột 'file_name' vào df2
                    df2 = pd.concat([file_name_col, df2], axis=1)

                # Lưu file parquet mới
                df2.to_parquet(output_path, index=False)
                print(f"Đã xử lý: {file}")

            else:
                print(f"Bỏ qua: {file} không tồn tại trong {folder_2}")

# Thay đổi đường dẫn thư mục theo nhu cầu
folder_1 = "temp_dataset"
folder_2 = "test_dataset"
output_folder = "test_dataset"

process_parquet_folders(folder_1, folder_2, output_folder)
