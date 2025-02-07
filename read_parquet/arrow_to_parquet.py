import os
import datasets

# Đường dẫn đến thư mục chứa file .arrow
input_dir = "/home/pc/Desktop/dataspeech/dataspeech/data/vlsp2020_train"
output_dir = "/home/pc/Desktop/dataspeech/dataspeech/data/vlsp2020_train_parquet"

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Lặp qua tất cả các tệp .arrow và chuyển đổi sang .parquet
for file_name in os.listdir(input_dir):
    if file_name.endswith(".arrow"):
        arrow_path = os.path.join(input_dir, file_name)
        parquet_path = os.path.join(output_dir, file_name.replace(".arrow", ".parquet"))

        # Load dữ liệu từ file .arrow
        dataset = datasets.Dataset.from_file(arrow_path)

        # Lưu dưới dạng .parquet
        dataset.to_parquet(parquet_path)

        print(f"Đã chuyển đổi: {file_name} → {parquet_path}")
