import pandas as pd

# Đọc file Parquet
df = pd.read_parquet("processed2_parquet/train_edited.parquet")

# Hiển thị thông tin ban đầu
print("Dữ liệu ban đầu:")
print(df.head())

value = "pesq_speech_quality"
# Tổng hợp thông tin về cột value
print(f"\nTổng hợp thông tin về cột {value}:")
if value in df.columns:
    # Số lượng giá trị duy nhất
    unique_values = df[value].unique()
    print(f"Số lượng giá trị duy nhất: {len(unique_values)}")
    print(f"Giá trị duy nhất: {unique_values}")

    # Đếm số lượng từng giá trị
    value_counts = df[value].value_counts()
    print("\nSố lượng từng giá trị:")
    print(value_counts)
else:
    print(f"{value} không tồn tại trong DataFrame.")
