import pandas as pd
import argparse

def read_parquet_file(file_path):
    """
    Đọc file Parquet từ đường dẫn và trả về DataFrame.

    Tham số:
      - file_path (str): Đường dẫn đến file Parquet.

    Trả về:
      - pd.DataFrame: DataFrame chứa dữ liệu nếu đọc thành công.
      - None: Nếu có lỗi xảy ra trong quá trình đọc file.
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Đã đọc file Parquet: {file_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file Parquet '{file_path}': {e}")
        return None

def main():
    # Thiết lập argparse để nhận đường dẫn file từ dòng lệnh
    parser = argparse.ArgumentParser(description="Đọc file Parquet và in ra vài dòng dữ liệu mẫu.")
    parser.add_argument("file_path", type=str, help="Đường dẫn đến file Parquet cần đọc")
    args = parser.parse_args()

    # Đọc file Parquet
    df = read_parquet_file(args.file_path)
    if df is not None:
        print("\nDữ liệu mẫu:")
        print(df['speaking_rate'][:10])
    else:
        print("Không thể đọc file Parquet. Vui lòng kiểm tra lại đường dẫn hoặc định dạng file.")

if __name__ == "__main__":
    main()
