import pandas as pd
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Count the number of rows in .parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing .parquet files.",
    )
    return parser.parse_args()

def count_parquet_rows(input_dir):
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

    if not parquet_files:
        print(f"No .parquet files found in the input directory: {input_dir}")
        return

    print(f"Found {len(parquet_files)} .parquet file(s) in {input_dir}.")
    
    for file_name in parquet_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            df = pd.read_parquet(file_path)
            print(f"File: {file_name}, Rows: {len(df)}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

def main():
    args = parse_arguments()
    count_parquet_rows(args.input_dir)

if __name__ == "__main__":
    main()
