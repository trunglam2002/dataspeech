import pandas as pd
import os
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Clean .parquet files and remove unused .wav files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing .parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where cleaned .parquet files will be saved.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Path to the directory containing .wav files.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs='+',
        required=True,
        help="List of column names to check for NaN values. Example: --columns snr c50",
    )
    return parser.parse_args()

def clean_and_save_parquet_files(input_dir, output_dir, columns):
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

    if not parquet_files:
        print(f"No .parquet files found in the input directory: {input_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} .parquet file(s) in {input_dir}.")
    valid_filenames = set()

    for file_name in parquet_files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        try:
            df = pd.read_parquet(input_path)
            print(f"\nProcessing file: {file_name}")

            print("Total number of NaN values in each specified column before cleaning:")
            print(df[columns].isnull().sum())

            files_with_nan = df[df[columns].isnull().any(axis=1)]
            if not files_with_nan.empty:
                print("Rows with NaN values in specified columns:")
                print(files_with_nan['file_name'].tolist() if 'file_name' in df.columns else "No 'file_name' column available.")
            else:
                print("No NaN values found in specified columns.")

            df_cleaned = df.dropna(subset=columns)

            print("Total number of NaN values in each specified column after cleaning:")
            print(df_cleaned[columns].isnull().sum())

            df_cleaned.to_parquet(output_path, index=False)
            print(f"Cleaned file saved to: {output_path}")
            
            if 'file_name' in df_cleaned.columns:
                valid_filenames.update(os.path.basename(f) for f in df_cleaned['file_name'].tolist())

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    return valid_filenames

def clean_wav_files(wav_dir, valid_filenames):
    if not os.path.exists(wav_dir):
        print(f"WAV directory does not exist: {wav_dir}")
        return
    
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    removed_files = 0
    
    for wav_file in wav_files:
        if wav_file not in valid_filenames:
            file_path = os.path.join(wav_dir, wav_file)
            try:
                os.remove(file_path)
                removed_files += 1
                print(f"Removed unused WAV file: {wav_file}")
            except Exception as e:
                print(f"Failed to remove {wav_file}: {e}")
    
    print(f"Total removed WAV files: {removed_files}")

def main():
    args = parse_arguments()
    valid_filenames = clean_and_save_parquet_files(args.input_dir, args.output_dir, args.columns)
    clean_wav_files(args.wav_dir, valid_filenames)

if __name__ == "__main__":
    main()
