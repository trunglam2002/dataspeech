import os

# Define the directory containing the parquet files
directory = "/home/pc/Desktop/dataspeech/dataspeech/processed2_parquet/"

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".parquet"):
        # Extract the relevant parts of the filename
        parts = filename.split("_")
        if len(parts) >= 3:
            name = parts[0]
            voice_number = parts[-1].replace("voice_", "").replace(".parquet", "")
            new_name = f"{voice_number}_{name}.parquet"

            # Get full paths for renaming
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

print("Renaming completed.")
