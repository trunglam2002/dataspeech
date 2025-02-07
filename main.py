from datasets import load_dataset, Audio, DatasetDict
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply, squim_apply
import torch
import argparse
import os
import glob

def process_dataset(dataset, args, audio_column_name, text_column_name):
    # Apply transformations as per the original script

    if args.apply_squim_quality_estimation:
        print("Compute SI-SDR, PESQ, STOI")
        squim_dataset = dataset.map(
            squim_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_squim if torch.cuda.device_count() > 0 else args.cpu_num_workers,
            remove_columns=[audio_column_name],  # avoid rewriting audio
            fn_kwargs={"audio_column_name": audio_column_name,},
        )

    print("Compute pitch")
    pitch_dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=16_000)).map(
        pitch_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count() > 0 else False,
        num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_pitch if torch.cuda.device_count() > 0 else args.cpu_num_workers,
        remove_columns=[audio_column_name],
        fn_kwargs={"audio_column_name": audio_column_name, "penn_batch_size": args.penn_batch_size},
    )

    print("Compute snr and reverb")
    snr_dataset = dataset.map(
        snr_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count() > 0 else False,
        num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_snr if torch.cuda.device_count() > 0 else args.cpu_num_workers,
        remove_columns=[audio_column_name],
        fn_kwargs={"audio_column_name": audio_column_name},
    )

    print("Compute speaking rate")
    if "speech_duration" in snr_dataset[next(iter(snr_dataset.keys()))].features:
        rate_dataset = snr_dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size=args.cpu_writer_batch_size,
            fn_kwargs={"audio_column_name": audio_column_name, "text_column_name": text_column_name},
        )
    else:
        rate_dataset = dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size=args.cpu_writer_batch_size,
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name, "text_column_name": text_column_name},
        )

    for split in dataset.keys():
        dataset[split] = pitch_dataset[split].add_column("snr", snr_dataset[split]["snr"]).add_column("c50", snr_dataset[split]["c50"])
        if "speech_duration" in snr_dataset[split]:
            dataset[split] = dataset[split].add_column("speech_duration", snr_dataset[split]["speech_duration"])
        dataset[split] = dataset[split].add_column("speaking_rate", rate_dataset[split]["speaking_rate"]).add_column("phonemes", rate_dataset[split]["phonemes"])
        if args.apply_squim_quality_estimation:
            dataset[split] = dataset[split].add_column("stoi", squim_dataset[split]["stoi"]).add_column("si-sdr", squim_dataset[split]["sdr"]).add_column("pesq", squim_dataset[split]["pesq"])

    return dataset

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()

    # Existing arguments
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")

    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Column name of the audio column to be enriched.")
    parser.add_argument("--text_column_name", default="text", type=str, help="Text column name.")
    parser.add_argument("--rename_column", action="store_true", help="If activated, rename audio and text column names to 'audio' and 'text'. Useful if you want to merge datasets afterwards.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--cpu_writer_batch_size", default=1000, type=int, help="writer_batch_size for transformations that don't use GPUs. See: https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/main_classes#datasets.Dataset.map.writer_batch_size")
    parser.add_argument("--batch_size", default=2, type=int, help="This parameter specifies how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--penn_batch_size", default=4096, type=int, help="Pitch estimation chunks audio into smaller pieces and processes them in batch. This specifies the batch size. If you are using a GPU, pick a batch size that doesn't cause memory errors.")
    parser.add_argument("--num_workers_per_gpu_for_pitch", default=1, type=int, help="Number of workers per GPU for the pitch estimation if GPUs are available. Defaults to 1 if some are available. Useful if you want multiple processes per GPU to maximize GPU usage.")
    parser.add_argument("--num_workers_per_gpu_for_snr", default=1, type=int, help="Number of workers per GPU for the SNR and reverberation estimation if GPUs are available. Defaults to 1 if some are available. Useful if you want multiple processes per GPU to maximize GPU usage.")
    parser.add_argument("--apply_squim_quality_estimation", action="store_true", help="If set, will also use torchaudio-squim estimation (SI-SNR, STOI and PESQ).")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int, help="Number of workers per GPU for the SI-SNR, STOI and PESQ estimation if GPUs are available. Defaults to 1 if some are available. Useful if you want multiple processes per GPU to maximize GPU usage.")

    # New arguments for directory processing
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_name", type=str, help="Name of the dataset to load from Hugging Face Datasets.")
    group.add_argument("--input_dir", type=str, help="Path to the directory containing Parquet files to process individually.")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory where processed Parquet files will be saved.")

    args = parser.parse_args()

    # Ensure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_dir:
        # Process each Parquet file in the input directory separately
        parquet_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in the directory: {args.input_dir}")

        for parquet_file in parquet_files:
            print(f"Processing file: {parquet_file}")
            dataset = load_dataset("parquet", data_files=parquet_file, split="train")

            # Convert to DatasetDict if necessary
            if not isinstance(dataset, DatasetDict):
                dataset = DatasetDict({"train": dataset})

            # Handle renaming of columns
            audio_column_name = "audio" if args.rename_column else args.audio_column_name
            text_column_name = "text" if args.rename_column else args.text_column_name
            if args.rename_column:
                dataset = dataset.rename_columns({args.audio_column_name: "audio", args.text_column_name: "text"})

            # Process the dataset
            processed_dataset = process_dataset(dataset, args, audio_column_name, text_column_name)

            # Define output file path
            base_filename = os.path.basename(parquet_file)
            output_file = os.path.join(args.output_dir, base_filename)

            # Save the processed dataset
            print(f"Saving processed file to: {output_file}")
            for split in processed_dataset.keys():
                # Tạo tên file .parquet cho mỗi split
                parquet_file = os.path.join(
                    args.output_dir, f"{os.path.splitext(os.path.basename(output_file))[0]}.parquet"
                )
                # Chuyển dataset sang pandas DataFrame và lưu dưới dạng .parquet
                processed_dataset[split].to_pandas().to_parquet(parquet_file, index=False)
                print(f"Saved {split} to {parquet_file}")


            # Optionally push to hub
            if args.repo_id:
                print("Pushing to the hub...")
                # Assuming each file has a unique repo_id, you might need to modify this
                # For simplicity, appending the filename to repo_id
                repo_id = f"{args.repo_id}/{os.path.splitext(base_filename)[0]}"
                if args.configuration:
                    processed_dataset.push_to_hub(repo_id, args.configuration)
                else:
                    processed_dataset.push_to_hub(repo_id)

    else:
        # Existing single dataset processing
        if args.configuration:
            dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.cpu_num_workers)
        else:
            dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers)

        audio_column_name = "audio" if args.rename_column else args.audio_column_name
        text_column_name = "text" if args.rename_column else args.text_column_name
        if args.rename_column:
            dataset = dataset.rename_columns({args.audio_column_name: "audio", args.text_column_name: "text"})

        # Process the dataset
        processed_dataset = process_dataset(dataset, args, audio_column_name, text_column_name)

        # Save to disk
        if args.output_dir:
            print("Saving to disk...")
            processed_dataset.save_to_disk(args.output_dir)

        # Push to hub
        if args.repo_id:
            print("Pushing to the hub...")
            if args.configuration:
                processed_dataset.push_to_hub(args.repo_id, args.configuration)
            else:
                processed_dataset.push_to_hub(args.repo_id)
