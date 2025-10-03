#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import os
import sys

def build_filename_series(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized function to build the output filename for all rows.
    Leverages pandas' built-in datetime functionality for performance.
    """
    # Convert 'Expiry' to datetime objects, coercing errors to NaT
    expiry_dt = pd.to_datetime(df["Expiry"], errors='coerce')

    # Filter out rows where conversion failed (NaT)
    valid_mask = expiry_dt.notna()

    # Apply date formatting only to valid dates
    day = expiry_dt[valid_mask].dt.strftime("%d")
    # %b is locale-dependent, but we force uppercase, assuming 'en' locale for standard abbreviations (e.g., 'JUL')
    mon = expiry_dt[valid_mask].dt.strftime("%b").str.upper()
    year = expiry_dt[valid_mask].dt.strftime("%y")

    # Construct the filename using f-strings for the valid rows
    filenames = df.loc[valid_mask, 'Symbol'].str.cat(
        [
            df.loc[valid_mask, 'Strike'],
            df.loc[valid_mask, 'OptionType'],
            day,
            mon,
            year
        ],
        sep="_"
    )
    filenames = filenames.str.cat([pd.Series(".csv", index=filenames.index)], sep="")

    # Create a result Series and initialize with None for invalid rows
    result = pd.Series(None, index=df.index, dtype=object)
    result.loc[valid_mask] = filenames

    return result

def process_csv(input_file: Path, output_dir: Path):
    start_file = time.perf_counter()
    try:
        # Load the CSV. Using 'dtype=str' ensures all fields are treated consistently before processing.
        df = pd.read_csv(input_file, dtype=str)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, error reading: {e}")
        return

    required_cols = {"DateTime", "Symbol", "Strike", "OptionType", "Expiry"}
    if not required_cols.issubset(df.columns):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, missing required columns.")
        return

    # Drop invalid rows based on required columns
    initial_rows = len(df)
    df = df.dropna(subset=list(required_cols))
    if len(df) == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, no valid rows after cleanup.")
        return

    # 1. Vectorized calculation of filename
    df["filename"] = build_filename_series(df)
    df = df.dropna(subset=["filename"]) # Drop rows where filename could not be generated (e.g., bad 'Expiry')

    # 2. Extract Date part from DateTime (Vectorized)
    df["Date"] = df["DateTime"].str.slice(0, 10)

    # 3. Construct the full output path string (Vectorized)
    # This path string will be used for efficient grouping and I/O.
    # Format: output_dir/Date/Symbol/filename
    # FIX: Replaced Path.sep with os.sep
    path_prefix = str(output_dir) + os.sep
    df["out_path_str"] = path_prefix + df["Date"].str.cat(
        [
            df["Symbol"],
            df["filename"]
        ],
        sep=os.sep # FIX: Replaced Path.sep with os.sep
    )

    # 4. Group by the final output path
    # This drastically reduces the number of disk I/O operations from rows (31M) to files (much less).
    written_rows_count = 0
    unique_files_count = 0

    for out_path_str, group_df in df.groupby("out_path_str"):
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine mode and header based on file existence
        is_append = out_path.exists()
        mode = "a" if is_append else "w"
        header = not is_append

        # Write the entire group in a single operation
        group_df.to_csv(
            out_path,
            mode=mode,
            index=False,
            header=header,
            columns=df.columns[:-3] # Exclude temporary columns: filename, Date, out_path_str
        )
        written_rows_count += len(group_df)
        unique_files_count += 1
        # Logging now reports batch write
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Written {len(group_df):,} rows to -> {out_path}")

    end_file = time.perf_counter()
    duration = end_file - start_file
    print(
        f"\n{input_file.name}: Processed {initial_rows:,} rows (Kept: {written_rows_count:,}). "
        f"Generated {unique_files_count:,} new file(s). "
        f"⏱️ Time for file: {duration:.2f} seconds\n"
    )

def process_folder(input_dir: Path, output_dir: Path):
    """Iterates through all CSV files recursively and processes them."""
    csv_files = list(input_dir.rglob("*.csv"))
    total_files = len(csv_files)
    print(f"Found {total_files} CSV files to process.\n")

    for i, csv_file in enumerate(csv_files, 1):
        print(f"Processing file {i}/{total_files}: {csv_file}")
        process_csv(csv_file, output_dir)


def sort_output_files(output_dir: Path):
    """
    Iterates through all generated CSV files and sorts their contents by DateTime.
    Includes a console progress indicator.
    """
    print("\n=======================================================")
    print("Starting post-processing sort of all output files...")
    start_sort = time.perf_counter()

    # Get all files and total count
    output_files = list(output_dir.rglob("*.csv"))
    total_files_to_sort = len(output_files)
    sorted_count = 0

    for i, output_file in enumerate(output_files, 1):
        try:
            # Print progress indicator before processing
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sorting file {i:,}/{total_files_to_sort:,}... {output_file.name}\r", end="")
            sys.stdout.flush() # Ensure the progress is immediately visible

            # Read the entire file ensuring all columns are read as strings
            df = pd.read_csv(output_file, dtype=str)

            # Sort by the 'DateTime' column.
            df.sort_values(by="DateTime", inplace=True)

            # Overwrite the file with sorted data
            df.to_csv(output_file, index=False, header=True)
            sorted_count += 1

        except Exception as e:
            # Print error on a new line so it doesn't mess up the progress bar
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Warning: Could not sort {output_file}. Error: {e}")

    # Clear the progress line after completion
    print(" " * 100 + "\r", end="")

    end_sort = time.perf_counter()
    duration = end_sort - start_sort
    print(f"Finished sorting. {sorted_count:,} files were successfully sorted in {duration:.2f} seconds.")
    print("=======================================================")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized script to reorganize and split CSV data by grouping unique option parameters."
    )
    parser.add_argument("input_dir", type=str, help="Path to input folder (relative or absolute)")
    parser.add_argument("output_dir", type=str, help="Path to output folder (relative or absolute)")

    args = parser.parse_args()

    # Use resolve() for clean, absolute paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' does not exist or is not a directory.")
        return

    # process_folder(input_dir, output_dir)
    # ADDED: Sorting of each output file after all processing is complete
    sort_output_files(output_dir)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()

    elapsed = end - start
    print(f"\n=======================================================")
    print(f"🏁 All files processed! ⏱️ Total execution time: {elapsed:.2f} seconds")
    print(f"=======================================================")
