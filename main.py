#!/usr/bin/env python3
import argparse
from typing import List
from typing_extensions import Dict
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import os
import sys

DATE_FORMAT = "%d-%b-%y"

def append_to_csv_from_list(data_list: List[Dict[str, str]], output_file: Path):
    """
    Converts a list of dictionaries into a Pandas DataFrame and appends it
    to a specified CSV file, handling the header automatically.

    Args:
        data_list: The list of dictionaries (rows) to be appended.
        output_file: The path to the target CSV file.
    """
    if not data_list:
        print("Warning: Attempted to append an empty list of data. No action taken.")
        return

    # Convert the list of dicts to a DataFrame
    df_to_append = pd.DataFrame(data_list)

    # Check if file exists to decide on writing the header
    include_header = not os.path.exists(output_file)

    try:
        # Use 'a' for append mode
        df_to_append.to_csv(
            output_file,
            mode='a',
            index=False,
            header=include_header
        )
    except Exception as e:
        print(f"\nError saving data to CSV: {e}")

def extract_path_components(path_str: str) -> Dict[str, str]:
    """
    Parses a specific file path structure to extract trading option metadata.

    Args:
        path_str: The full path string.

    Returns:
        A dictionary containing the extracted components.
    """
    # Use Path to easily handle path parts regardless of OS
    p = Path(path_str)

    # 1. File Name (the last component)
    file_name = p.name

    # 2. Extract parts from the path hierarchy
    # Date: Assumed to be two levels up from the file (e.g., '2025-01-03')
    try:
        date = p.parent.parent.name
    except IndexError:
        print(f"Error: Path structure for {path_str} is too short. Cannot reliably extract Date from folders.")
        date = "N/A"

    # 3. Extract parts from the filename (e.g., NIFTY_22900_CE_09_JAN_25.csv)
    # The stem is the filename without the extension
    filename_parts = p.stem.split('_')

    # Initialize defaults
    symbol, strike, optiontype, expiry_str = "N/A", "N/A", "N/A", "N/A"

    if len(filename_parts) >= 6:
        symbol = filename_parts[0]  # e.g., NIFTY
        strike = filename_parts[1]     # e.g., 22900
        optiontype = filename_parts[2] # e.g., CE

        # Expiry: Reconstruct the DD_MON_YY part into a readable date string
        day, month, year = filename_parts[3], filename_parts[4], filename_parts[5]
        expiry_str = f"{day}-{month}-{year}"

        # Optional: verify date format
        try:
            datetime.strptime(expiry_str, DATE_FORMAT)
        except ValueError:
            print(f"Warning: Could not parse expiry date part: {expiry_str} for file {file_name}")

    else:
        print(f"Error: Filename '{file_name}' does not contain expected 6 parts.")


    # Create the result dictionary
    result = {
        "date": date,
        "file_name": file_name,
        "symbol": symbol, # Renamed from 'underline'
        "strike": strike,
        "optiontype": optiontype,
        "expire": expiry_str
    }

    return result


def build_filename_series(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized function to build the output filename for all rows based on row data.
    Returns NaN for rows where 'Expiry' is missing or unparseable.
    """
    # Convert 'Expiry' to datetime objects, coercing errors to NaT (Not a Time)
    expiry_dt = pd.to_datetime(df["Expiry"], errors='coerce')

    # Filter out rows where conversion failed (NaT) or if other required fields are missing
    # We check for missing metadata here, but the primary fallback logic is handled in process_csv
    valid_mask = expiry_dt.notna() & df["Symbol"].notna() & df["Strike"].notna() & df["OptionType"].notna()

    # Apply date formatting only to valid dates
    day = expiry_dt[valid_mask].dt.strftime("%d")
    # %b is locale-dependent, but we force uppercase for standard abbreviations (e.g., 'JUL')
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

    # Create a result Series and initialize with NaN for invalid/missing rows
    result = pd.Series(None, index=df.index, dtype=object)
    result.loc[valid_mask] = filenames

    # Convert the Series to use NaN instead of None for easier filtering later
    return result.where(result.notna())

def parse_filename_components_from_input_path(input_file: Path) -> dict or None:
    """
    Parses Symbol, Strike, and OptionType from an input filename,
    assuming format: SYMBOL_STRIKE_OPTIONTYPE_DD_MON_YY.csv
    """
    try:
        parts = input_file.stem.split('_')
        # We need at least 6 parts for SYMBOL_STRIKE_OPTIONTYPE_DD_MON_YY
        if len(parts) >= 6:
            # We only extract the symbol, strike, and option type for path/filename
            return {
                "Symbol": parts[0],
                "Strike": parts[1],
                "OptionType": parts[2],
                "Filename": input_file.name # Use the original filename as the output filename
            }
    except Exception:
        pass
    # If parsing fails or parts are missing
    return None

def process_csv(input_file: Path, output_dir: Path, universe_csv_file_path: Path):
    start_file = time.perf_counter()
    try:
        # Load the CSV. Using 'dtype=str' ensures all fields are treated consistently.
        df = pd.read_csv(input_file, dtype=str)
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, error reading: {e}")
        return

    # Capture the original columns before adding any temporary ones
    initial_cols = list(df.columns)

    # Check for required columns existence
    required_cols = {"DateTime", "Symbol", "Strike", "OptionType", "Expiry"}
    if not required_cols.issubset(df.columns):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, missing required columns.")
        return

    initial_rows = len(df)

    # 1. Primary Filter: Drop rows where 'DateTime' is missing (no time series data).
    df = df.dropna(subset=["DateTime"])
    rows_after_dt_filter = len(df)

    if rows_after_dt_filter == 0:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, no rows with valid 'DateTime' after cleanup.")
        return

    # 2. Vectorized calculation of filename (Standard Path)
    df["filename"] = build_filename_series(df)

    # 3. Identify rows that failed standard filename generation (Need Fallback)
    fallback_mask = df["filename"].isna()
    rows_needing_fallback = fallback_mask.sum()

    if rows_needing_fallback > 0:
        fallback_data = parse_filename_components_from_input_path(input_file)

        if fallback_data:
            # Apply fallback symbol (for grouping) and filename to the masked rows
            fallback_symbol = fallback_data["Symbol"]
            fallback_filename = fallback_data["Filename"]

            df.loc[fallback_mask, "Symbol"] = fallback_symbol
            df.loc[fallback_mask, "filename"] = fallback_filename

        else:
            # If the input filename itself is not parsable, these rows are unrecoverable.
            # Drop the rows that need fallback but couldn't get it.
            df = df[~fallback_mask]
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Dropped {rows_needing_fallback:,} unrecoverable rows from {input_file.name} (bad row data and unparsable filename).")


    # 4. Final Filter: Drop any rows where filename is still NaN (i.e., failed both standard and fallback)
    df = df.dropna(subset=["filename"])

    rows_kept = len(df)
    if rows_kept == 0:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Skipping {input_file}, no valid rows after all cleanup and fallback attempts.")
        return

    # 5. Extract Date part from DateTime (Vectorized)
    df["Date"] = df["DateTime"].str.slice(0, 10)

    # 6. Construct the full output path string (Vectorized)
    # Format: output_dir/Date/Symbol/filename
    path_prefix = str(output_dir) + os.sep
    df["out_path_str"] = path_prefix + df["Date"].str.cat(
        [
            df["Symbol"],
            df["filename"]
        ],
        sep=os.sep
    )

    # 7. Group by the final output path and write
    written_rows_count = 0
    unique_files_count = 0

    universe_data: List[Dict[str, str]] = []

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
            columns=initial_cols # Write only the columns present in the original input file
        )
        written_rows_count += len(group_df)
        unique_files_count += 1

        data = extract_path_components(out_path.as_uri())
        universe_data.append(data)

    append_to_csv_from_list(universe_data, universe_csv_file_path)

    end_file = time.perf_counter()
    duration = end_file - start_file
    print(
        f"{input_file.name}: Processed {initial_rows:,} rows (Kept: {rows_kept:,}). "
        f"Generated {unique_files_count:,} new file(s). "
        f"⏱️ Time for file: {duration:.2f} seconds"
    )
    return written_rows_count, unique_files_count

def process_folder(input_dir: Path, output_dir: Path, universe_csv_file_path: Path):
    """Iterates through all CSV files recursively and processes them with a progress bar."""
    start_processing = time.perf_counter()

    csv_files = list(input_dir.rglob("*.csv"))
    total_files = len(csv_files)

    total_rows_written = 0
    total_files_generated = 0

    print(f"Found {total_files} CSV files to process.\n")

    for i, csv_file in enumerate(csv_files, 1):
        # Progress bar update before processing the file
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing file {i:,}/{total_files:,}: {csv_file.name}...\r", end="")
        sys.stdout.flush()

        # Process the file
        try:
            result = process_csv(csv_file, output_dir, universe_csv_file_path)
            if result is not None:
                rows_written, files_generated = result
                total_rows_written += rows_written
                total_files_generated += files_generated
        except TypeError:
            # Handle cases where process_csv returns None if an error occurs during initial read/check
            pass

    end_processing = time.perf_counter()
    duration = end_processing - start_processing

    # Clear the progress line after completion and print summary
    print(" " * 100 + "\r", end="")
    print("=======================================================")
    print(f"Input Processing Complete: {total_files} files processed.")
    print(f"Total Rows Written: {total_rows_written:,}")
    print(f"Total Output Files Created/Appended: {total_files_generated:,}")
    print(f"Total Time Taken: {duration / 60:,} minutes")
    print("=======================================================")

def sort_universe_file(universe_file: Path):
    """
    Sorts a universe file by the 'date' column using a Pandas DataFrame.

    Args:
        universe_file: The Path object pointing to the CSV file to be sorted.
    """
    print("\n=======================================================")
    print(f"Sorting universe file {universe_file.name} by 'date'...")

    try:
        # 1. Read the file into a DataFrame
        # We read the 'date' column as a string to ensure consistent sorting,
        # since it's already in the 'YYYY-MM-DD' format which sorts correctly lexicographically.
        df = pd.read_csv(universe_file)

        # Ensure the 'date' column exists (it should, based on extract_path_components)
        if "date" not in df.columns:
            print(f"Error: 'date' column not found in {universe_file.name}. Skipping sort.")
            return

        # 2. Sort the DataFrame by the 'date' column (ascending by default)
        df.sort_values(by="date", inplace=True)

        # 3. Overwrite the original file with the sorted content
        df.to_csv(universe_file, index=False, header=True)
        print(f"Successfully sorted {len(df)} rows in {universe_file.name}.")
        print("=======================================================")

    except Exception as e:
        print(f"Error during sorting of {universe_file.name}: {e}")

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
            # IMPORTANT: This assumes 'DateTime' column is always present due to the process_csv filter
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
    print(f"Finished sorting. {sorted_count:,} files were successfully sorted in {duration / 60:.2f} minutes.")
    print("=======================================================")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized script to reorganize and split CSV data by grouping unique option parameters, with fallback support."
    )
    parser.add_argument("input_dir", type=str, help="Path to input folder (relative or absolute)")
    parser.add_argument("output_dir", type=str, help="Path to output folder (relative or absolute)")

    args = parser.parse_args()

    # Use resolve() for clean, absolute paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir, "data").resolve()
    universe_csv_file_path = Path(args.output_dir, "universe.csv").resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' does not exist or is not a directory.")
        return

    process_folder(input_dir, output_dir, universe_csv_file_path)
    # Sorting of each output file after all processing is complete
    sort_output_files(output_dir)
    # Create universe file
    sort_universe_file(universe_csv_file_path)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()

    elapsed = end - start
    print(f"\n=======================================================")
    print(f"🏁 Total execution time: {elapsed:.2f} seconds")
    print(f"=======================================================")
