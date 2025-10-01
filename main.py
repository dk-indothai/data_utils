#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

def build_filename(row):
    """Build output filename based on Symbol, Strike, OptionType, Expiry"""
    try:
        expiry = datetime.strptime(row["Expiry"], "%Y-%m-%d")
        day = expiry.strftime("%d")
        mon = expiry.strftime("%b").upper()   # JUL
        year = expiry.strftime("%y")          # 25
        return f"{row['Symbol']}_{row['Strike']}_{row['OptionType']}_{day}_{mon}_{year}.csv"
    except Exception:
        return None

def process_csv(input_file: Path, output_dir: Path):
    try:
        df = pd.read_csv(input_file, dtype=str)
    except Exception as e:
        print(f"Skipping {input_file}, error reading: {e}")
        return

    required_cols = {"DateTime", "Symbol", "Strike", "OptionType", "Expiry"}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {input_file}, missing required columns.")
        return

    # Drop invalid rows
    df = df.dropna(subset=["DateTime", "Symbol", "Strike", "OptionType", "Expiry"])

    # Extract Date part from DateTime
    df["Date"] = df["DateTime"].str.slice(0, 10)

    # Process per row (so each unique combination can generate correct filename)
    for _, row in df.iterrows():
        filename = build_filename(row)
        if not filename:
            continue

        date = row["Date"]
        symbol = row["Symbol"]

        # Output path: output/date/symbol/<generated filename>
        out_path = output_dir / date / symbol / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save row (append if file exists)
        if out_path.exists():
            row.to_frame().T.to_csv(out_path, mode="a", index=False, header=False)
        else:
            row.to_frame().T.to_csv(out_path, mode="w", index=False, header=True)

        print(f"Written row -> {out_path}")

def process_folder(input_dir: Path, output_dir: Path):
    for csv_file in input_dir.rglob("*.csv"):
        process_csv(csv_file, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively copy all CSV files from input folder to output folder."
    )
    parser.add_argument("input_dir", type=str, help="Path to input folder (relative or absolute)")
    parser.add_argument("output_dir", type=str, help="Path to output folder (relative or absolute)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' does not exist or is not a directory.")
        return

    process_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()
