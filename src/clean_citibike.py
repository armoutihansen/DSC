#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys


###############################################################################
# Logging setup
###############################################################################

def setup_logger(log_file="./logs/citibike_cleaning.log"):
    logger = logging.getLogger("citibike_cleaning")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()

###############################################################################
# Cleaning function for a single CSV file
###############################################################################

def clean_citibike_csv(csv_path: Path) -> pd.DataFrame:
    """
    Clean ONE CitiBike NYC CSV chunk (one file in a month folder).

    Steps:
    - enforce dtypes
    - parse datetime
    - drop missing start/end coords
    - drop station_id columns
    - fill missing station names with "Unknown"
    - log everything
    """
    logger.info(f"Loading {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    logger.info(f"{csv_path.name}: initial rows = {len(df)}")

    # Enforce string dtype for identifiers/names
    id_cols = [
        "ride_id", "start_station_name", "end_station_name",
        "start_station_id", "end_station_id"
    ]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Ride type, member type
    for col in ["rideable_type", "member_casual"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Coordinates
    for col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Datetimes
    for col in ["started_at", "ended_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove missing start coords
    before = len(df)
    df = df.dropna(subset=["start_lat", "start_lng"])
    dropped = before - len(df)
    logger.info(f"{csv_path.name}: dropped {dropped} rows missing start coords")

    # Remove missing end coords
    before = len(df)
    df = df.dropna(subset=["end_lat", "end_lng"])
    dropped = before - len(df)
    logger.info(f"{csv_path.name}: dropped {dropped} rows missing end coords")

    # Drop station IDs
    drop_cols = [c for c in ["start_station_id", "end_station_id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"{csv_path.name}: dropped columns {drop_cols}")

    # Fill missing station names
    for col in ["start_station_name", "end_station_name"]:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            df[col] = df[col].fillna("Unknown")
            missing_after = df[col].isna().sum()
            filled = missing_before - missing_after
            logger.info(f"{csv_path.name}: filled {filled} missing values in {col}")

    # Convert categories to string before Parquet
    for col in ["rideable_type", "member_casual"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Log remaining missing values
    na = df.isna().sum()
    total_na = int(na.sum())
    if total_na > 0:
        logger.warning(f"{csv_path.name}: {total_na} missing values remain")
        missing_cols = na[na > 0]
        logger.warning(f"{csv_path.name}: missing by column:\n{missing_cols}")
    else:
        logger.info(f"{csv_path.name}: no missing values after cleaning")

    logger.info(f"{csv_path.name}: cleaned rows = {len(df)}")
    return df


###############################################################################
# Main script: Walk directory structure YYYY/MM under --raw_root
###############################################################################

def main(raw_root: Path, out_root: Path):
    """
    Expects directory structure:
    raw_root/YYYY/MM/*.csv

    Writes:
    out_root/year=YYYY/month=MM/data.parquet
    """
    logger.info(f"Raw root: {raw_root}")
    logger.info(f"Output root: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    for year_dir in sorted(raw_root.iterdir()):
        if not year_dir.is_dir():
            continue

        # year directory must be 2023/2024/2025
        if not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year < 2023 or year > 2025:
            continue

        logger.info(f"Processing year {year}")

        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            if not month_dir.name.isdigit():
                continue
            month = int(month_dir.name)
            if month < 1 or month > 12:
                continue

            logger.info(f"Processing month {year}-{month:02d}")

            csv_files = sorted(month_dir.glob("*.csv"))

            if not csv_files:
                logger.warning(f"No CSV files in {month_dir}")
                continue

            # Clean and combine all CSVs in this month
            month_dfs = []
            for csv_path in csv_files:
                df_clean = clean_citibike_csv(csv_path)
                month_dfs.append(df_clean)

            df_month = pd.concat(month_dfs, ignore_index=True)

            # Final NA check for combined month
            na = df_month.isna().sum()
            total_na = int(na.sum())
            if total_na > 0:
                logger.warning(f"{year}-{month:02d}: {total_na} missing values remain after combining chunks")
                logger.warning(f"Missing by column:\n{na[na > 0]}")
            else:
                logger.info(f"{year}-{month:02d}: no missing values after combining chunks")

            # Write Parquet partition
            partition_dir = out_root / f"{year}" / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            out_file = partition_dir / "data.parquet"

            df_month.to_parquet(out_file, index=False)
            logger.info(f"Wrote {out_file}")


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean CitiBike NYC tripdata (2023–2025), using directory structure raw_root/YYYY/MM/*.csv"
    )
    parser.add_argument("--raw_root", type=Path, required=True,
                        help="Root directory containing raw data organized as raw_root/YYYY/MM/")
    parser.add_argument("--out_root", type=Path, required=True,
                        help="Where to write cleaned Parquet files")

    args = parser.parse_args()

    main(args.raw_root, args.out_root)
