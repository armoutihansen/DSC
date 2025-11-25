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
    Clean CitiBike NYC CSV chunk.
    """

    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)
    logger.info(f"{csv_path.name}: initial rows = {n_raw}")


    # -------------------------------------------------------------------------
    # 1. Enforce all column dtypes
    # -------------------------------------------------------------------------

    # ride_id always string
    if "ride_id" in df.columns:
        df["ride_id"] = df["ride_id"].astype("string")

    # station names always string
    for col in ["start_station_name", "end_station_name"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # convert station IDs to float64 (coerce invalid)
    for col in ["start_station_id", "end_station_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # member and rideable_type as string
    for col in ["rideable_type", "member_casual"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # coordinates
    for col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # datetime parsing
    for col in ["started_at", "ended_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")


    # -------------------------------------------------------------------------
    # 2. Remove rows with missing station IDs
    # -------------------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["start_station_id", "end_station_id"])
    logger.info(f"{csv_path.name}: dropped {before - len(df)} rows with missing station IDs")


    # -------------------------------------------------------------------------
    # 3. Canonical station names: choose MOST COMMON name per ID
    # -------------------------------------------------------------------------

    # Most common name per start_station_id
    start_map = (
        df.groupby(["start_station_id", "start_station_name"])
          .size()
          .reset_index(name="n")
          .sort_values(["start_station_id", "n"], ascending=[True, False])
          .drop_duplicates(subset=["start_station_id"])
          .set_index("start_station_id")["start_station_name"]
    )

    # Most common name per end_station_id
    end_map = (
        df.groupby(["end_station_id", "end_station_name"])
          .size()
          .reset_index(name="n")
          .sort_values(["end_station_id", "n"], ascending=[True, False])
          .drop_duplicates(subset=["end_station_id"])
          .set_index("end_station_id")["end_station_name"]
    )

    # Apply names
    df["start_station_name"] = df["start_station_id"].map(start_map)
    df["end_station_name"]   = df["end_station_id"].map(end_map)


    # # -------------------------------------------------------------------------
    # # 4. Remove rows missing coordinates
    # # -------------------------------------------------------------------------
    # before = len(df)
    # df = df.dropna(subset=["start_lat", "start_lng"])
    # logger.info(f"{csv_path.name}: dropped {before - len(df)} rows missing start coords")

    # before = len(df)
    # df = df.dropna(subset=["end_lat", "end_lng"])
    # logger.info(f"{csv_path.name}: dropped {before - len(df)} rows missing end coords")


    # -------------------------------------------------------------------------
    # 5. Final missing value check
    # -------------------------------------------------------------------------
    na = df.isna().sum()
    total_na = int(na.sum())
    if total_na > 0:
        logger.warning(f"{csv_path.name}: {total_na} missing values remain")
        logger.warning(f"{csv_path.name}: missing by column:\n{na[na > 0]}")
    else:
        logger.info(f"{csv_path.name}: no missing values after cleaning")

    n_clean = len(df)
    logger.info(f"{csv_path.name}: cleaned rows = {n_clean}")
    return df, n_raw, n_clean

###############################################################################
# Main script: Walk directory structure YYYY/MM under --raw_root
###############################################################################

def main(raw_dir: Path, out_dir: Path):
    """
    Structure:
    raw_dir/YYYY/MM/*.csv

    Writes:
    out_dir/year=YYYY/month=MM/data.parquet
    """
    logger.info(f"Raw root: {raw_dir}")
    logger.info(f"Output root: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_raw_all = 0
    total_clean_all = 0

    for year_dir in sorted(raw_dir.iterdir()):
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
                df_clean, n_raw, n_clean = clean_citibike_csv(csv_path)
                month_dfs.append(df_clean)
                total_raw_all += n_raw
                total_clean_all += n_clean

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
            partition_dir = out_dir / f"{year}" / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            out_file = partition_dir / "data.parquet"

            df_month.to_parquet(out_file, index=False)
            logger.info(f"Wrote {out_file}")
    
    total_removed = total_raw_all - total_clean_all
    pct_kept = 100 * total_clean_all / total_raw_all
    pct_removed = 100 * total_removed / total_raw_all

    logger.info("========== Global cleaning summary ==========")
    logger.info(f"Total rows before cleaning: {total_raw_all:,}")
    logger.info(f"Total rows after cleaning:  {total_clean_all:,}")
    logger.info(f"Rows removed:               {total_removed:,}")
    logger.info(f"Percentage retained:        {pct_kept:.2f}%")
    logger.info(f"Percentage removed:         {pct_removed:.2f}%")
    logger.info("=============================================")


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean CitiBike NYC tripdata"
    )
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="Root directory containing raw data organized as raw_dir/YYYY/MM/")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Where to write cleaned Parquet files")

    args = parser.parse_args()

    main(args.raw_dir, args.out_dir)