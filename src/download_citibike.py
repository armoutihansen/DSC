#!/usr/bin/env python3

import argparse
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import re

import requests

BASE_URL = "https://s3.amazonaws.com/tripdata/{name}"


def parse_year_month_from_name(filename: str):
    """
    Extract (year, month) from names like:
    '202301-citibike-tripdata.csv' or '202301-citibike-tripdata.zip'.
    """
    m = re.search(r"(\d{4})(\d{2})", filename)
    if not m:
        raise ValueError(f"Could not find YYYYMM in: {filename}")
    year = int(m.group(1))
    month = int(m.group(2))
    return year, month


def download_zip(name: str) -> bytes:
    """Download a zip file from the CitiBike S3 bucket and return its bytes."""
    url = BASE_URL.format(name=name)
    print(f"Downloading {name} from {url} ...")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download {name}: HTTP {resp.status_code}")
    return resp.content


def extract_2023(content: bytes, raw_dir: Path):
    """
    2023: outer zip contains inner zips (one per month).
    We unzip outer, then unzip each inner zip's CSV files into raw_dir/YYYY/MM.
    """
    print("Extracting 2023 yearly archive (with inner zips)...")
    with ZipFile(BytesIO(content)) as outer:
        for info in outer.infolist():
            inner_name = info.filename

            # Skip macOS metadata files like __MACOSX/.../._something.zip
            if "__MACOSX" in inner_name or "/._" in inner_name or inner_name.startswith("._"):
                print(f"  Skipping macOS metadata: {inner_name}")
                continue

            if not inner_name.lower().endswith(".zip"):
                continue

            print(f"  Found inner zip: {inner_name}")
            inner_bytes = outer.read(inner_name)

            with ZipFile(BytesIO(inner_bytes)) as inner_zip:
                for member in inner_zip.infolist():
                    csv_name = member.filename
                    if not csv_name.lower().endswith(".csv"):
                        continue

                    year, month = parse_year_month_from_name(csv_name)
                    month_dir = raw_dir / f"{year}" / f"{month:02d}"
                    month_dir.mkdir(parents=True, exist_ok=True)

                    out_path = month_dir / Path(csv_name).name
                    print(f"    Extracting {csv_name} -> {out_path}")
                    with inner_zip.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())


def extract_monthly(content: bytes, raw_dir: Path, year: int, month: int):
    """
    2024–2025: each monthly zip contains CSV files directly.
    We unzip all CSVs into raw_dir/YYYY/MM.
    """
    print(f"Extracting monthly archive for {year}-{month:02d}...")
    month_dir = raw_dir / f"{year}" / f"{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(BytesIO(content)) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name.lower().endswith(".csv"):
                continue

            out_path = month_dir / Path(name).name
            print(f"  Extracting {name} -> {out_path}")
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())


def main(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving data under: {raw_dir.resolve()}")

    # 1) 2023: one yearly zip with inner zips
    yearly_2023 = "2023-citibike-tripdata.zip"
    content_2023 = download_zip(yearly_2023)
    extract_2023(content_2023, raw_dir)

    # 2) 2024–2025: monthly zips with CSVs inside
    for year in (2024, 2025):
        for month in range(1, 13):
            if year == 2025 and month > 10:
                break

            zip_name = f"{year}{month:02d}-citibike-tripdata.zip"
            try:
                content = download_zip(zip_name)
            except RuntimeError as e:
                print(f"  Skipping {zip_name}: {e}")
                continue

            extract_monthly(content, raw_dir, year, month)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CitiBike tripdata"
                    "into raw_dir/YYYY/MM/*.csv."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Root directory for raw data, e.g. ./data/raw/citibike",
    )
    args = parser.parse_args()
    main(args.raw_dir)