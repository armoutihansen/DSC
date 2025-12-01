# CitiBike NYC: Demand, Risk & Net Flow Analysis (2023–2025)

This repository contains a data-science study combining **CitiBike trip data** with **NYPD collision data** to analyze demand, station-level net flow, and a transparent **risk-per-trip** measure (by station, time of day, and their interaction).  
The results support **user warnings**, **insurance pricing**, and **operational safety interventions**.

👉 **Report:** https://armoutihansen.xyz/DSC/  

---

## Repository Structure

- **index.html** — report with all figures and analysis.
- **notebooks/**
  - `clean_citibike.ipynb` — example workflow for cleaning CitiBike data.
  - `clean_collision_data.ipynb` — cleaning + cyclist-involvement parsing for NYPD crash data.
  - `EDA_citibike.ipynb`- data analysis of CitiBike data.
  - `risk_analysis.ipynb`- risk analysis.
  - `net_flow_analysis.ipynb`- net flow prediction.
- **src/**
  - `download_citibike.py`- helper script to download all CitiBike data.
  - `clean_citibike_csv.py`- helper script that cleans all CitiBike CSVs and export them as parquet.
  - `figures/` - generated plots used in the report.
  - `logs/`- all logs from data cleaning
- **data/processed/**
  - `cleaned_collision_data.csv` - cleaned collision data.

*Raw datasets are not included and must be downloaded separately from official sources.*

---

## Data Sources

- [**CitiBike Trip Data**](https://s3.amazonaws.com/tripdata/index.html)
- [**NYPD Collision Data**](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)

---

## Getting Started

```bash
git clone https://github.com/armoutihansen/DSC.git
cd DSC
conda env create -f environment.yml
conda activate dsc
python ./src/download_citibike.py --raw_dir ./data/raw/citibike (Warning: takes a long time)
python ./src/clean_citibike_csv.py --raw_dir ./data/raw/citibike --out_dir ../data/processed/citibike
```
