# CitiBike NYC: Demand, Risk & Net Flow Analysis (2023–2025)

This repository contains a data-science study combining **CitiBike trip data** with **NYPD collision data** to analyze demand, station-level net flow, and a transparent **risk-per-trip** measure (by station, time of day, and their interaction).  
The results support **user warnings**, **insurance pricing**, and **operational safety interventions**.

👉 **Live report (HTML):** https://armoutihansen.xyz/DSC/  

---

## Repository Structure (Short)

- **index.html** — main narrative report with all figures and analysis.
- **notebooks/**
  - `clean_citibike.ipynb` — example workflow for cleaning CitiBike data.
  - `clean_collision_data.ipynb` — cleaning + cyclist-involvement parsing for NYPD crash data.
  - additional EDA, risk, and prediction notebooks (demand, net flow, EB smoothing, etc.).
- **src/** — helper scripts (risk computation, feature engineering, utilities).
- **figures/** (if present) — generated plots used in the report.

*Raw datasets are not included and must be downloaded separately from official sources.*

---

## Data Sources

- **CitiBike System Data:** trip histories, station metadata.  
  https://citibikenyc.com/system-data  
- **NYPD Collision Data:** crash locations, severity, cyclist involvement.  
  NYC Open Data portal.

---

## Getting Started

```bash
git clone https://github.com/armoutihansen/DSC.git
cd DSC
# create environment as you prefer (conda/venv)
pip install -r requirements.txt  # if available, otherwise install packages used in notebooks
jupyter lab
```
