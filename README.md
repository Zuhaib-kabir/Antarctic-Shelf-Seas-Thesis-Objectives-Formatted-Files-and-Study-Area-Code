# Antarctic Shelf Seas Thesis — Objectives Formatted Files and Study Area Code

This repository contains the **formatted Python workflows for the thesis study area and four research objectives** focused on the Antarctic shelf seas and the wider Southern Ocean.

The scripts are designed primarily for **Google Colab** and reference thesis data stored in Google Drive under:

```text
/content/drive/MyDrive/SAM_Thesis/
```

## Repository contents

| File | Purpose |
|---|---|
| [`Study-Area-Fig2.py`](./Study-Area-Fig2.py) | Integrated Antarctic shelf-seas study-area map |
| [`Objective1-formatted-files.py`](./Objective1-formatted-files.py) | Argo upper-ocean heat-content and sea-ice data preparation |
| [`Objective2-formatted-files.py`](./Objective2-formatted-files.py) | Sea-level, steric, freshwater, salinity, stratification, EOF, forcing, validation, and statistical products |
| [`Objective3-formatted-files.py`](./Objective3-formatted-files.py) | Non-GBIF carbon and biological data-processing pipeline |
| [`Objective4-formatted-files.py`](./Objective4-formatted-files.py) | GBIF Antarctic biodiversity download, cleaning, gridding, and diversity-index pipeline |

## Study domain

Most workflows use the Antarctic / Southern Ocean domain:

```text
Latitude:  90°S to 60°S
Longitude: 180°W to 180°E
```

The main analysis period is **2008–2025**. The study-area workflow also uses Argo profiles extending back to **2001**.

The scripts use 13 Antarctic shelf-sea sectors:

- Weddell Sea (WED)
- King Haakon VII Sea (KHV)
- Riiser-Larsen Sea (RLS)
- Lazarev Sea (LAZ)
- Cosmonauts Sea (COS)
- Cooperation Sea (COO)
- Davis Sea (DAV)
- Mawson Sea (MAW)
- D'Urville Sea (DUR)
- Somov Sea (SOM)
- Ross Sea (ROS)
- Amundsen Sea (AMU)
- Bellingshausen Sea (BEL)

## Study area — `Study-Area-Fig2.py`

Creates the integrated thesis study-area map using:

- GEBCO 2024 bathymetry
- OSTIA sea-ice concentration
- Argo hydrographic profile locations
- GBIF occurrence locations
- 13 Antarctic shelf-sea sectors
- thesis concept annotations and legends

Main inputs include:

```text
GEBCO_2024_CF.nc
OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc
argo_SO_profiles_2001_2025_cleaned_gridded.nc
GBIF_clean_gridded_MONTHLY5000_2008_2025_1deg.csv
```

## Objective 1 — `Objective1-formatted-files.py`

Prepares Argo upper-ocean heat-content and sea-ice analysis products.

Main processing includes:

- profile-level 0–100 m ocean heat content
- local objective analysis of monthly OHC anomalies
- sea-wise monthly sea-ice concentration
- monthly Argo OHC aggregation
- monthly Argo profile counts
- TEOS-10 calculations using `gsw`
- cosine-latitude weighting
- no temporal interpolation of missing OHC months

Main outputs:

```text
Argo_profile_OHC_0_100m.csv
OHC_0_100m_monthly_anomalies_local_objective_analysis_2008_2025.nc
SIC_seawise_monthly_2008_2025_LOW_RAM.csv
Argo_monthly_OHC_0_100m_GJ_m2.csv
Argo_monthly_OHC_profile_counts.csv
```

## Objective 2 — `Objective2-formatted-files.py`

Master Southern Ocean sea-level analysis data-product pipeline.

It creates and organizes analysis-ready `.nc`, `.csv`, and `.xlsx` products used by the main figures, supplementary figures, and tables.

Core EN4-derived products include:

- total steric height, 0–1000 m
- thermosteric height, 0–1000 m
- halosteric height, 0–1000 m
- freshwater content, 0–200 m
- surface salinity, 0–30 m
- surface-salinity anomaly, 0–30 m
- stratification, 20–200 m
- ocean heat content, 0–1000 m

The workflow also contains processing for:

- SLA–steric comparisons
- seasonal climatologies
- steric decomposition
- trend analysis
- Sen slopes and bootstrap confidence intervals
- modified Mann–Kendall / FDR analysis
- EOF analysis
- local forcing and AAO attribution
- Argo–EN4 validation
- integration-depth sensitivity
- supplementary statistical tables

## Objective 3 — `Objective3-formatted-files.py`

Contains the non-GBIF Antarctic carbon and biological data-processing pipeline.

Main processing includes:

- coordinate and time standardization
- monthly aggregation
- sea-wise area-weighted time series
- integrated net primary production

```text
NPPINT = NPPv × Zeu
```

- phytoplankton functional-type seasonal tables
- carbon and biological sea-wise metrics
- biological-carbon-pump MLD/stratification tables
- lag-correlation and bootstrap lag analyses
- non-GBIF sea-classification input

Main outputs include:

```text
NPPINT_monthly_2008_2025_SO.nc
Obj2Fig21_PFT_seasonal_mean.csv
Obj2Fig21_PFT_relative_contribution.csv
Fig22_seawise_monthly_timeseries_LOW_RAM.csv
Fig22_seawise_metrics_LOW_RAM.csv
Fig23_seawise_monthly_timeseries_LOW_RAM.csv
Fig24_BCP_seawise_monthly_LOW_RAM.csv
Fig24_BCP_seawise_summary_ACTIVE_LOW_RAM.csv
Fig25_lag_correlation_seawise.csv
Fig25_lag_correlation_mean.csv
sea_classification_input_2008_2025_NO_GBIF.csv
```

## Objective 4 — `Objective4-formatted-files.py`

Contains the GBIF Antarctic biodiversity pipeline.

The workflow:

1. downloads GBIF occurrence records
2. cleans and filters records
3. assigns records to a 1° grid
4. creates analysis-ready biodiversity metrics and files

Study settings:

```text
Latitude:               -90° to -60°
Longitude:              -180° to 180°
Period:                 2008–2025
Grid resolution:        1°
Maximum records/month:  5,000
Random seed:            42
Coordinate uncertainty: ≤100,000 m
```

Biodiversity metrics include:

- occurrence count
- species richness
- Shannon diversity
- Simpson diversity
- Pielou evenness

> The biodiversity indices are occurrence-based proxies derived from GBIF records and should not be interpreted as direct abundance estimates.

## Software environment

Common Python packages used across the repository include:

```text
numpy
pandas
xarray
dask
netCDF4
h5netcdf
scipy
gsw
matplotlib
cartopy
requests
```

The scripts include their own package-installation commands or dependency checks.

## Data paths

The code expects thesis datasets in Google Drive, especially:

```text
/content/drive/MyDrive/SAM_Thesis/Data/
/content/drive/MyDrive/SAM_Thesis/Processed/
/content/drive/MyDrive/SAM_Thesis/Fig/
/content/drive/MyDrive/SAM_Thesis/paper2/
```

Large observational datasets are referenced by path and are not embedded in the Python scripts.

## Running the scripts

1. Open the required script in Google Colab.
2. Ensure the required thesis datasets are available in Google Drive.
3. Run the script.
4. Authorize Google Drive access.
5. Allow installation of missing packages.
6. Verify the configured input paths.
7. Run the processing workflow.
8. Check the configured output directory for generated products.

Some scripts include `RUN_...`, `BUILD_...`, cache, or reuse switches so individual stages can be rerun without rebuilding everything.

## Reproducibility notes

The formatted workflows preserve the main thesis settings, including:

- 2008–2025 analysis period
- 13 Antarctic shelf-sea sectors
- longitude normalization to `[-180°, 180°)`
- cosine-latitude weighting where used
- TEOS-10 calculations where required
- fixed reference-state definitions in sea-level processing
- low-RAM and checkpointed workflows for large datasets
- explicit quality-control thresholds

## Repository structure

```text
.
├── Study-Area-Fig2.py
├── Objective1-formatted-files.py
├── Objective2-formatted-files.py
├── Objective3-formatted-files.py
├── Objective4-formatted-files.py
├── .gitignore
├── LICENSE
└── README.md
```

## License

This repository is distributed under the license included in [`LICENSE`](./LICENSE).

## Scope

This repository provides an organized record of the **thesis study-area code and the four objective-specific formatted processing workflows** for the Antarctic shelf seas and Southern Ocean.
