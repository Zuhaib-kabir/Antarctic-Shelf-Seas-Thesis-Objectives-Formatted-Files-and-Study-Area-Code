"""
GBIF ANTARCTIC BIODIVERSITY DATA PIPELINE

Purpose
This script prepares GBIF occurrence data for analysis of Antarctic
biodiversity from 2008 to 2025.

The workflow is organized into three main stages:

1. DOWNLOAD GBIF
   - Install/import required Python libraries.
   - Define study-area, time-period, and GBIF API settings.
   - Download up to 5,000 GBIF occurrence records for each year-month.
   - Save one raw CSV file per month so the download can be resumed safely.

2. PROCESS GBIF
   - Combine the monthly GBIF CSV files.
   - Clean coordinates, year/month values, species names, duplicates, and
     coordinate uncertainty.
   - Restrict records to the Antarctic study area and 2008-2025 period.
   - Assign each occurrence to a 1-degree latitude/longitude grid cell.
   - Save combined, cleaned, summary, and gridded CSV files.

3. FORMATTED FILES FOR ANALYSIS
   - Calculate occurrence count, species richness, Shannon diversity,
     Simpson diversity, and Pielou evenness.
   - Produce total, yearly, and monthly biodiversity tables.
   - Convert the biodiversity tables to analysis-ready NetCDF files.
   - Create minimum-record masks for maps/statistical analysis.
   - Create a yearly biodiversity time-series CSV.

Study design retained from the original thesis workflow
Latitude range:  -90 to -60 degrees
Longitude range: -180 to 180 degrees
Years:           2008 to 2025
Grid resolution: 1 degree
GBIF cap:        maximum 5,000 occurrence records per year-month
Random seed:     42
Coordinate uncertainty filter: <= 100,000 m, while retaining records where
                               uncertainty is not reported

Important interpretation note:
The biodiversity indices produced here are occurrence-based proxies derived
from GBIF records. They should not be interpreted as direct biological
abundance estimates.

Environment
Designed for Google Colab with Google Drive mounted at /content/drive.

The plotting/map code from the original notebook is intentionally not included
here because this file is dedicated to data download, processing, and creation
of analysis-ready outputs.
"""


# MOUNT GOOGLE DRIVE
from google.colab import drive

drive.mount("/content/drive")


# DOWNLOAD GBIF
# This section contains all package installation/imports, global configuration,
# GBIF API settings, and monthly download functions.
# Install required libraries if they are missing
import importlib.util
import subprocess
import sys


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
}


def install_missing_packages():
    """Install only packages that are not already available."""
    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )
    else:
        print("Required Python packages are already installed.")


install_missing_packages()


# Imports
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Main workflow switches
# Change any value to False when you want to skip that stage.
# Each stage reads its required input from disk, so stages can be rerun
# independently after the necessary earlier files already exist.

RUN_DOWNLOAD = True
RUN_PROCESS = True
RUN_FORMAT_FOR_ANALYSIS = True


# Main paths
OUT_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Data/Biodiversity_Indices")
RAW_MONTHLY_DIR = OUT_DIR / "Raw_GBIF_Monthly_5000"
PROCESSED_DIR = OUT_DIR / "Processed"
FIG_DIR = OUT_DIR / "Figures"

for directory in [OUT_DIR, RAW_MONTHLY_DIR, PROCESSED_DIR, FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Study settings
LAT_MIN, LAT_MAX = -90.0, -60.0
LON_MIN, LON_MAX = -180.0, 180.0

YEAR_MIN, YEAR_MAX = 2008, 2025
GRID_RES = 1.0

MAX_RECORDS_PER_MONTH = 5000
RANDOM_SEED = 42

MAX_UNCERTAINTY_M = 100000

# Minimum number of records used for different outputs.
MIN_RECORDS_MAP = 5
MIN_RECORDS_STATS = 10

# The original workflow used >=5 records for the yearly summary time series.
# It is kept explicit here so this methodological choice is not hidden.
MIN_RECORDS_TIMESERIES = 5


# GBIF API settings
BASE_URL = "https://api.gbif.org/v1/occurrence/search"
REQUEST_LIMIT = 300
REQUEST_TIMEOUT = 60
REQUEST_PAUSE_SECONDS = 0.05

GBIF_COLUMNS = [
    "gbifID",
    "species",
    "scientificName",
    "acceptedScientificName",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "lat",
    "lon",
    "year",
    "month",
    "eventDate",
    "basisOfRecord",
    "coordinateUncertaintyInMeters",
    "issues",
]


# Shared output filenames
COMBINED_RAW_FILE = (
    PROCESSED_DIR
    / "GBIF_raw_occurrence_Antarctic_2008_2025_MONTHLY_5000max.csv"
)

CLEAN_FILE = (
    PROCESSED_DIR
    / "GBIF_clean_occurrence_MONTHLY5000_2008_2025.csv"
)

MONTHLY_DOWNLOAD_SUMMARY_FILE = (
    PROCESSED_DIR
    / "GBIF_monthly_5000_download_summary_2008_2025.csv"
)

GRIDDED_FILE = (
    PROCESSED_DIR
    / "GBIF_clean_gridded_MONTHLY5000_2008_2025_1deg.csv"
)


def print_configuration():
    """Print the core settings used by the pipeline."""
    print("\n" + "=" * 72)
    print("GBIF ANTARCTIC BIODIVERSITY PIPELINE")
    print("=" * 72)
    print(f"Output folder:             {OUT_DIR}")
    print(f"Raw monthly folder:        {RAW_MONTHLY_DIR}")
    print(f"Processed folder:          {PROCESSED_DIR}")
    print(f"Study latitude:            {LAT_MIN} to {LAT_MAX}")
    print(f"Study longitude:           {LON_MIN} to {LON_MAX}")
    print(f"Years:                     {YEAR_MIN}-{YEAR_MAX}")
    print(f"Grid resolution:           {GRID_RES} degree")
    print(f"Maximum records/month:     {MAX_RECORDS_PER_MONTH}")
    print(f"Max coordinate uncertainty:{MAX_UNCERTAINTY_M} m")
    print("=" * 72)


def build_gbif_session():
    """
    Create a requests session with automatic retry for temporary server or
    rate-limit errors.
    """
    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


GBIF_SESSION = build_gbif_session()


def monthly_gbif_file(year, month):
    """Return the final CSV path for one year-month."""
    return RAW_MONTHLY_DIR / f"GBIF_Antarctic_{year}_{month:02d}.csv"


def monthly_temp_file(year, month):
    """Return the temporary checkpoint CSV path for one year-month."""
    return RAW_MONTHLY_DIR / f"GBIF_Antarctic_{year}_{month:02d}_TEMP.csv"


def extract_gbif_record(item):
    """Extract only the fields required by the thesis workflow."""
    issues = item.get("issues")

    return {
        "gbifID": item.get("gbifID"),
        "species": item.get("species"),
        "scientificName": item.get("scientificName"),
        "acceptedScientificName": item.get("acceptedScientificName"),
        "kingdom": item.get("kingdom"),
        "phylum": item.get("phylum"),
        "class": item.get("class"),
        "order": item.get("order"),
        "family": item.get("family"),
        "genus": item.get("genus"),
        "lat": item.get("decimalLatitude"),
        "lon": item.get("decimalLongitude"),
        "year": item.get("year"),
        "month": item.get("month"),
        "eventDate": item.get("eventDate"),
        "basisOfRecord": item.get("basisOfRecord"),
        "coordinateUncertaintyInMeters": item.get(
            "coordinateUncertaintyInMeters"
        ),
        "issues": ",".join(issues) if issues else None,
    }


def download_gbif_year_month(year, month):
    """
    Download up to MAX_RECORDS_PER_MONTH GBIF occurrence records for one
    year-month.

    Monthly pagination keeps the requested API offset far below GBIF's large
    offset limit. Existing completed monthly files are skipped, making the
    workflow resumable.
    """
    final_file = monthly_gbif_file(year, month)
    temp_file = monthly_temp_file(year, month)

    # A final file is only written after the monthly request loop finishes.
    # Header-only files are valid when GBIF returns no records.
    if final_file.exists() and final_file.stat().st_size > 0:
        print(f"Already downloaded: {year}-{month:02d}")
        return final_file

    print("\n" + "-" * 72)
    print(f"Downloading GBIF records for {year}-{month:02d}")
    print("-" * 72)

    records = []
    offset = 0
    next_checkpoint = 1500

    while len(records) < MAX_RECORDS_PER_MONTH:
        remaining = MAX_RECORDS_PER_MONTH - len(records)
        current_limit = min(REQUEST_LIMIT, remaining)

        params = {
            "decimalLatitude": f"{LAT_MIN},{LAT_MAX}",
            "decimalLongitude": f"{LON_MIN},{LON_MAX}",
            "year": str(year),
            "month": str(month),
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "limit": current_limit,
            "offset": offset,
        }

        try:
            response = GBIF_SESSION.get(
                BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"Request error for {year}-{month:02d}: {exc}")
            break

        if response.status_code != 200:
            print(
                f"GBIF request failed for {year}-{month:02d}: "
                f"HTTP {response.status_code}"
            )
            print(response.text[:300])
            break

        try:
            payload = response.json()
        except ValueError as exc:
            print(f"Invalid JSON returned by GBIF: {exc}")
            break

        results = payload.get("results", [])

        if not results:
            print(f"No more GBIF records for {year}-{month:02d}")
            break

        records.extend(extract_gbif_record(item) for item in results)

        # Advance by the number actually returned rather than by the requested
        # limit. This is safer if the final page is shorter than expected.
        offset += len(results)

        if len(records) >= next_checkpoint:
            pd.DataFrame(records, columns=GBIF_COLUMNS).to_csv(
                temp_file,
                index=False,
            )
            print(
                f"{year}-{month:02d}: "
                f"{len(records):,} records downloaded"
            )
            next_checkpoint += 1500

        if payload.get("endOfRecords", False):
            break

        time.sleep(REQUEST_PAUSE_SECONDS)

    # Hard safety cap. It should normally be unnecessary because request sizes
    # are already limited above.
    if len(records) > MAX_RECORDS_PER_MONTH:
        df_month = (
            pd.DataFrame(records, columns=GBIF_COLUMNS)
            .sample(
                n=MAX_RECORDS_PER_MONTH,
                random_state=RANDOM_SEED,
            )
            .reset_index(drop=True)
        )
    else:
        df_month = pd.DataFrame(records, columns=GBIF_COLUMNS)

    # Always write column headers, even for a month with zero records.
    df_month.to_csv(final_file, index=False)

    if temp_file.exists():
        temp_file.unlink()

    print(f"Saved: {final_file}")
    print(f"Records for {year}-{month:02d}: {len(df_month):,}")

    return final_file


def download_all_gbif_months():
    """Download all requested year-month combinations."""
    total_months = (YEAR_MAX - YEAR_MIN + 1) * 12
    completed = 0

    for year in range(YEAR_MIN, YEAR_MAX + 1):
        for month in range(1, 13):
            download_gbif_year_month(year, month)
            completed += 1
            print(f"Monthly files completed: {completed}/{total_months}")

    print("\nGBIF monthly download stage finished.")
    print(f"Expected monthly files: {total_months}")


# PROCESS GBIF
# This section combines the monthly downloads, cleans the records, applies the
# spatial/temporal filters, performs quality control, and assigns 1-degree grid

def find_monthly_files():
    """Find completed monthly GBIF files, excluding temporary checkpoints."""
    pattern = str(RAW_MONTHLY_DIR / "GBIF_Antarctic_????_??.csv")
    files = sorted(Path(f) for f in glob.glob(pattern))

    return [
        f
        for f in files
        if "_TEMP" not in f.name
    ]


def combine_monthly_gbif_files():
    """
    Combine all valid monthly CSV files into one raw occurrence table.

    Returns
    -------
    pandas.DataFrame
        Combined raw GBIF records.
    """
    files = find_monthly_files()

    print("\n" + "=" * 72)
    print("COMBINING MONTHLY GBIF FILES")
    print("=" * 72)
    print(f"Monthly files found: {len(files)}")

    if not files:
        raise FileNotFoundError(
            f"No monthly GBIF files found in: {RAW_MONTHLY_DIR}"
        )

    dataframes = []
    empty_files = []
    bad_files = []

    for file_path in files:
        try:
            if file_path.stat().st_size == 0:
                empty_files.append(file_path.name)
                continue

            temp = pd.read_csv(file_path, low_memory=False)

            if temp.empty:
                empty_files.append(file_path.name)
                continue

            dataframes.append(temp)
            print(f"Read: {file_path.name} | records: {len(temp):,}")

        except pd.errors.EmptyDataError:
            empty_files.append(file_path.name)

        except Exception as exc:
            bad_files.append((file_path.name, str(exc)))
            print(f"Could not read {file_path.name}: {exc}")

    if not dataframes:
        raise ValueError(
            "No valid monthly GBIF records were found. "
            "Check the monthly download folder."
        )

    df_raw = pd.concat(dataframes, ignore_index=True)
    df_raw.to_csv(COMBINED_RAW_FILE, index=False)

    print("\nCOMBINE CHECK")
    print(f"Combined raw records:       {len(df_raw):,}")
    print(f"Empty monthly files skipped:{len(empty_files):,}")
    print(f"Unreadable files:           {len(bad_files):,}")
    print(f"Saved: {COMBINED_RAW_FILE}")

    if "year" in df_raw.columns:
        year_numeric = pd.to_numeric(df_raw["year"], errors="coerce")
        if year_numeric.notna().any():
            print(
                "Raw year range: "
                f"{int(year_numeric.min())}-{int(year_numeric.max())}"
            )

    if empty_files:
        print("Example empty monthly files:", empty_files[:20])

    if bad_files:
        print("Unreadable monthly files:")
        for filename, message in bad_files[:20]:
            print(f"  - {filename}: {message}")

    return df_raw


def standardize_required_columns(df):
    """Standardize possible coordinate column names and validate requirements."""
    rename_dict = {}

    if "decimalLatitude" in df.columns and "lat" not in df.columns:
        rename_dict["decimalLatitude"] = "lat"

    if "decimalLongitude" in df.columns and "lon" not in df.columns:
        rename_dict["decimalLongitude"] = "lon"

    df = df.rename(columns=rename_dict)

    required_cols = ["lat", "lon", "year", "month", "species"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "Missing required GBIF columns: "
            + ", ".join(missing_cols)
        )

    return df


def remove_duplicate_gbif_ids(df):
    """
    Remove duplicate non-missing GBIF IDs while retaining records whose GBIF ID
    is missing.

    This avoids the common pandas behavior where multiple missing IDs can be
    treated as duplicates of each other.
    """
    if "gbifID" not in df.columns:
        return df

    has_id = df["gbifID"].notna()

    with_id = df.loc[has_id].drop_duplicates(
        subset=["gbifID"],
        keep="first",
    )
    without_id = df.loc[~has_id]

    return (
        pd.concat([with_id, without_id], axis=0)
        .sort_index()
        .copy()
    )


def clean_gbif_records(df_raw):
    """Apply the thesis data-cleaning and quality-control rules."""
    df = standardize_required_columns(df_raw.copy())

    print("\n" + "=" * 72)
    print("CLEANING GBIF RECORDS")
    print("=" * 72)

    # Convert required numeric fields first.
    for column in ["lat", "lon", "year", "month"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove records that cannot be located or assigned to a valid time.
    df = df.dropna(
        subset=["lat", "lon", "year", "month", "species"]
    ).copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # Keep only calendar months.
    df = df[df["month"].between(1, 12)].copy()

    # Standardize longitude to [-180, 180).
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0

    # Restrict the data to the thesis study area and analysis period.
    df = df[
        df["lat"].between(LAT_MIN, LAT_MAX)
        & df["lon"].between(LON_MIN, LON_MAX)
        & df["year"].between(YEAR_MIN, YEAR_MAX)
    ].copy()

    # Remove duplicate occurrence IDs, but preserve rows without an ID.
    df = remove_duplicate_gbif_ids(df)

    # Clean species names using pandas' nullable string type.
    df["species"] = df["species"].astype("string").str.strip()
    df = df[
        df["species"].notna()
        & df["species"].ne("")
    ].copy()

    # Coordinate uncertainty quality filter.
    # Missing uncertainty values are retained, matching the original workflow.
    if "coordinateUncertaintyInMeters" in df.columns:
        df["coordinateUncertaintyInMeters"] = pd.to_numeric(
            df["coordinateUncertaintyInMeters"],
            errors="coerce",
        )

        df = df[
            df["coordinateUncertaintyInMeters"].isna()
            | (
                df["coordinateUncertaintyInMeters"]
                <= MAX_UNCERTAINTY_M
            )
        ].copy()

    return df.reset_index(drop=True)


def enforce_monthly_record_cap(df):
    """
    Safety check: ensure no cleaned year-month contains more than the configured
    5,000-record cap.

    Normally this does nothing because the download stage already enforces the
    cap. The function exists as an explicit validation safeguard.
    """
    group_sizes = df.groupby(["year", "month"]).size()

    if group_sizes.empty or group_sizes.max() <= MAX_RECORDS_PER_MONTH:
        return df

    print(
        "Warning: at least one cleaned year-month exceeds "
        f"{MAX_RECORDS_PER_MONTH:,} records."
    )
    print("Applying safety resampling with the configured random seed.")

    sampled_groups = []

    for _, group in df.groupby(
        ["year", "month"],
        sort=False,
        group_keys=False,
    ):
        if len(group) > MAX_RECORDS_PER_MONTH:
            group = group.sample(
                n=MAX_RECORDS_PER_MONTH,
                random_state=RANDOM_SEED,
            )

        sampled_groups.append(group)

    return (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )


def create_monthly_download_summary(df):
    """Create and save a year-month record-count quality-control table."""
    monthly_summary = (
        df.groupby(["year", "month"])
        .size()
        .reset_index(name="clean_records")
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    monthly_summary["exceeds_5000"] = (
        monthly_summary["clean_records"] > MAX_RECORDS_PER_MONTH
    )

    monthly_summary.to_csv(
        MONTHLY_DOWNLOAD_SUMMARY_FILE,
        index=False,
    )

    print(f"Saved: {MONTHLY_DOWNLOAD_SUMMARY_FILE}")

    if not monthly_summary.empty:
        print(
            "Maximum cleaned records in any year-month: "
            f"{monthly_summary['clean_records'].max():,}"
        )
        print(
            "Any month exceeds 5,000? "
            f"{monthly_summary['exceeds_5000'].any()}"
        )

    return monthly_summary


def build_grid_coordinates():
    """Create the 1-degree grid edges and cell centers."""
    lat_edges = np.arange(
        LAT_MIN,
        LAT_MAX + GRID_RES,
        GRID_RES,
        dtype=float,
    )
    lon_edges = np.arange(
        LON_MIN,
        LON_MAX + GRID_RES,
        GRID_RES,
        dtype=float,
    )

    lat_centers = lat_edges[:-1] + GRID_RES / 2.0
    lon_centers = lon_edges[:-1] + GRID_RES / 2.0

    return lat_edges, lon_edges, lat_centers, lon_centers


def assign_grid_cells(df):
    """Assign each cleaned occurrence to a 1-degree grid-cell center."""
    (
        lat_edges,
        lon_edges,
        lat_centers,
        lon_centers,
    ) = build_grid_coordinates()

    df = df.copy()

    df["lat_bin"] = pd.cut(
        df["lat"],
        bins=lat_edges,
        labels=lat_centers,
        include_lowest=True,
    )

    df["lon_bin"] = pd.cut(
        df["lon"],
        bins=lon_edges,
        labels=lon_centers,
        include_lowest=True,
    )

    df = df.dropna(subset=["lat_bin", "lon_bin"]).copy()

    df["lat_bin"] = df["lat_bin"].astype(float)
    df["lon_bin"] = df["lon_bin"].astype(float)

    return df.reset_index(drop=True)


def print_clean_data_check(df):
    """Print key quality-control information for the cleaned occurrence data."""
    print("\n" + "=" * 72)
    print("CLEAN DATA CHECK")
    print("=" * 72)
    print(f"Clean records:       {len(df):,}")
    print(f"Unique species:      {df['species'].nunique():,}")
    print(f"Year range:          {df['year'].min()}-{df['year'].max()}")
    print(
        "Latitude range:      "
        f"{df['lat'].min():.4f} to {df['lat'].max():.4f}"
    )
    print(
        "Longitude range:     "
        f"{df['lon'].min():.4f} to {df['lon'].max():.4f}"
    )
    print(
        "Maximum year-month:  "
        f"{df.groupby(['year', 'month']).size().max():,} records"
    )

    print("\nRecords by year:")
    print(
        df.groupby("year")
        .size()
        .rename("records")
        .to_string()
    )

    print("\nTop 20 species:")
    print(df["species"].value_counts().head(20).to_string())


def process_gbif_data():
    """
    Run the complete processing stage and save combined, cleaned, summary, and
    gridded occurrence files.
    """
    df_raw = combine_monthly_gbif_files()

    df_clean = clean_gbif_records(df_raw)
    df_clean = enforce_monthly_record_cap(df_clean)

    df_clean.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved cleaned occurrence data: {CLEAN_FILE}")

    create_monthly_download_summary(df_clean)

    df_gridded = assign_grid_cells(df_clean)
    df_gridded.to_csv(GRIDDED_FILE, index=False)

    print(f"Gridded records: {len(df_gridded):,}")
    print(f"Saved gridded occurrence data: {GRIDDED_FILE}")

    print_clean_data_check(df_gridded)

    return df_gridded


# FORMATTED FILES FOR ANALYSIS
# This section calculates biodiversity indices and writes thesis-ready CSV and
# NetCDF products for total, yearly, and monthly analysis.


INDEX_VARIABLES = [
    "occurrence_count",
    "species_richness",
    "shannon_index",
    "simpson_index",
    "pielou_evenness",
]


BIODIVERSITY_TOTAL_CSV = (
    PROCESSED_DIR
    / "Biodiversity_indices_TOTAL_MONTHLY5000_2008_2025_1deg.csv"
)

BIODIVERSITY_YEARLY_CSV = (
    PROCESSED_DIR
    / "Biodiversity_indices_YEARLY_MONTHLY5000_2008_2025_1deg.csv"
)

BIODIVERSITY_MONTHLY_CSV = (
    PROCESSED_DIR
    / "Biodiversity_indices_MONTHLY_MONTHLY5000_2008_2025_1deg.csv"
)

TOTAL_NC = (
    PROCESSED_DIR
    / "Biodiversity_indices_TOTAL_MONTHLY5000_2008_2025_1deg.nc"
)

YEARLY_NC = (
    PROCESSED_DIR
    / "Biodiversity_indices_YEARLY_MONTHLY5000_2008_2025_1deg.nc"
)

MONTHLY_NC = (
    PROCESSED_DIR
    / "Biodiversity_indices_MONTHLY_MONTHLY5000_2008_2025_1deg.nc"
)

TOTAL_MASKED_MIN5_NC = (
    PROCESSED_DIR
    / "Biodiversity_indices_TOTAL_MONTHLY5000_2008_2025_1deg_masked_min5.nc"
)

YEARLY_MASKED_MIN10_NC = (
    PROCESSED_DIR
    / "Biodiversity_indices_YEARLY_MONTHLY5000_2008_2025_1deg_masked_min10.nc"
)

MONTHLY_MASKED_MIN10_NC = (
    PROCESSED_DIR
    / "Biodiversity_indices_MONTHLY_MONTHLY5000_2008_2025_1deg_masked_min10.nc"
)

YEARLY_TIMESERIES_CSV = (
    PROCESSED_DIR
    / "Biodiversity_YEARLY_timeseries_MONTHLY5000_2008_2025.csv"
)


def calculate_biodiversity_table(df, group_columns):
    """
    Calculate occurrence-based biodiversity metrics for each requested group.

    Metrics
    -------
    occurrence_count
        Total occurrence records in the group.

    species_richness
        Number of unique species.

    shannon_index
        Shannon diversity:
            H = -sum(p_i * ln(p_i))

    simpson_index
        Simpson diversity:
            1 - sum(p_i^2)

    pielou_evenness
        Pielou evenness:
            J = H / ln(S)
        where S is species richness. J is set to 0 when S <= 1.

    Notes
    -----
    The calculation is vectorized through species-level counts rather than
    repeatedly using DataFrameGroupBy.apply, making the workflow clearer and
    avoiding pandas groupby-apply deprecation issues.
    """
    group_columns = list(group_columns)

    species_counts = (
        df.groupby(
            group_columns + ["species"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("species_occurrences")
        .reset_index()
    )

    if species_counts.empty:
        return pd.DataFrame(
            columns=group_columns + INDEX_VARIABLES
        )

    species_counts["group_total"] = (
        species_counts.groupby(
            group_columns,
            observed=True,
        )["species_occurrences"]
        .transform("sum")
    )

    species_counts["p"] = (
        species_counts["species_occurrences"]
        / species_counts["group_total"]
    )

    species_counts["neg_p_log_p"] = -(
        species_counts["p"]
        * np.log(species_counts["p"])
    )

    species_counts["p_squared"] = species_counts["p"] ** 2

    summary = (
        species_counts.groupby(
            group_columns,
            observed=True,
        )
        .agg(
            occurrence_count=("species_occurrences", "sum"),
            species_richness=("species", "size"),
            shannon_index=("neg_p_log_p", "sum"),
            sum_p_squared=("p_squared", "sum"),
        )
        .reset_index()
    )

    summary["simpson_index"] = 1.0 - summary["sum_p_squared"]

    summary["pielou_evenness"] = np.where(
        summary["species_richness"] > 1,
        summary["shannon_index"]
        / np.log(summary["species_richness"]),
        0.0,
    )

    summary = summary.drop(columns=["sum_p_squared"])

    return summary


def table_to_grid_2d(table, value_col, lat_centers, lon_centers):
    """Convert one biodiversity table column to a 2-D latitude/longitude grid."""
    grid = table.pivot(
        index="lat_bin",
        columns="lon_bin",
        values=value_col,
    )

    grid = grid.reindex(
        index=lat_centers,
        columns=lon_centers,
    )

    return grid.to_numpy()


def add_dataset_metadata(ds, temporal_scale):
    """Add consistent metadata to an analysis-ready xarray Dataset."""
    descriptions = {
        "total": (
            "Occurrence-based Antarctic biodiversity indices from GBIF, "
            "2008-2025"
        ),
        "yearly": (
            "Yearly occurrence-based Antarctic biodiversity indices from "
            "GBIF, 2008-2025"
        ),
        "monthly": (
            "Monthly occurrence-based Antarctic biodiversity indices from "
            "GBIF, 2008-2025"
        ),
    }

    ds.attrs.update(
        {
            "description": descriptions[temporal_scale],
            "source": "GBIF species occurrence records",
            "study_period": f"{YEAR_MIN}-{YEAR_MAX}",
            "latitude_range": f"{LAT_MIN} to {LAT_MAX}",
            "longitude_range": f"{LON_MIN} to {LON_MAX}",
            "grid_resolution_degrees": GRID_RES,
            "standardization": (
                "Maximum 5000 occurrence records downloaded per year-month; "
                "months with fewer records retained fully."
            ),
            "coordinate_uncertainty_filter_m": MAX_UNCERTAINTY_M,
            "note": (
                "Indices are occurrence-based biodiversity proxies, not true "
                "abundance estimates. Interpret cells with low occurrence "
                "counts cautiously."
            ),
        }
    )

    ds["lat"].attrs.update(
        {
            "long_name": "latitude",
            "units": "degrees_north",
        }
    )
    ds["lon"].attrs.update(
        {
            "long_name": "longitude",
            "units": "degrees_east",
        }
    )

    variable_metadata = {
        "occurrence_count": {
            "long_name": "GBIF occurrence count",
            "description": "Number of GBIF occurrence records in the grid cell",
        },
        "species_richness": {
            "long_name": "species richness",
            "description": "Number of unique species in the grid cell",
        },
        "shannon_index": {
            "long_name": "Shannon diversity index",
            "description": "-sum(p_i * ln(p_i))",
        },
        "simpson_index": {
            "long_name": "Simpson diversity index",
            "description": "1 - sum(p_i^2)",
        },
        "pielou_evenness": {
            "long_name": "Pielou evenness",
            "description": "Shannon diversity divided by ln(species richness)",
        },
    }

    for variable, attrs in variable_metadata.items():
        if variable in ds:
            ds[variable].attrs.update(attrs)

    return ds


def save_netcdf(ds, output_file):
    """Save an xarray Dataset as a compressed NetCDF4 file."""
    encoding = {
        variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for variable in ds.data_vars
    }

    ds.to_netcdf(
        output_file,
        engine="netcdf4",
        encoding=encoding,
    )

    print(f"Saved: {output_file}")


def build_total_dataset(bio_total, lat_centers, lon_centers):
    """Create the total 2008-2025 biodiversity NetCDF dataset."""
    ds = xr.Dataset(
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
        }
    )

    for variable in INDEX_VARIABLES:
        array = table_to_grid_2d(
            bio_total,
            variable,
            lat_centers,
            lon_centers,
        ).astype(np.float32)

        ds[variable] = (("lat", "lon"), array)

    return add_dataset_metadata(ds, temporal_scale="total")


def build_yearly_dataset(
    bio_yearly,
    years,
    lat_centers,
    lon_centers,
):
    """Create the yearly biodiversity NetCDF dataset."""
    ds = xr.Dataset(
        coords={
            "year": years,
            "lat": lat_centers,
            "lon": lon_centers,
        }
    )

    for variable in INDEX_VARIABLES:
        array = np.full(
            (
                len(years),
                len(lat_centers),
                len(lon_centers),
            ),
            np.nan,
            dtype=np.float32,
        )

        for year_index, year in enumerate(years):
            temp = bio_yearly[
                bio_yearly["year"] == year
            ]

            if not temp.empty:
                array[year_index, :, :] = table_to_grid_2d(
                    temp,
                    variable,
                    lat_centers,
                    lon_centers,
                ).astype(np.float32)

        ds[variable] = (
            ("year", "lat", "lon"),
            array,
        )

    return add_dataset_metadata(ds, temporal_scale="yearly")


def build_monthly_dataset(
    bio_monthly,
    years,
    months,
    lat_centers,
    lon_centers,
):
    """Create the year-month biodiversity NetCDF dataset."""
    ds = xr.Dataset(
        coords={
            "year": years,
            "month": months,
            "lat": lat_centers,
            "lon": lon_centers,
        }
    )

    for variable in INDEX_VARIABLES:
        array = np.full(
            (
                len(years),
                len(months),
                len(lat_centers),
                len(lon_centers),
            ),
            np.nan,
            dtype=np.float32,
        )

        for year_index, year in enumerate(years):
            for month_index, month in enumerate(months):
                temp = bio_monthly[
                    (bio_monthly["year"] == year)
                    & (bio_monthly["month"] == month)
                ]

                if not temp.empty:
                    array[
                        year_index,
                        month_index,
                        :,
                        :,
                    ] = table_to_grid_2d(
                        temp,
                        variable,
                        lat_centers,
                        lon_centers,
                    ).astype(np.float32)

        ds[variable] = (
            ("year", "month", "lat", "lon"),
            array,
        )

    return add_dataset_metadata(ds, temporal_scale="monthly")


def create_yearly_timeseries(bio_yearly):
    """
    Create the yearly thesis summary table using the original >=5-record
    validity rule.
    """
    bio_yearly_valid = bio_yearly[
        bio_yearly["occurrence_count"] >= MIN_RECORDS_TIMESERIES
    ].copy()

    bio_ts = (
        bio_yearly_valid.groupby("year")
        .agg(
            occurrence_count=("occurrence_count", "sum"),
            species_richness=("species_richness", "mean"),
            shannon_index=("shannon_index", "mean"),
            simpson_index=("simpson_index", "mean"),
            pielou_evenness=("pielou_evenness", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )

    bio_ts.to_csv(YEARLY_TIMESERIES_CSV, index=False)

    print(f"Saved: {YEARLY_TIMESERIES_CSV}")

    return bio_ts


def format_files_for_analysis():
    """
    Calculate biodiversity indices and create all analysis-ready CSV/NetCDF
    products.
    """
    if not GRIDDED_FILE.exists():
        raise FileNotFoundError(
            "The gridded occurrence file does not exist. "
            "Run the PROCESS GBIF stage first.\n"
            f"Expected file: {GRIDDED_FILE}"
        )

    print("\n" + "=" * 72)
    print("FORMATTING BIODIVERSITY FILES FOR ANALYSIS")
    print("=" * 72)

    df = pd.read_csv(
        GRIDDED_FILE,
        low_memory=False,
        dtype={"species": "string"},
    )

    required_columns = {
        "year",
        "month",
        "lat_bin",
        "lon_bin",
        "species",
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            "Gridded input is missing required columns: "
            + ", ".join(missing)
        )

    # Ensure expected numeric dtypes after reading from CSV.
    for column in ["year", "month", "lat_bin", "lon_bin"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "year",
            "month",
            "lat_bin",
            "lon_bin",
            "species",
        ]
    ).copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["lat_bin"] = df["lat_bin"].astype(float)
    df["lon_bin"] = df["lon_bin"].astype(float)

    # Biodiversity index tables
    print("Calculating total biodiversity indices...")
    bio_total = calculate_biodiversity_table(
        df,
        ["lat_bin", "lon_bin"],
    )
    bio_total.to_csv(BIODIVERSITY_TOTAL_CSV, index=False)
    print(f"Saved: {BIODIVERSITY_TOTAL_CSV}")

    print("Calculating yearly biodiversity indices...")
    bio_yearly = calculate_biodiversity_table(
        df,
        ["year", "lat_bin", "lon_bin"],
    )
    bio_yearly.to_csv(BIODIVERSITY_YEARLY_CSV, index=False)
    print(f"Saved: {BIODIVERSITY_YEARLY_CSV}")

    print("Calculating monthly biodiversity indices...")
    bio_monthly = calculate_biodiversity_table(
        df,
        ["year", "month", "lat_bin", "lon_bin"],
    )
    bio_monthly.to_csv(BIODIVERSITY_MONTHLY_CSV, index=False)
    print(f"Saved: {BIODIVERSITY_MONTHLY_CSV}")

    # Analysis-ready NetCDF datasets
    _, _, lat_centers, lon_centers = build_grid_coordinates()

    years = np.arange(
        YEAR_MIN,
        YEAR_MAX + 1,
        dtype=int,
    )
    months = np.arange(1, 13, dtype=int)

    print("Building total NetCDF dataset...")
    ds_total = build_total_dataset(
        bio_total,
        lat_centers,
        lon_centers,
    )
    save_netcdf(ds_total, TOTAL_NC)

    print("Building yearly NetCDF dataset...")
    ds_yearly = build_yearly_dataset(
        bio_yearly,
        years,
        lat_centers,
        lon_centers,
    )
    save_netcdf(ds_yearly, YEARLY_NC)

    print("Building monthly NetCDF dataset...")
    ds_monthly = build_monthly_dataset(
        bio_monthly,
        years,
        months,
        lat_centers,
        lon_centers,
    )
    save_netcdf(ds_monthly, MONTHLY_NC)

    # Minimum-record masks
    ds_total_masked = ds_total.where(
        ds_total["occurrence_count"] >= MIN_RECORDS_MAP
    )
    ds_total_masked.attrs["minimum_occurrence_count_mask"] = (
        MIN_RECORDS_MAP
    )
    save_netcdf(
        ds_total_masked,
        TOTAL_MASKED_MIN5_NC,
    )

    ds_yearly_masked = ds_yearly.where(
        ds_yearly["occurrence_count"] >= MIN_RECORDS_STATS
    )
    ds_yearly_masked.attrs["minimum_occurrence_count_mask"] = (
        MIN_RECORDS_STATS
    )
    save_netcdf(
        ds_yearly_masked,
        YEARLY_MASKED_MIN10_NC,
    )

    ds_monthly_masked = ds_monthly.where(
        ds_monthly["occurrence_count"] >= MIN_RECORDS_STATS
    )
    ds_monthly_masked.attrs["minimum_occurrence_count_mask"] = (
        MIN_RECORDS_STATS
    )
    save_netcdf(
        ds_monthly_masked,
        MONTHLY_MASKED_MIN10_NC,
    )

    # Yearly summary time series
    bio_ts = create_yearly_timeseries(bio_yearly)

    # Final quality-control report
    print("\n" + "=" * 72)
    print("FINAL BIODIVERSITY DATA CHECK")
    print("=" * 72)
    print(f"Gridded occurrence records: {len(df):,}")
    print(f"Unique species:             {df['species'].nunique():,}")
    print(f"Total biodiversity cells:   {len(bio_total):,}")
    print(f"Yearly biodiversity rows:   {len(bio_yearly):,}")
    print(f"Monthly biodiversity rows:  {len(bio_monthly):,}")
    print(f"Yearly time-series rows:     {len(bio_ts):,}")

    print("\nMAIN THESIS ANALYSIS FILES")
    print(f"Map NetCDF (minimum 5 records):")
    print(f"  {TOTAL_MASKED_MIN5_NC}")

    print(f"\nYearly analysis NetCDF (minimum 10 records):")
    print(f"  {YEARLY_MASKED_MIN10_NC}")

    print(f"\nMonthly analysis NetCDF (minimum 10 records):")
    print(f"  {MONTHLY_MASKED_MIN10_NC}")

    print("\nYearly biodiversity time series:")
    print(f"  {YEARLY_TIMESERIES_CSV}")

    # Close datasets after all derived files have been written.
    ds_total.close()
    ds_yearly.close()
    ds_monthly.close()
    ds_total_masked.close()
    ds_yearly_masked.close()
    ds_monthly_masked.close()


# MAIN WORKFLOW

def main():
    """Run the selected stages in the correct order."""
    print_configuration()

    if RUN_DOWNLOAD:
        download_all_gbif_months()
    else:
        print("\nDOWNLOAD GBIF stage skipped.")

    if RUN_PROCESS:
        process_gbif_data()
    else:
        print("\nPROCESS GBIF stage skipped.")

    if RUN_FORMAT_FOR_ANALYSIS:
        format_files_for_analysis()
    else:
        print("\nFORMATTED FILES FOR ANALYSIS stage skipped.")

    print("\n" + "=" * 72)
    print("PIPELINE FINISHED")
    print("=" * 72)


if __name__ == "__main__":
    main()
