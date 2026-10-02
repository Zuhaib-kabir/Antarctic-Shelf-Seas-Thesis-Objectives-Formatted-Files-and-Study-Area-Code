"""
ANTARCTIC CARBON AND BIOLOGICAL DATA PROCESSING PIPELINE

Analysed period:
2008-2025

Study domain:
Antarctic / Southern Ocean study region:
    latitude  = 90°S to 60°S
    longitude = 180°W to 180°E

Purpose:
This script consolidates and professionally organizes the NON-GBIF data
processing code used across the thesis carbon and biological workflows.

It is designed to:
    1. Mount Google Drive in Google Colab.
    2. Install/import the required Python libraries.
    3. Validate and standardize NetCDF/CSV inputs.
    4. Standardize coordinate names and longitude convention.
    5. Restrict data to 2008-2025 and 90°S-60°S.
    6. Convert irregular/high-frequency data to monthly means when needed.
    7. Calculate area-weighted Antarctic marginal-sea time series.
    8. Create integrated NPP (NPPINT = NPPv × Zeu).
    9. Create PFT seasonal mean and relative-contribution tables.
   10. Create carbon/biological sea-wise monthly tables and response metrics.
   11. Create biological-carbon-pump (BCP) MLD/stratification tables.
   12. Create lag-correlation / bootstrap lag-analysis files.
   13. Create a non-GBIF integrated sea-classification input table.

IMPORTANT:
This file is focused on
data formatting, quality control, and creation of analysis-ready files.

Source workflows consolidated:
- MyThesis_Obj3_fig17.py:
    PIC/POC data checks and unit logic.
- MyThesis_Obj3_fig18To20.py:
    Fig. 22 sea-ice / carbon / bloom low-RAM processing.
- MyThesis_Obj3_fig21_23.py:
    PFT seasonal processing, NPPINT NetCDF creation, Fig. 23 sea-wise file.
- MyThesis_Obj3_fig24_30.py:
    Fig. 24 BCP processing and Fig. 25 lag-processing blocks.
    GBIF-dependent Fig. 26-30 processing is not included.
- MyThesis_Obj3_fig31.py:
    Integrated sea-classification input processing, rewritten without GBIF.

Main output files
DATA directory:
    NPPINT_monthly_2008_2025_SO.nc
    SIC_seawise_monthly_2008_2025_LOW_RAM.csv

PROCESSED directory:
    Obj2Fig21_PFT_seasonal_mean.csv
    Obj2Fig21_PFT_relative_contribution.csv

    Fig22_seawise_monthly_timeseries_LOW_RAM.csv
    Fig22_seawise_metrics_LOW_RAM.csv

    Fig23_seawise_monthly_timeseries_LOW_RAM.csv

    Fig24_BCP_seawise_monthly_LOW_RAM.csv
    Fig24_BCP_seawise_summary_ACTIVE_LOW_RAM.csv

    Fig25_seawise_monthly_SIC_POC_CHL_NPPint_LOW_RAM.csv
    Fig25_lag_correlation_seawise.csv
    Fig25_lag_correlation_mean.csv
    Fig25_best_lag_bootstrap_by_sea_season_variable.csv
    Fig25_best_lag_summary_by_sea_season_variable.csv

    sea_classification_input_2008_2025_NO_GBIF.csv

Scientific notes:
1. PIC:
   The original thesis workflow states that PIC is supplied in mol C m^-3.
   When the units indicate mol m^-3, this script converts PIC to mg C m^-3:
       PIC_mg_C_m3 = PIC_mol_C_m3 × 12.0107 × 1000

2. POC:
   Retained in its source units. The thesis files identify it as mg C m^-3.

3. NPPINT:
   Approximate integrated productivity:
       NPPINT = NPPv × Zeu
   where:
       NPPv = mg C m^-3 d^-1
       Zeu  = m
       NPPINT = mg C m^-2 d^-1

4. Area-weighted sea means:
   Latitude weighting uses cos(latitude).

5. Antarctic marginal-sea sectors:
   The same 13 non-overlapping longitude sectors used in the thesis are used
   here, including dateline handling for the Ross Sea.

6. Stratification:
   If the Argo-derived stratification file contains `nprof`, the sea-wise
   stratification mean is weighted by:
       nprof × cos(latitude)
   Otherwise an area-weighted mean is used.

7. This script does not change the thesis analysis period or sea definitions.
"""


# MOUNT GOOGLE DRIVE
from google.colab import drive

drive.mount("/content/drive")


# INSTALL REQUIRED LIBRARIES
# Normal .py files cannot use notebook shell syntax such as:
#     !pip install ...
# Therefore package installation is handled through subprocess.
import importlib.util
import subprocess
import sys


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "dask": "dask",
    "scipy": "scipy",
}


def install_missing_packages():
    """Install only packages that are not already available."""
    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if not missing:
        print("All required Python packages are already installed.")
        return

    print("Installing missing packages:", ", ".join(missing))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing]
    )


install_missing_packages()


# IMPORTS
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore")


# GLOBAL RUN CONTROLS

RUN_CREATE_NPPINT = True
RUN_PFT_SEASONAL_TABLES = True
RUN_CARBON_BIOLOGY_SEAWISE = True
RUN_BCP_MLD_STRAT = True
RUN_LAG_ANALYSIS = True
RUN_CLASSIFICATION_NO_GBIF = True


# GLOBAL PATHS

THESIS_DIR = Path("/content/drive/MyDrive/SAM_Thesis")
DATA_DIR = THESIS_DIR / "Data"
PROCESSED_DIR = THESIS_DIR / "Processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# INPUT FILES

FILES = {
    "SIC": DATA_DIR / "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc",
    "SIT": DATA_DIR / "GLORYS_SIT_sithick_monthly_2008_2025_SO.nc",
    "SST": DATA_DIR / "OSTIA_SST_monthly_2008_2025_SO.nc",
    "MLD": DATA_DIR / "MLD_monthly_2008_2025_SO.nc",
    "STRAT": PROCESSED_DIR / "stratification_monthly_2008_2025_SO.nc",
    "FWC": PROCESSED_DIR / "freshwater_content_monthly_2008_2025_SO.nc",
    "PIC": DATA_DIR / "PIC_monthly_2008_2025_SO.nc",
    "POC": DATA_DIR / "POC_monthly_2008_2025_SO.nc",
    "CHL": DATA_DIR / "CHL_monthly_2008_2025_SO.nc",
    "NPPV": DATA_DIR / "nppv.nc",
    "ZEU": DATA_DIR / "ZEU_monthly_2008_2025_SO.nc",
    "DIATO": DATA_DIR / "DIATO_monthly_2008_2025_SO.nc",
    "HAPTO": DATA_DIR / "HAPTO_monthly_2008_2025_SO.nc",
    "PICO": DATA_DIR / "PICO_monthly_2008_2025_SO.nc",
}


# OUTPUT FILES

NPPINT_FILE = DATA_DIR / "NPPINT_monthly_2008_2025_SO.nc"
SIC_SEAWISE_FILE = DATA_DIR / "SIC_seawise_monthly_2008_2025_LOW_RAM.csv"

PFT_MEAN_FILE = PROCESSED_DIR / "Obj2Fig21_PFT_seasonal_mean.csv"
PFT_REL_FILE = PROCESSED_DIR / "Obj2Fig21_PFT_relative_contribution.csv"

FIG22_MONTHLY_FILE = PROCESSED_DIR / "Fig22_seawise_monthly_timeseries_LOW_RAM.csv"
FIG22_METRICS_FILE = PROCESSED_DIR / "Fig22_seawise_metrics_LOW_RAM.csv"
FIG23_MONTHLY_FILE = PROCESSED_DIR / "Fig23_seawise_monthly_timeseries_LOW_RAM.csv"
FIG24_MONTHLY_FILE = PROCESSED_DIR / "Fig24_BCP_seawise_monthly_LOW_RAM.csv"
FIG24_SUMMARY_FILE = PROCESSED_DIR / "Fig24_BCP_seawise_summary_ACTIVE_LOW_RAM.csv"

FIG25_MONTHLY_FILE = PROCESSED_DIR / "Fig25_seawise_monthly_SIC_POC_CHL_NPPint_LOW_RAM.csv"
FIG25_LAG_SEAWISE_FILE = PROCESSED_DIR / "Fig25_lag_correlation_seawise.csv"
FIG25_LAG_MEAN_FILE = PROCESSED_DIR / "Fig25_lag_correlation_mean.csv"
FIG25_BOOT_FILE = PROCESSED_DIR / "Fig25_best_lag_bootstrap_by_sea_season_variable.csv"
FIG25_SUMMARY_FILE = PROCESSED_DIR / "Fig25_best_lag_summary_by_sea_season_variable.csv"

CLASSIFICATION_FILE = PROCESSED_DIR / "sea_classification_input_2008_2025_NO_GBIF.csv"


# GLOBAL SETTINGS

START_DATE = "2008-01-01"
END_DATE = "2025-12-31"
START_MONTH = "2008-01-01"
END_MONTH = "2025-12-01"

LAT_MIN = -90.0
LAT_MAX = -60.0

MOLAR_MASS_C_G_MOL = 12.0107
PIC_MOL_TO_MG_C = MOLAR_MASS_C_G_MOL * 1000.0

ACTIVE_MONTHS = [9, 10, 11, 12, 1, 2, 3, 4]

SEASONS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Early autumn": [3, 4],
}
SEASON_ORDER = ["Spring", "Summer", "Early autumn"]

OPEN_CHUNKS = {"time": 1, "lat": 120, "lon": 720}

RANDOM_SEED = 42
RETREAT_THRESHOLD_PERCENT = 15.0
POST_BLOOM_MONTHS = 3
LAGS = np.arange(-3, 4, 1)
ICE_DRIVER_MODE = "retreat_anomaly"
N_BOOT = 300


# ANTARCTIC MARGINAL SEA SECTORS
SEA_INFO = [
    ("WED", "Weddell Sea", "60°W–20°W", -60, -20),
    ("KHV", "King Haakon VII Sea", "20°W–0°", -20, 0),
    ("RLS", "Riiser-Larsen Sea", "0°–10°E", 0, 10),
    ("LAZ", "Lazarev Sea", "10°E–30°E", 10, 30),
    ("COS", "Cosmonauts Sea", "30°E–50°E", 30, 50),
    ("COO", "Cooperation Sea", "50°E–70°E", 50, 70),
    ("DAV", "Davis Sea", "70°E–90°E", 70, 90),
    ("MAW", "Mawson Sea", "90°E–130°E", 90, 130),
    ("DUR", "D'Urville Sea", "130°E–150°E", 130, 150),
    ("SOM", "Somov Sea", "150°E–170°E", 150, 170),
    ("ROS", "Ross Sea", "170°E–130°W", 170, -130),
    ("AMU", "Amundsen Sea", "130°W–100°W", -130, -100),
    ("BEL", "Bellingshausen Sea", "100°W–60°W", -100, -60),
]

SEA_CODES = [item[0] for item in SEA_INFO]


# VARIABLE NAME CANDIDATES

VAR_CANDIDATES = {
    "SIC": ["SIC", "sic", "SIF", "sif", "siconc", "sea_ice_fraction",
            "ice_conc", "sea_ice_concentration"],
    "SST": ["SST", "sst", "analysed_sst", "thetao"],
    "MLD": ["MLD", "mld", "mlotst", "mixed_layer_depth", "mlotst_mean"],
    "SIT": ["SIT", "sithick", "sea_ice_thickness", "sithick_mean"],
    "STRAT": ["stratification", "STRAT", "strat", "stratification_index",
              "delta_sigma", "delta_sigma_theta", "density_difference", "sigma_diff"],
    "FWC": ["freshwater_content", "FWC", "fwc"],
    "PIC": ["PIC", "pic", "particulate_inorganic_carbon"],
    "POC": ["POC", "poc", "particulate_organic_carbon"],
    "CHL": ["CHL", "chl", "chlor_a", "chlorophyll", "chlorophyll_a"],
    "NPPV": ["NPP", "npp", "nppv", "NPPV", "net_primary_production"],
    "NPPINT": ["NPPINT", "NPPint", "nppint", "NPP_INT",
               "NPP_integrated", "integrated_npp", "npp_int"],
    "ZEU": ["ZEU", "Zeu", "zeu", "euphotic_depth", "euphotic_depth_z"],
    "DIATO": ["DIATO", "diato", "Diato", "DIATOM", "diatom", "Diatoms", "diatoms"],
    "HAPTO": ["HAPTO", "hapto", "Hapto", "HAPTOPHYTE", "haptophyte",
              "Haptophytes", "haptophytes"],
    "PICO": ["PICO", "pico", "Pico", "PICOPHYTO", "picophyto",
             "Picophytoplankton", "picophytoplankton"],
}


# SHARED HELPERS

def check_file(path, label=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label or path.name} not found:\n{path}")
    print(f"{label or path.name}: {path}")
    return path


def check_files(keys):
    print("\nChecking required input files...")
    for key in keys:
        check_file(FILES[key], key)
    print("Required files found.")


def month_start_datetime(values):
    s = pd.Series(pd.to_datetime(values))
    return s.dt.to_period("M").dt.to_timestamp()


def full_monthly_range():
    return pd.date_range(START_MONTH, END_MONTH, freq="MS")


def normalize_time_column(df, name="time"):
    if name not in df.columns:
        raise ValueError(f"No '{name}' column. Columns: {list(df.columns)}")
    out = df.copy()
    out[name] = month_start_datetime(out[name])
    return out.sort_values(name).reset_index(drop=True)


def standardize_coords(ds):
    rename_dict = {}

    for name in list(ds.coords):
        lower = name.lower()
        if lower in ["latitude", "nav_lat", "y"]:
            rename_dict[name] = "lat"
        elif lower in ["longitude", "nav_lon", "x"]:
            rename_dict[name] = "lon"
        elif lower in ["valid_time", "time_counter"]:
            rename_dict[name] = "time"

    for name in list(ds.dims):
        lower = name.lower()
        if lower in ["latitude", "nav_lat", "y"] and name not in rename_dict:
            rename_dict[name] = "lat"
        elif lower in ["longitude", "nav_lon", "x"] and name not in rename_dict:
            rename_dict[name] = "lon"
        elif lower in ["valid_time", "time_counter"] and name not in rename_dict:
            rename_dict[name] = "time"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" in ds.coords:
        try:
            if float(ds["lon"].max()) > 180:
                ds = ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0)
        except Exception:
            pass

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def prepare_antarctic_dataset(ds, require_time=True):
    ds = standardize_coords(ds)

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Dataset must contain latitude and longitude coordinates.")

    if require_time:
        if "time" not in ds.coords:
            raise ValueError("Dataset has no time coordinate.")
        ds = ds.sel(time=slice(START_DATE, END_DATE))

    ds = ds.where((ds["lat"] >= LAT_MIN) & (ds["lat"] <= LAT_MAX), drop=True)

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def open_dataset_safe(path, chunks=None):
    if chunks is None:
        chunks = OPEN_CHUNKS

    errors = []
    for engine in ["h5netcdf", "netcdf4", None]:
        try:
            kwargs = {"decode_times": True, "chunks": chunks}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(str(path), **kwargs)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")

    raise RuntimeError(f"Could not open:\n{path}\n" + "\n".join(errors))


def find_var(ds, label, require_time=True):
    for name in VAR_CANDIDATES.get(label, []):
        if name in ds.data_vars:
            return name

    candidates = []
    for var in ds.data_vars:
        da = ds[var]
        if require_time:
            if "time" in da.dims and "lat" in da.dims and "lon" in da.dims:
                candidates.append(var)
        elif "lat" in da.dims and "lon" in da.dims:
            candidates.append(var)

    candidates = [v for v in candidates if v.lower() != "nprof"]

    if len(candidates) == 1:
        print(f"Using detected variable for {label}: {candidates[0]}")
        return candidates[0]

    raise ValueError(
        f"Could not detect variable for {label}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def squeeze_depth_if_present(da):
    for dim in list(da.dims):
        if dim.lower() in ["depth", "lev", "level", "deptht"]:
            print(f"Squeezing depth dimension: {dim}")
            da = da.isel({dim: 0}, drop=True)
    return da


def make_monthly_if_needed(da):
    if "time" not in da.dims:
        return da

    if da.sizes.get("time", 0) > 216:
        print("More than 216 time steps detected; resampling to monthly mean.")
        da = da.resample(time="1MS").mean(skipna=True)

    da = da.assign_coords(time=month_start_datetime(da["time"].values).values)

    idx = pd.Index(pd.to_datetime(da["time"].values))
    if idx.duplicated().any():
        da = da.groupby("time").mean(skipna=True)

    return da


def lon_mask(lon, lon_min, lon_max):
    if lon_min < lon_max:
        return (lon >= lon_min) & (lon < lon_max)
    return (lon >= lon_min) | (lon < lon_max)


def area_weighted_sea_mean(da, lon_min, lon_max):
    sub = da.where(lon_mask(da["lon"], lon_min, lon_max), drop=True)
    if sub.sizes.get("lon", 0) == 0:
        raise ValueError(f"No longitude cells for sector {lon_min} to {lon_max}.")
    weights = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(weights).mean(dim=("lat", "lon"), skipna=True)


def convert_sic_to_percent(da):
    sample_max = float(da.max(skipna=True).compute())
    if sample_max <= 1.5:
        print("SIC appears to be fraction; converting to percent.")
        da = da * 100.0
    da = da.clip(min=0.0, max=100.0)
    da.attrs["units"] = "%"
    return da


def convert_pic_to_mg_c_m3(da):
    units = str(da.attrs.get("units", "")).strip()
    normalized = units.lower().replace(" ", "")

    if "mmol" in normalized:
        print("PIC mmol units detected; converting to mg C m^-3.")
        out = da * MOLAR_MASS_C_G_MOL
    elif "mol" in normalized and "mg" not in normalized:
        print("PIC mol units detected; converting to mg C m^-3.")
        out = da * PIC_MOL_TO_MG_C
    elif "mg" in normalized:
        print("PIC already appears mg-based; no conversion.")
        out = da
    else:
        print(f"WARNING: unrecognized PIC units '{units or 'unknown'}'; no conversion.")
        out = da

    out.attrs = dict(da.attrs)
    if out is not da:
        out.attrs["units"] = "mg C m^-3"
        out.attrs["conversion_note"] = (
            "Converted using carbon molar mass 12.0107 g mol^-1."
        )
    return out


def convert_temperature_to_celsius_if_needed(da, label="temperature"):
    try:
        sample = (
            da.isel(time=0).mean(skipna=True).compute().item()
            if "time" in da.dims
            else da.mean(skipna=True).compute().item()
        )
        if np.isfinite(sample) and sample > 100:
            print(f"{label}: converting Kelvin to Celsius.")
            out = da - 273.15
            out.attrs = dict(da.attrs)
            out.attrs["units"] = "degree_Celsius"
            return out
    except Exception:
        pass
    return da


def transform_variable(da, label):
    da = squeeze_depth_if_present(da.astype("float32"))
    da = da.where(np.isfinite(da))

    if label == "SIC":
        da = convert_sic_to_percent(da)
    elif label == "PIC":
        da = convert_pic_to_mg_c_m3(da)
    elif label == "SST":
        da = convert_temperature_to_celsius_if_needed(da, "SST")

    return da


def extract_seawise_monthly(path, label):
    check_file(path, label)

    ds = prepare_antarctic_dataset(open_dataset_safe(path))
    varname = find_var(ds, label)
    da = make_monthly_if_needed(transform_variable(ds[varname], label))

    print(f"\n{label}: variable={varname}")
    print("Shape:", da.shape)
    print("Units:", da.attrs.get("units", "unknown"))

    out = pd.DataFrame({"time": full_monthly_range()})

    for sea_code, sea_name, lon_label, lon_min, lon_max in SEA_INFO:
        print(f"  {label} | {sea_code}: {sea_name} ({lon_label})")
        ts = area_weighted_sea_mean(da, lon_min, lon_max)

        with ProgressBar():
            loaded = ts.compute()

        temp = pd.DataFrame({
            "time": month_start_datetime(loaded["time"].values),
            f"{label}_{sea_code}": loaded.values.astype("float32"),
        })
        out = out.merge(temp, on="time", how="left")

        del ts, loaded, temp
        gc.collect()

    ds.close()
    del ds, da
    gc.collect()

    return out


def merge_monthly_tables(tables):
    merged = pd.DataFrame({"time": full_monthly_range()})
    for table in tables:
        merged = merged.merge(normalize_time_column(table), on="time", how="left")
    return merged


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("\nSaved CSV:", path)
    print("Shape:", df.shape)
    print("Total missing values:", int(df.isna().sum().sum()))


def seasonal_mean_by_year(da, months, min_months=2):
    da_s = da.where(da["time"].dt.month.isin(months), drop=True)
    if da_s.sizes.get("time", 0) == 0:
        return None

    sy = da_s["time"].dt.year
    if 12 in months and 1 in months:
        sy = xr.where(da_s["time"].dt.month == 12, sy + 1, sy)

    da_s = da_s.assign_coords(season_year=sy.rename("season_year"))
    yearly = da_s.groupby("season_year").mean("time", skipna=True)

    sy_values = np.asarray(sy.values)
    valid_years = [
        int(year)
        for year in np.unique(sy_values)
        if np.sum(sy_values == year) >= min_months
    ]

    if not valid_years:
        return None

    return yearly.sel(season_year=valid_years)


def seasonal_climatology(da, months, season_name):
    min_months = 1 if season_name == "Early autumn" else 2
    yearly = seasonal_mean_by_year(da, months, min_months)

    if yearly is None:
        raise ValueError(f"No valid seasonal data for {season_name}.")

    return yearly.sel(season_year=slice(2008, 2025)).mean(
        "season_year", skipna=True
    )


# FILE: nppv.nc + ZEU_monthly_2008_2025_SO.nc
# OUTPUT: NPPINT_monthly_2008_2025_SO.nc
def create_nppint_file():
    check_files(["NPPV", "ZEU"])

    ds_npp = prepare_antarctic_dataset(
        open_dataset_safe(FILES["NPPV"], chunks={"time": 1})
    )
    ds_zeu = prepare_antarctic_dataset(
        open_dataset_safe(FILES["ZEU"], chunks={"time": 1})
    )

    npp_name = find_var(ds_npp, "NPPV")
    zeu_name = find_var(ds_zeu, "ZEU")

    npp = make_monthly_if_needed(transform_variable(ds_npp[npp_name], "NPPV"))
    zeu = make_monthly_if_needed(transform_variable(ds_zeu[zeu_name], "ZEU"))

    common = np.intersect1d(
        npp["time"].values.astype("datetime64[M]"),
        zeu["time"].values.astype("datetime64[M]"),
    )
    if len(common) == 0:
        raise ValueError("No common monthly times between NPPv and Zeu.")

    npp = npp.assign_coords(time=npp["time"].values.astype("datetime64[M]")).sel(time=common)
    zeu = zeu.assign_coords(time=zeu["time"].values.astype("datetime64[M]")).sel(time=common)

    zeu_grid = zeu.interp(lat=npp["lat"], lon=npp["lon"], method="nearest")

    nppint = (npp * zeu_grid).astype("float32")
    nppint.name = "NPPINT"
    nppint.attrs.update({
        "long_name": "Approximate depth-integrated net primary productivity",
        "units": "mg C m-2 d-1",
        "calculation": "NPPINT = NPPv × Zeu",
        "note": "Approximate euphotic-zone integrated NPP; missing values remain NaN.",
    })

    ds_out = nppint.to_dataset()
    ds_out.attrs.update({
        "title": "Approximate monthly integrated NPP over Antarctic shelf seas",
        "spatial_domain": "90S-60S, 180W-180E",
        "temporal_coverage": "2008-2025 monthly",
        "method": "NPPINT = NPPv × Zeu",
    })

    encoding = {
        "NPPINT": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 5,
            "shuffle": True,
            "_FillValue": -9999.0,
            "chunksizes": (
                1,
                min(120, ds_out.sizes["lat"]),
                min(720, ds_out.sizes["lon"]),
            ),
        }
    }

    if NPPINT_FILE.exists():
        NPPINT_FILE.unlink()

    with ProgressBar():
        ds_out.to_netcdf(NPPINT_FILE, engine="netcdf4", encoding=encoding)

    print("\nSaved:", NPPINT_FILE)

    ds_npp.close()
    ds_zeu.close()
    del ds_npp, ds_zeu, ds_out, npp, zeu, zeu_grid, nppint
    gc.collect()


# FILES: DIATO / HAPTO / PICO
# OUTPUTS: PFT seasonal mean + relative contribution CSV

def process_pft_seasonal_tables():
    check_files(["DIATO", "HAPTO", "PICO"])
    pfts = ["DIATO", "HAPTO", "PICO"]

    means = {
        pft: pd.DataFrame(index=SEA_CODES, columns=SEASON_ORDER, dtype=float)
        for pft in pfts
    }

    for pft in pfts:
        ds = prepare_antarctic_dataset(
            open_dataset_safe(
                FILES[pft],
                chunks={"time": 12, "lat": 120, "lon": 1200},
            )
        )
        varname = find_var(ds, pft)
        da = make_monthly_if_needed(transform_variable(ds[varname], pft))
        da = da.where(da >= 0)

        for season_name in SEASON_ORDER:
            clim = seasonal_climatology(da, SEASONS[season_name], season_name)

            # Add a temporary time dimension so the shared sea-mean function can
            # be used without duplicating weighting code.
            clim3d = clim.expand_dims(time=[pd.Timestamp("2000-01-01")])

            for sea_code, _, _, lon_min, lon_max in SEA_INFO:
                value = area_weighted_sea_mean(
                    clim3d, lon_min, lon_max
                ).squeeze("time", drop=True)

                means[pft].loc[sea_code, season_name] = float(
                    value.compute().item()
                )

        ds.close()
        del ds, da
        gc.collect()

    rel = {
        pft: pd.DataFrame(index=SEA_CODES, columns=SEASON_ORDER, dtype=float)
        for pft in pfts
    }

    for season in SEASON_ORDER:
        total = (
            means["DIATO"][season]
            + means["HAPTO"][season]
            + means["PICO"][season]
        ).replace(0, np.nan)

        for pft in pfts:
            rel[pft][season] = means[pft][season] / total * 100.0

    mean_rows, rel_rows = [], []

    for pft in pfts:
        a = means[pft].copy()
        a.insert(0, "Sea", a.index)
        a.insert(0, "PFT", pft)
        mean_rows.append(a.reset_index(drop=True))

        b = rel[pft].copy()
        b.insert(0, "Sea", b.index)
        b.insert(0, "PFT", pft)
        rel_rows.append(b.reset_index(drop=True))

    save_csv(pd.concat(mean_rows, ignore_index=True), PFT_MEAN_FILE)
    save_csv(pd.concat(rel_rows, ignore_index=True), PFT_REL_FILE)


# FILES: SIC + PIC + POC + CHL
# OUTPUTS: SIC sea-wise, Fig22 monthly/metrics, Fig23 monthly

def circular_mean_month(months):
    values = np.asarray(months, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan

    angles = 2 * np.pi * (values - 1) / 12.0
    angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    if angle < 0:
        angle += 2 * np.pi
    return angle / (2 * np.pi) * 12.0 + 1.0


def month_to_day(month):
    return np.nan if not np.isfinite(month) else (month - 1.0) * 30.44 + 15.0


def active_mean(series):
    return series[series.index.month.isin(ACTIVE_MONTHS)].mean(skipna=True)


def build_season_year_events(df_sea, sea_code):
    df = df_sea.copy().sort_values("time").reset_index(drop=True)
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["season_year"] = np.where(df["month"] >= 7, df["year"], df["year"] - 1)

    rows = []

    for sy, group in df.groupby("season_year"):
        group = group.sort_values("time").reset_index(drop=True)
        if len(group) < 8:
            continue

        search = group[group["month"].isin(ACTIVE_MONTHS)].copy()
        if search.empty:
            continue

        below = search[search["SIC"] < RETREAT_THRESHOLD_PERCENT]
        if len(below):
            retreat = below.iloc[0]
        else:
            search["dSIC"] = search["SIC"].diff()
            if search["dSIC"].notna().sum() == 0:
                continue
            retreat = search.loc[search["dSIC"].idxmin()]

        retreat_time = retreat["time"]
        idx = group.index[group["time"] == retreat_time].tolist()
        if not idx:
            continue

        retreat_idx = idx[0]
        peak_window = group.iloc[
            retreat_idx:retreat_idx + POST_BLOOM_MONTHS + 1
        ].copy()

        if peak_window["CHL"].notna().sum() == 0:
            continue

        bloom = peak_window.loc[peak_window["CHL"].idxmax()]
        bloom_time = bloom["time"]
        lag_months = (
            (bloom_time.year - retreat_time.year) * 12
            + bloom_time.month - retreat_time.month
        )

        baseline = group.iloc[max(0, retreat_idx - 2):retreat_idx]
        chl_base = baseline["CHL"].mean(skipna=True)
        poc_base = baseline["POC"].mean(skipna=True)
        chl_peak = bloom["CHL"]
        poc_peak = peak_window["POC"].max(skipna=True)

        rows.append({
            "sea": sea_code,
            "season_year": int(sy),
            "retreat_time": retreat_time,
            "retreat_month": retreat_time.month,
            "bloom_time": bloom_time,
            "bloom_month": bloom_time.month,
            "lag_days": lag_months * 30.44,
            "chl_response": chl_peak - chl_base if np.isfinite(chl_base) else np.nan,
            "poc_response": poc_peak - poc_base if np.isfinite(poc_base) else np.nan,
        })

    return pd.DataFrame(rows)


def create_fig22_metrics(monthly):
    rows = []

    for sea in SEA_CODES:
        temp = pd.DataFrame({
            "time": pd.to_datetime(monthly["time"]),
            "SIC": monthly[f"SIC_{sea}"],
            "POC": monthly[f"POC_{sea}"],
            "PIC": monthly[f"PIC_{sea}"],
            "CHL": monthly[f"CHL_{sea}"],
        }).sort_values("time")

        indexed = temp.set_index("time")
        events = build_season_year_events(temp, sea)

        if events.empty:
            retreat_day = bloom_day = lag_days = np.nan
            chl_response = poc_response = np.nan
            n_events = 0
        else:
            retreat_day = month_to_day(
                circular_mean_month(events["retreat_month"].values)
            )
            bloom_day = month_to_day(
                circular_mean_month(events["bloom_month"].values)
            )
            lag_days = events["lag_days"].mean()
            chl_response = events["chl_response"].mean()
            poc_response = events["poc_response"].mean()
            n_events = len(events)

        rows.append({
            "sea": sea,
            "sic_active_mean_percent": active_mean(indexed["SIC"]),
            "poc_active_mean": active_mean(indexed["POC"]),
            "pic_active_mean_mg_C_m3": active_mean(indexed["PIC"]),
            "retreat_day_mean": retreat_day,
            "bloom_day_mean": bloom_day,
            "lag_days_mean": lag_days,
            "chl_response_mean": chl_response,
            "poc_response_mean": poc_response,
            "n_events": n_events,
        })

    return pd.DataFrame(rows).set_index("sea").loc[SEA_CODES].reset_index()


def process_carbon_biology_seawise():
    check_files(["SIC", "PIC", "POC", "CHL"])

    sic = extract_seawise_monthly(FILES["SIC"], "SIC")
    pic = extract_seawise_monthly(FILES["PIC"], "PIC")
    poc = extract_seawise_monthly(FILES["POC"], "POC")
    chl = extract_seawise_monthly(FILES["CHL"], "CHL")

    monthly = merge_monthly_tables([sic, poc, pic, chl])

    cols = ["time"]
    for var in ["SIC", "POC", "PIC", "CHL"]:
        cols += [f"{var}_{sea}" for sea in SEA_CODES]
    monthly = monthly[cols]

    sic_out = pd.DataFrame({"time": monthly["time"]})
    for sea in SEA_CODES:
        sic_out[sea] = monthly[f"SIC_{sea}"]

    save_csv(sic_out, SIC_SEAWISE_FILE)
    save_csv(monthly, FIG22_MONTHLY_FILE)
    save_csv(create_fig22_metrics(monthly), FIG22_METRICS_FILE)

    fig23_cols = ["time"]
    for var in ["SIC", "POC", "CHL"]:
        fig23_cols += [f"{var}_{sea}" for sea in SEA_CODES]

    save_csv(monthly[fig23_cols].copy(), FIG23_MONTHLY_FILE)


# =============================================================================
# FILES: Fig23 CSV + MLD + STRAT
# OUTPUTS: Fig24 BCP monthly + active-season summary
# =============================================================================

def nprof_weighted_strat_mean(strat, nprof, lon_min, lon_max):
    mask = lon_mask(strat["lon"], lon_min, lon_max)
    strat_sub = strat.where(mask, drop=True)
    nprof_sub = nprof.where(mask, drop=True)

    strat_sub = strat_sub.where(nprof_sub > 0)
    area_w = np.cos(np.deg2rad(strat_sub["lat"])).broadcast_like(strat_sub)
    total_w = nprof_sub.astype("float32") * area_w

    numerator = (strat_sub * total_w).sum(("lat", "lon"), skipna=True)
    denominator = total_w.where(np.isfinite(strat_sub)).sum(
        ("lat", "lon"), skipna=True
    )

    return (numerator / denominator).where(denominator > 0)


def sea_nprof_total(nprof, lon_min, lon_max):
    return nprof.where(
        lon_mask(nprof["lon"], lon_min, lon_max), drop=True
    ).sum(("lat", "lon"), skipna=True)


def extract_strat_monthly():
    check_file(FILES["STRAT"], "STRAT")

    ds = prepare_antarctic_dataset(open_dataset_safe(FILES["STRAT"]))
    varname = find_var(ds, "STRAT")
    strat = make_monthly_if_needed(transform_variable(ds[varname], "STRAT"))

    nprof = None
    if "nprof" in ds.data_vars:
        nprof = make_monthly_if_needed(ds["nprof"].astype("float32"))
        print("Using nprof-weighted STRAT.")
    else:
        print("WARNING: nprof not found; using area-weighted STRAT.")

    out = pd.DataFrame({"time": full_monthly_range()})

    for sea, _, _, lon_min, lon_max in SEA_INFO:
        if nprof is not None:
            ts = nprof_weighted_strat_mean(strat, nprof, lon_min, lon_max)
            nt = sea_nprof_total(nprof, lon_min, lon_max)

            with ProgressBar():
                loaded = ts.compute()
                nloaded = nt.compute()

            temp = pd.DataFrame({
                "time": month_start_datetime(loaded["time"].values),
                f"STRAT_{sea}": loaded.values.astype("float32"),
                f"STRAT_NPROF_{sea}": nloaded.values.astype("float32"),
            })
        else:
            ts = area_weighted_sea_mean(strat, lon_min, lon_max)
            with ProgressBar():
                loaded = ts.compute()

            temp = pd.DataFrame({
                "time": month_start_datetime(loaded["time"].values),
                f"STRAT_{sea}": loaded.values.astype("float32"),
            })

        out = out.merge(temp, on="time", how="left")

    ds.close()
    return out


def process_bcp_mld_strat():
    check_file(FIG23_MONTHLY_FILE, "Fig23 monthly CSV")
    check_files(["MLD", "STRAT"])

    base = normalize_time_column(pd.read_csv(FIG23_MONTHLY_FILE))
    keep = ["time"] + [
        f"{var}_{sea}"
        for var in ["POC", "CHL"]
        for sea in SEA_CODES
    ]
    base = base[keep]

    mld = extract_seawise_monthly(FILES["MLD"], "MLD")
    strat = extract_strat_monthly()

    wide = merge_monthly_tables([base, mld, strat])

    ordered = ["time"]
    for var in ["POC", "CHL", "MLD", "STRAT", "STRAT_NPROF"]:
        for sea in SEA_CODES:
            col = f"{var}_{sea}"
            if col in wide.columns:
                ordered.append(col)

    wide = wide[ordered]
    save_csv(wide, FIG24_MONTHLY_FILE)

    active = wide.copy()
    active["month"] = pd.to_datetime(active["time"]).dt.month
    active = active[active["month"].isin(ACTIVE_MONTHS)]

    rows = []

    for sea in SEA_CODES:
        row = {"sea": sea}

        for var in ["POC", "CHL", "MLD", "STRAT"]:
            col = f"{var}_{sea}"
            vals = (
                active[col].replace([np.inf, -np.inf], np.nan).dropna()
                if col in active.columns
                else pd.Series(dtype=float)
            )

            prefix = var.lower()
            row[f"{prefix}_active_mean"] = vals.mean()
            row[f"{prefix}_active_median"] = vals.median()
            row[f"{prefix}_active_std"] = vals.std()
            row[f"{prefix}_n"] = len(vals)

        ncol = f"STRAT_NPROF_{sea}"
        if ncol in active.columns:
            nvals = active[ncol].replace([np.inf, -np.inf], np.nan).dropna()
            row["strat_total_nprof_active"] = nvals.sum()
            row["strat_mean_nprof_per_month"] = nvals.mean()
            row["strat_months_with_profiles"] = int((nvals > 0).sum())
        else:
            row["strat_total_nprof_active"] = np.nan
            row["strat_mean_nprof_per_month"] = np.nan
            row["strat_months_with_profiles"] = 0

        def pair_n(xvar, yvar):
            xcol, ycol = f"{xvar}_{sea}", f"{yvar}_{sea}"
            if xcol not in active.columns or ycol not in active.columns:
                return 0
            return len(
                active[[xcol, ycol]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

        row["n_poc_mld"] = pair_n("POC", "MLD")
        row["n_chl_mld"] = pair_n("CHL", "MLD")
        row["n_poc_strat"] = pair_n("POC", "STRAT")
        row["n_chl_strat"] = pair_n("CHL", "STRAT")

        rows.append(row)

    save_csv(pd.DataFrame(rows), FIG24_SUMMARY_FILE)


# FILES: SIC CSV + Fig24 CSV + NPPINT NetCDF
# OUTPUTS: Fig25 monthly, lag, bootstrap and summary CSVs

def lag_corr(x, y, lag):
    pair = pd.DataFrame({
        "x": pd.Series(x).astype(float),
        "y": pd.Series(y).astype(float).shift(-lag),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(pair) < 8 or pair["x"].std() == 0 or pair["y"].std() == 0:
        return np.nan

    return pair["x"].corr(pair["y"])


def best_lag_from_subset(data, sea, var, months, years_sample=None):
    temp = data.copy()

    if years_sample is not None:
        temp = pd.concat(
            [temp[temp["year"] == y] for y in years_sample],
            ignore_index=True,
        )

    temp = temp[temp["month"].isin(months)]
    if len(temp) < 8:
        return np.nan

    r = np.array([
        lag_corr(
            temp[f"ICE_DRIVER_{sea}"],
            temp[f"{var}_{sea}_anom"],
            lag,
        )
        for lag in LAGS
    ], dtype=float)

    if np.all(~np.isfinite(r)):
        return np.nan

    return int(LAGS[np.nanargmax(r)])


def process_lag_analysis():
    check_file(SIC_SEAWISE_FILE, "SIC sea-wise CSV")
    check_file(FIG24_MONTHLY_FILE, "Fig24 monthly CSV")
    check_file(NPPINT_FILE, "NPPINT NetCDF")

    sic = normalize_time_column(pd.read_csv(SIC_SEAWISE_FILE))
    sic = sic.rename(columns={sea: f"SIC_{sea}" for sea in SEA_CODES})
    sic = sic[["time"] + [f"SIC_{sea}" for sea in SEA_CODES]]

    bio = normalize_time_column(pd.read_csv(FIG24_MONTHLY_FILE))
    bio_cols = [
        f"{var}_{sea}"
        for var in ["POC", "CHL"]
        for sea in SEA_CODES
    ]
    bio = bio[["time"] + bio_cols]

    npp = extract_seawise_monthly(NPPINT_FILE, "NPPINT")
    npp = npp.rename(columns={
        f"NPPINT_{sea}": f"NPPint_{sea}" for sea in SEA_CODES
    })

    full = merge_monthly_tables([sic, bio, npp])
    save_csv(full, FIG25_MONTHLY_FILE)

    dfm = normalize_time_column(full)
    dfm["month"] = dfm["time"].dt.month
    dfm["year"] = dfm["time"].dt.year

    for var in ["SIC", "POC", "CHL", "NPPint"]:
        for sea in SEA_CODES:
            col = f"{var}_{sea}"
            clim = dfm.groupby("month")[col].transform("mean")
            dfm[f"{col}_anom"] = dfm[col] - clim

    for sea in SEA_CODES:
        sic_anom = f"SIC_{sea}_anom"
        dfm[f"ICE_DRIVER_{sea}"] = (
            -dfm[sic_anom]
            if ICE_DRIVER_MODE == "retreat_anomaly"
            else dfm[sic_anom]
        )

    variables = ["POC", "CHL", "NPPint"]

    lag_rows = []
    for var in variables:
        for sea in SEA_CODES:
            for lag in LAGS:
                lag_rows.append({
                    "variable": var,
                    "sea": sea,
                    "lag": int(lag),
                    "r": lag_corr(
                        dfm[f"ICE_DRIVER_{sea}"],
                        dfm[f"{var}_{sea}_anom"],
                        lag,
                    ),
                })

    lag_seawise = pd.DataFrame(lag_rows)
    save_csv(lag_seawise, FIG25_LAG_SEAWISE_FILE)

    mean_rows = []
    for var in variables:
        for lag in LAGS:
            vals = lag_seawise[
                (lag_seawise["variable"] == var)
                & (lag_seawise["lag"] == lag)
            ]["r"].dropna()

            n = len(vals)
            std = vals.std(ddof=1)
            se = std / np.sqrt(n) if n > 1 else np.nan

            mean_rows.append({
                "variable": var,
                "lag": int(lag),
                "mean_r": vals.mean(),
                "std_r": std,
                "se_r": se,
                "ci95": 1.96 * se if n > 1 else np.nan,
                "n_seas": n,
            })

    lag_mean = pd.DataFrame(mean_rows)

    best_rows = []
    for var in variables:
        sub = lag_mean[lag_mean["variable"] == var]
        if sub["mean_r"].notna().any():
            idx = sub["mean_r"].idxmax()
            best_rows.append({
                "variable": var,
                "best_lag_mean_curve": lag_mean.loc[idx, "lag"],
                "best_mean_r": lag_mean.loc[idx, "mean_r"],
            })
        else:
            best_rows.append({
                "variable": var,
                "best_lag_mean_curve": np.nan,
                "best_mean_r": np.nan,
            })

    lag_mean = lag_mean.merge(pd.DataFrame(best_rows), on="variable", how="left")
    save_csv(lag_mean, FIG25_LAG_MEAN_FILE)

    rng = np.random.default_rng(RANDOM_SEED)
    years = sorted(dfm["year"].dropna().unique())

    boot_rows = []
    for season, months in SEASONS.items():
        for sea in SEA_CODES:
            for var in variables:
                original = best_lag_from_subset(dfm, sea, var, months)

                for b in range(N_BOOT):
                    sample_years = rng.choice(
                        years, size=len(years), replace=True
                    )
                    boot_rows.append({
                        "season": season,
                        "sea": sea,
                        "variable": var,
                        "bootstrap": b,
                        "best_lag": best_lag_from_subset(
                            dfm, sea, var, months, sample_years
                        ),
                        "original_best_lag": original,
                    })

    boot = pd.DataFrame(boot_rows)
    save_csv(boot, FIG25_BOOT_FILE)

    rows = []
    for season in SEASONS:
        for sea in SEA_CODES:
            for var in variables:
                sub = boot[
                    (boot["season"] == season)
                    & (boot["sea"] == sea)
                    & (boot["variable"] == var)
                ]
                vals = sub["best_lag"].dropna()
                original = sub["original_best_lag"].dropna()

                rows.append({
                    "season": season,
                    "sea": sea,
                    "variable": var,
                    "original_best_lag": (
                        original.iloc[0] if len(original) else np.nan
                    ),
                    "median_boot_lag": vals.median() if len(vals) else np.nan,
                    "mean_boot_lag": vals.mean() if len(vals) else np.nan,
                    "q05_boot_lag": vals.quantile(0.05) if len(vals) else np.nan,
                    "q25_boot_lag": vals.quantile(0.25) if len(vals) else np.nan,
                    "q75_boot_lag": vals.quantile(0.75) if len(vals) else np.nan,
                    "q95_boot_lag": vals.quantile(0.95) if len(vals) else np.nan,
                    "n_boot_valid": len(vals),
                })

    save_csv(pd.DataFrame(rows), FIG25_SUMMARY_FILE)


# INTEGRATED SEA CLASSIFICATION INPUT — GBIF EXCLUDED
# OUTPUT: sea_classification_input_2008_2025_NO_GBIF.csv

def sea_longterm_means(path, label, output_col):
    ds = prepare_antarctic_dataset(open_dataset_safe(path))
    varname = find_var(ds, label)
    da = make_monthly_if_needed(transform_variable(ds[varname], label))

    rows = []

    for sea, name, lon_text, lon_min, lon_max in SEA_INFO:
        ts = area_weighted_sea_mean(da, lon_min, lon_max)
        value = float(ts.mean("time", skipna=True).compute().item())

        rows.append({
            "Sea": sea,
            "Sea_name": name,
            "Longitude_range": lon_text,
            output_col: value,
        })

    ds.close()
    return pd.DataFrame(rows)


def calculate_retreat_timing_percent(series, threshold=15.0):
    series = series.dropna().sort_index()
    values_out = []

    if series.empty:
        return np.nan, np.nan, 0

    for year in np.arange(series.index.year.min(), series.index.year.max()):
        dates = pd.date_range(f"{year}-09-01", f"{year+1}-04-01", freq="MS")
        vals = np.array([
            series.loc[d] if d in series.index else np.nan
            for d in dates
        ], dtype=float)

        if np.all(~np.isfinite(vals)):
            continue

        below = np.where((vals < threshold) & np.isfinite(vals))[0]

        if len(below):
            idx = int(below[0])
        else:
            diff = np.diff(vals)
            if np.all(~np.isfinite(diff)):
                continue
            idx = int(np.nanargmin(diff)) + 1

        values_out.append(idx)

    if not values_out:
        return np.nan, np.nan, 0

    mean_since_sep = float(np.mean(values_out))
    return mean_since_sep, mean_since_sep + 9.0, len(values_out)


def sic_classification_metrics():
    df = normalize_time_column(pd.read_csv(SIC_SEAWISE_FILE))
    rows = []

    for sea, name, lon_text, _, _ in SEA_INFO:
        ts = pd.Series(
            df[sea].to_numpy(dtype=float),
            index=pd.to_datetime(df["time"]),
        ).replace([np.inf, -np.inf], np.nan)

        if np.nanmax(ts.values) <= 1.5:
            ts *= 100.0

        valid = np.isfinite(ts.values)

        if valid.sum() >= 24:
            x = (ts.index - ts.index[0]).days.values / 365.25
            trend_fraction = np.polyfit(x[valid], ts.values[valid], 1)[0] / 100.0
        else:
            trend_fraction = np.nan

        retreat, calendar, n_years = calculate_retreat_timing_percent(ts)

        rows.append({
            "Sea": sea,
            "Sea_name": name,
            "Longitude_range": lon_text,
            "SIC_mean": np.nanmean(ts.values) / 100.0,
            "SIC_trend_per_year": trend_fraction,
            "SIC_retreat_month_since_sep": retreat,
            "SIC_retreat_calendar_month": calendar,
            "SIC_retreat_valid_years": n_years,
        })

    return pd.DataFrame(rows)


def merge_on_sea(base, new):
    vals = [
        c for c in new.columns
        if c not in ["Sea", "Sea_name", "Longitude_range"]
    ]
    return base.merge(new[["Sea"] + vals], on="Sea", how="left")


def create_classification_no_gbif():
    check_files([
        "SIC", "SST", "MLD", "SIT", "STRAT", "FWC",
        "CHL", "PIC", "POC", "DIATO", "HAPTO", "PICO",
    ])
    check_file(NPPINT_FILE, "NPPINT")
    check_file(SIC_SEAWISE_FILE, "SIC sea-wise CSV")

    df = sic_classification_metrics()

    jobs = [
        ("SST", FILES["SST"], "SST_mean_C"),
        ("MLD", FILES["MLD"], "MLD_mean"),
        ("SIT", FILES["SIT"], "SIT_mean"),
        ("STRAT", FILES["STRAT"], "Stratification_mean"),
        ("FWC", FILES["FWC"], "FWC_mean"),
        ("CHL", FILES["CHL"], "CHL_mean"),
        ("POC", FILES["POC"], "POC_mean"),
        ("PIC", FILES["PIC"], "PIC_mean_mg_C_m3"),
        ("NPPINT", NPPINT_FILE, "NPPint_mean"),
        ("DIATO", FILES["DIATO"], "DIATO_mean"),
        ("HAPTO", FILES["HAPTO"], "HAPTO_mean"),
        ("PICO", FILES["PICO"], "PICO_mean"),
    ]

    for label, path, out_col in jobs:
        df = merge_on_sea(
            df,
            sea_longterm_means(path, label, out_col),
        )

    total_pft = df["DIATO_mean"] + df["HAPTO_mean"] + df["PICO_mean"]

    df["Diatom_contribution_percent"] = np.where(
        total_pft > 0, df["DIATO_mean"] / total_pft * 100.0, np.nan
    )
    df["Haptophyte_contribution_percent"] = np.where(
        total_pft > 0, df["HAPTO_mean"] / total_pft * 100.0, np.nan
    )
    df["Pico_contribution_percent"] = np.where(
        total_pft > 0, df["PICO_mean"] / total_pft * 100.0, np.nan
    )

    df["Notes"] = (
        "NON-GBIF integrated classification input. "
        "No occurrence, richness, Shannon, Simpson or Pielou variables included."
    )

    ordered = [
        "Sea", "Sea_name", "Longitude_range",
        "SIC_mean", "SIC_trend_per_year",
        "SIC_retreat_month_since_sep", "SIC_retreat_calendar_month",
        "SIC_retreat_valid_years",
        "SST_mean_C", "MLD_mean", "SIT_mean",
        "Stratification_mean", "FWC_mean",
        "CHL_mean", "POC_mean", "PIC_mean_mg_C_m3", "NPPint_mean",
        "DIATO_mean", "HAPTO_mean", "PICO_mean",
        "Diatom_contribution_percent",
        "Haptophyte_contribution_percent",
        "Pico_contribution_percent",
        "Notes",
    ]

    save_csv(df[[c for c in ordered if c in df.columns]], CLASSIFICATION_FILE)


# FINAL REPORT + MAIN

def final_output_report():
    outputs = [
        NPPINT_FILE,
        SIC_SEAWISE_FILE,
        PFT_MEAN_FILE,
        PFT_REL_FILE,
        FIG22_MONTHLY_FILE,
        FIG22_METRICS_FILE,
        FIG23_MONTHLY_FILE,
        FIG24_MONTHLY_FILE,
        FIG24_SUMMARY_FILE,
        FIG25_MONTHLY_FILE,
        FIG25_LAG_SEAWISE_FILE,
        FIG25_LAG_MEAN_FILE,
        FIG25_BOOT_FILE,
        FIG25_SUMMARY_FILE,
        CLASSIFICATION_FILE,
    ]

    print("\n" + "=" * 80)
    print("FINAL NON-GBIF CARBON/BIOLOGICAL OUTPUT REPORT")
    print("=" * 80)

    for path in outputs:
        print(
            f"{'CREATED' if Path(path).exists() else 'NOT CREATED':11s} | {path}"
        )


def main():
    print("\n" + "=" * 80)
    print("ANTARCTIC CARBON + BIOLOGICAL DATA PIPELINE")
    print("GBIF / BIODIVERSITY DATA: EXCLUDED")
    print("=" * 80)

    if RUN_CREATE_NPPINT:
        create_nppint_file()

    if RUN_PFT_SEASONAL_TABLES:
        process_pft_seasonal_tables()

    if RUN_CARBON_BIOLOGY_SEAWISE:
        process_carbon_biology_seawise()

    if RUN_BCP_MLD_STRAT:
        process_bcp_mld_strat()

    if RUN_LAG_ANALYSIS:
        process_lag_analysis()

    if RUN_CLASSIFICATION_NO_GBIF:
        create_classification_no_gbif()

    final_output_report()
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
