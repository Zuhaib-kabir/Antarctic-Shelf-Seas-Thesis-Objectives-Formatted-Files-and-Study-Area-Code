
"""
SOUTHERN OCEAN / SEA-LEVEL ANALYSIS — MASTER DATA-PRODUCT GENERATION SCRIPT

PURPOSE:

Reorganized from sla_data_thesis.py and so_sealevel_paper_fig.py.

This master file keeps analysis-ready .nc, .csv and .xlsx producers used
by main figures, supplementary figures and supplementary tables.

EXCLUDED:
• DUACS/SLA downloading.
• EN4 raw-data downloading/extraction.
• raw yearly acquisition/assembly code.
• NetCDF inventory-report creation.
• PNG/PDF rendering and savefig() code.
• plot-only redraw workflows.
• superseded repeated producer versions where a later corrected version exists.

ORGANIZATION:
Google Drive is mounted once. Every output family begins with '# File name:'
comments and is followed by the package installation, imports, inputs,
scientific settings, calculations, validation and file-writing code needed
for that product family.

Multiple outputs produced by one shared calculation are grouped together to
avoid repeating the same expensive computation.

SCIENTIFIC PRESERVATION:
Source years, masks, sectors, thresholds, reference-state definitions,
statistical methods, FDR/bootstrap procedures, filenames and terminology are
preserved. The only direct source repair restores accidentally commented
longitude modulo operations: ((lon + 180) % 360) - 180.

"""

# GOOGLE DRIVE MOUNT — RUN ONCE

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception:
    print("Google Drive mounting skipped because this is not Google Colab.")


# DATA PRODUCT GROUP 01: CORE EN4-DERIVED SOUTHERN OCEAN MONTHLY PRODUCTS — LATEST CORRECTED
# Source: sla_data_thesis.py
# File name: EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc
# File name: EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc
# File name: EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc
# File name: EN4_freshwater_content_0_200m_monthly_2008_2025_SO.nc
# File name: EN4_surface_salinity_0_30m_monthly_2008_2025_SO.nc
# File name: EN4_surface_salinity_anomaly_0_30m_monthly_2008_2025_SO.nc
# File name: EN4_stratification_20_200m_monthly_2008_2025_SO.nc
# File name: EN4_OHC_0_1000m_monthly_2008_2025_SO.nc
# File name: EN4_derived_products_processing_summary.csv

# CREATE CORRECTED EN4-DERIVED MONTHLY SOUTHERN OCEAN PRODUCTS, 2008–2025
# Outputs:
# 1) EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc
# 2) EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc
# 3) EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc
# 4) EN4_freshwater_content_0_200m_monthly_2008_2025_SO.nc
# 5) EN4_surface_salinity_0_30m_monthly_2008_2025_SO.nc
# 6) EN4_surface_salinity_anomaly_0_30m_monthly_2008_2025_SO.nc
# 7) EN4_stratification_20_200m_monthly_2008_2025_SO.nc
# 8) EN4_OHC_0_1000m_monthly_2008_2025_SO.nc
#
# Scientific rules:
# - One fixed grid-cell all-month reference profile (2008–2025) is
#   used for every monthly steric calculation.
# - No calendar-month climatology is removed during file production.
# - 0–1000 m products require GEBCO depth >=1000 m and complete
#   monthly T/S coverage through the full 0–1000 m layer.
# - 0–200 m products require GEBCO depth >=200 m and complete
#   layer coverage through 200 m.
# - Shallow cells are excluded, never integrated only to local seabed.
# - EN4 temperature is treated as potential temperature in kelvin.
# - TEOS-10 SA, CT, specific volume, sigma0, and density are used.

# 0. Install packages in Colab

import sys, subprocess, importlib.util

required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "dask": "dask[array]",
    "scipy": "scipy",
    "gsw": "gsw",
}
missing = [pip for module, pip in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

# 1. Imports
import os
import gc
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gsw
from scipy.interpolate import interp1d
from netCDF4 import Dataset, date2num

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

# 2. Paths
EN4_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/EN4/final_monthly/"
    "EN4.2.2_g10_Antarctic_monthly_2008_2025.nc"
)

GEBCO_CANDIDATES = [
    "/content/drive/MyDrive/SAM_Thesis/Data/GEBCO_2024_CF.nc",
    "/content/drive/MyDrive/SAM_Thesis/Data/GEBCO_2024_CEC.nc",
]
GEBCO_FILE = next((p for p in GEBCO_CANDIDATES if os.path.exists(p)), GEBCO_CANDIDATES[0])

OUT_DIR = Path(
    "/content/drive/MyDrive/SAM_Thesis/Processed/EN4_NetCDF_inventory"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = {
    "total": OUT_DIR / "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc",
    "thermo": OUT_DIR / "EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc",
    "halo": OUT_DIR / "EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc",
    "fwc": OUT_DIR / "EN4_freshwater_content_0_200m_monthly_2008_2025_SO.nc",
    "sss": OUT_DIR / "EN4_surface_salinity_0_30m_monthly_2008_2025_SO.nc",
    "sss_anom": OUT_DIR / "EN4_surface_salinity_anomaly_0_30m_monthly_2008_2025_SO.nc",
    "strat": OUT_DIR / "EN4_stratification_20_200m_monthly_2008_2025_SO.nc",
    "ohc": OUT_DIR / "EN4_OHC_0_1000m_monthly_2008_2025_SO.nc",
}
SUMMARY_CSV = OUT_DIR / "EN4_derived_products_processing_summary.csv"
VERIFY_TXT = OUT_DIR / "EN4_derived_products_verification.txt"

# 3. Scientific settings
YEAR_START = 2008
YEAR_END = 2025
MAX_STERIC_DEPTH = 1000.0
MAX_FWC_DEPTH = 200.0
MAX_SSS_DEPTH = 30.0
STRAT_TOP = 20.0
STRAT_BOTTOM = 200.0

# Editable assumptions:
FWC_REFERENCE_SALINITY = 34.8
OHC_REFERENCE_CT_C = 0.0
CP0 = 3991.86795711963  # TEOS-10 J kg-1 K-1
K_TO_C = 273.15

COMPLEVEL = 4
FILL_FLOAT = np.float32(9.96921e36)
FILL_MASK = np.int8(-127)
TEST_MONTHS = None  # e.g., 2 for testing; None for all 216 months

if not os.path.exists(EN4_FILE):
    raise FileNotFoundError(EN4_FILE)
if not os.path.exists(GEBCO_FILE):
    raise FileNotFoundError("GEBCO file not found: " + " | ".join(GEBCO_CANDIDATES))

# 4. Helper functions
def standardize_coords(ds):
    rename = {}
    options = {
        "time": ["valid_time", "date"],
        "depth": ["lev", "level", "pressure"],
        "lat": ["latitude", "nav_lat", "y"],
        "lon": ["longitude", "nav_lon", "x"],
    }
    for target, names in options.items():
        if target in ds.coords or target in ds.dims:
            continue
        for name in names:
            if name in ds.coords or name in ds.dims:
                rename[name] = target
                break
    return ds.rename(rename) if rename else ds


def normalize_lon(ds):
    if "lon" not in ds.coords or ds["lon"].ndim != 1:
        return ds
    lon = ((ds["lon"].astype(float) + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords(lon=lon)
    _, idx = np.unique(ds["lon"].values, return_index=True)
    return ds.isel(lon=np.sort(idx)).sortby("lon")


def find_var(ds, names):
    for name in names:
        if name in ds.data_vars:
            return name
    for name in names:
        for var in ds.data_vars:
            if name.lower() in var.lower():
                return var
    raise KeyError(f"Could not find {names}; available={list(ds.data_vars)}")


def target_depth_grid(native_depth):
    native_depth = np.asarray(native_depth, dtype=float)
    native_inside = native_depth[(native_depth > 0) & (native_depth < MAX_STERIC_DEPTH)]
    mandatory = np.array([0.0, STRAT_TOP, MAX_SSS_DEPTH, MAX_FWC_DEPTH, MAX_STERIC_DEPTH])
    depth = np.unique(np.round(np.concatenate([native_inside, mandatory]), 6))
    depth.sort()
    return depth.astype(float)


def interpolate_depth(values, native_depth, target_depth):
    values = np.asarray(values, dtype=float)
    f = interp1d(
        np.asarray(native_depth, dtype=float),
        values,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    out = f(target_depth)
    out[np.argmin(np.abs(target_depth - 0.0))] = values[0]  # 0 m represented by ~5 m EN4 level
    return out


def trapz(values, x, axis=0):
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fn(values, x=x, axis=axis)


def integrate_depth(values, depth):
    return trapz(values, np.asarray(depth, dtype=float), axis=0)


def integrate_steric(delta_specvol, pressure_dbar, gravity):
    # 1 dbar = 1e4 Pa; result is metres.
    return trapz(delta_specvol * 1.0e4 / gravity, pressure_dbar, axis=0)


def layer_mean(values, depth, thickness):
    return integrate_depth(values, depth) / float(thickness)


def depth_index(depth, value):
    idx = int(np.argmin(np.abs(depth - value)))
    if not np.isclose(depth[idx], value, atol=1e-6):
        raise ValueError(f"Depth {value} m missing from target grid")
    return idx


def create_file(path, var_name, long_name, units, valid_description, dates, lat, lon, water_depth, attrs):
    if path.exists():
        path.unlink()
    root = Dataset(path, "w", format="NETCDF4")
    root.createDimension("time", len(dates))
    root.createDimension("latitude", len(lat))
    root.createDimension("longitude", len(lon))

    t = root.createVariable("time", "f8", ("time",))
    y = root.createVariable("latitude", "f4", ("latitude",))
    x = root.createVariable("longitude", "f4", ("longitude",))
    time_units = "days since 1970-01-01 00:00:00"
    calendar = "proleptic_gregorian"
    t[:] = date2num(dates, time_units, calendar)
    t.units = time_units
    t.calendar = calendar
    t.standard_name = "time"
    y[:] = lat.astype(np.float32)
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    x[:] = lon.astype(np.float32)
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"

    chunks3 = (1, min(len(lat), 24), min(len(lon), 120))
    chunks2 = (min(len(lat), 24), min(len(lon), 120))
    science = root.createVariable(
        var_name, "f4", ("time", "latitude", "longitude"),
        zlib=True, complevel=COMPLEVEL, shuffle=True,
        chunksizes=chunks3, fill_value=FILL_FLOAT,
    )
    science.long_name = long_name
    science.units = units
    science.coordinates = "time latitude longitude"

    mask = root.createVariable(
        "valid_layer_mask", "i1", ("time", "latitude", "longitude"),
        zlib=True, complevel=COMPLEVEL, shuffle=True,
        chunksizes=chunks3, fill_value=FILL_MASK,
    )
    mask.long_name = "monthly valid-layer mask"
    mask.flag_values = np.array([0, 1], dtype=np.int8)
    mask.flag_meanings = "invalid valid"
    mask.description = valid_description

    bathy = root.createVariable(
        "gebco_water_depth", "f4", ("latitude", "longitude"),
        zlib=True, complevel=COMPLEVEL, shuffle=True,
        chunksizes=chunks2, fill_value=FILL_FLOAT,
    )
    bathy[:] = np.ma.masked_invalid(water_depth.astype(np.float32))
    bathy.long_name = "GEBCO ocean water depth"
    bathy.units = "m"
    bathy.positive = "down"

    root.Conventions = "CF-1.8"
    root.title = long_name
    root.source_en4_file = EN4_FILE
    root.source_gebco_file = GEBCO_FILE
    root.period = f"{YEAR_START}-01 through {YEAR_END}-12"
    root.region = "Southern Ocean EN4 Antarctic domain"
    root.processing_software = "Python, NumPy, Xarray, SciPy, netCDF4, GSW-Python"
    root.processing_time_utc = datetime.now(timezone.utc).isoformat()
    root.depth_interpolation = (
        "Linear between EN4 levels; exact 20, 30, 200 and 1000 m boundaries; "
        "0 m assigned the shallowest EN4 level (~5 m)."
    )
    for key, value in attrs.items():
        root.setncattr(key, value)
    return {"root": root, "science": science, "mask": mask}


def write_month(handle, i, values, valid):
    handle["science"][i] = np.ma.masked_invalid(np.where(valid, values, np.nan).astype(np.float32))
    handle["mask"][i] = valid.astype(np.int8)

# 5. Open EN4 and GEBCO

ds = xr.open_dataset(
    EN4_FILE,
    decode_times=True,
    mask_and_scale=True,
    chunks="auto",
)
ds = normalize_lon(standardize_coords(ds))

t_name = find_var(ds, ["temperature", "potential_temperature", "theta"])
s_name = find_var(ds, ["salinity", "practical_salinity"])
temp = ds[t_name].sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
sal = ds[s_name].sel(time=temp.time)
if TEST_MONTHS is not None:
    temp = temp.isel(time=slice(0, TEST_MONTHS))
    sal = sal.isel(time=slice(0, TEST_MONTHS))

native_depth = ds.depth.values.astype(float)
lat = ds.lat.values.astype(float)
lon = ds.lon.values.astype(float)
dates = pd.to_datetime(temp.time.values).to_pydatetime()
nt, ny, nx = len(dates), len(lat), len(lon)

analysis_depth = target_depth_grid(native_depth)
mask30 = analysis_depth <= MAX_SSS_DEPTH + 1e-6
mask200 = analysis_depth <= MAX_FWC_DEPTH + 1e-6
mask1000 = analysis_depth <= MAX_STERIC_DEPTH + 1e-6
depth30 = analysis_depth[mask30]
depth200 = analysis_depth[mask200]
depth1000 = analysis_depth[mask1000]
i20 = depth_index(analysis_depth, STRAT_TOP)
i200 = depth_index(analysis_depth, STRAT_BOTTOM)

print(f"EN4 months={nt}, lat={ny}, lon={nx}, native depths={len(native_depth)}")
print(f"Analysis depth levels through 1000 m={len(depth1000)}")

# GEBCO -> EN4 grid
bg = xr.open_dataset(GEBCO_FILE, decode_times=False, mask_and_scale=True, chunks="auto")
bg = normalize_lon(standardize_coords(bg))
bg_name = find_var(bg, ["elevation", "bathymetry", "depth", "z"])
elev = bg[bg_name].interp(
    lat=xr.DataArray(lat, dims="lat", coords={"lat": lat}),
    lon=xr.DataArray(lon, dims="lon", coords={"lon": lon}),
    method="nearest",
).compute().values.astype(float)
water_depth = np.where(elev < 0, -elev, np.nan)
ocean = np.isfinite(water_depth)
bathy30 = ocean & (water_depth >= 30.0)
bathy200 = ocean & (water_depth >= 200.0)
bathy1000 = ocean & (water_depth >= 1000.0)

# 6. Fixed 2008–2025 reference profiles
print("Calculating fixed all-month grid-cell reference profiles...")
t_ref_native = temp.mean("time", skipna=True).compute().values.astype(float)
s_ref_native = sal.mean("time", skipna=True).compute().values.astype(float)
t_ref_k = interpolate_depth(t_ref_native, native_depth, analysis_depth)
s_ref = interpolate_depth(s_ref_native, native_depth, analysis_depth)
pt_ref_c = t_ref_k - K_TO_C

ref_complete1000 = (
    np.isfinite(pt_ref_c[mask1000]).all(axis=0)
    & np.isfinite(s_ref[mask1000]).all(axis=0)
)
ref_s_complete30 = np.isfinite(s_ref[mask30]).all(axis=0)

# TEOS-10 pressure/reference arrays
p_depth_lat = gsw.p_from_z(-analysis_depth[:, None], lat[None, :])
p3 = np.broadcast_to(p_depth_lat[:, :, None], (len(analysis_depth), ny, nx))
lat3 = lat[None, :, None]
lon3 = lon[None, None, :]
g3 = gsw.grav(lat3, p3)
SA_ref = gsw.SA_from_SP(s_ref, p3, lon3, lat3)
CT_ref = gsw.CT_from_pt(SA_ref, pt_ref_c)
nu_ref = gsw.specvol(SA_ref, CT_ref, p3)
sss_ref = layer_mean(s_ref[mask30], depth30, MAX_SSS_DEPTH)
sss_ref_valid = bathy30 & ref_s_complete30 & np.isfinite(sss_ref)

# 7. Create output files
fixed_ref_attrs = {
    "reference_state": (
        "One fixed grid-cell, depth-dependent all-month mean potential-temperature "
        "and practical-salinity profile for 2008-2025."
    ),
    "seasonal_cycle_preserved": "Yes; calendar-month climatology was not removed.",
    "temperature_interpretation": "EN4 potential temperature in kelvin.",
    "salinity_interpretation": "EN4 Practical Salinity.",
    "teos10_conversion": "SA_from_SP and CT_from_pt.",
    "depth_mask_rule": (
        "GEBCO depth >=1000 m and complete monthly T/S through 0-1000 m; "
        "shallow cells are excluded, never integrated to local seabed."
    ),
}

handles = {}
handles["total"] = create_file(
    OUT["total"], "total_steric_height",
    "EN4 total steric height anomaly integrated over 0-1000 m", "m",
    "1 for GEBCO depth >=1000 m with complete monthly and reference T/S through 1000 m.",
    dates, lat, lon, water_depth,
    {**fixed_ref_attrs, "calculation": "Integral of monthly minus fixed-reference specific volume over pressure divided by local gravity."},
)
handles["thermo"] = create_file(
    OUT["thermo"], "thermosteric_height",
    "EN4 thermosteric height anomaly integrated over 0-1000 m", "m",
    "1 for GEBCO depth >=1000 m with complete monthly and reference T/S through 1000 m.",
    dates, lat, lon, water_depth,
    {
        **fixed_ref_attrs,
        "calculation": (
            "Monthly potential-temperature effect with Absolute Salinity "
            "held at the fixed reference profile. Conservative Temperature "
            "for the thermosteric state is recalculated as "
            "CT_from_pt(SA_reference, monthly_potential_temperature)."
        ),
        "thermosteric_definition": (
            "nu(SA_reference, CT_from_pt(SA_reference, pt_monthly), p) "
            "minus nu(SA_reference, CT_reference, p)"
        ),
    },
)
handles["halo"] = create_file(
    OUT["halo"], "halosteric_height",
    "EN4 halosteric height anomaly integrated over 0-1000 m", "m",
    "1 for GEBCO depth >=1000 m with complete monthly and reference T/S through 1000 m.",
    dates, lat, lon, water_depth,
    {
        **fixed_ref_attrs,
        "calculation": (
            "Total steric height minus corrected thermosteric height. "
            "The residual contains the salinity contribution and the "
            "nonlinear temperature-salinity interaction."
        ),
        "closure_identity": (
            "total_steric_height = thermosteric_height + halosteric_height"
        ),
    },
)
handles["fwc"] = create_file(
    OUT["fwc"], "freshwater_content",
    "EN4 freshwater content integrated over 0-200 m", "m",
    "1 for GEBCO depth >=200 m with complete monthly salinity through 200 m.",
    dates, lat, lon, water_depth,
    {
        "integration_layer": "0-200 m",
        "reference_salinity": FWC_REFERENCE_SALINITY,
        "calculation": "FWC = integral[(S_ref - SP)/S_ref] dz.",
        "seasonal_cycle_preserved": "Yes",
    },
)
handles["sss"] = create_file(
    OUT["sss"], "surface_salinity",
    "EN4 thickness-weighted mean Practical Salinity over 0-30 m", "1",
    "1 for GEBCO depth >=30 m with complete monthly salinity through 30 m.",
    dates, lat, lon, water_depth,
    {"averaging_layer": "0-30 m", "calculation": "Thickness-weighted vertical mean Practical Salinity.", "seasonal_cycle_preserved": "Yes"},
)
handles["sss_anom"] = create_file(
    OUT["sss_anom"], "surface_salinity_anomaly",
    "EN4 0-30 m surface salinity anomaly relative to fixed 2008-2025 mean", "1",
    "1 for GEBCO depth >=30 m with complete monthly and fixed-reference salinity through 30 m.",
    dates, lat, lon, water_depth,
    {
        "averaging_layer": "0-30 m",
        "reference_state": "Fixed grid-cell all-month 2008-2025 mean 0-30 m Practical Salinity.",
        "calculation": "Monthly 0-30 m mean salinity minus fixed all-month grid-cell mean.",
        "seasonal_cycle_preserved": "Yes; calendar-month climatology was not removed.",
    },
)
# Add fixed SSS reference to anomaly file.
ref_var = handles["sss_anom"]["root"].createVariable(
    "surface_salinity_reference", "f4", ("latitude", "longitude"),
    zlib=True, complevel=COMPLEVEL, shuffle=True,
    chunksizes=(min(ny, 24), min(nx, 120)), fill_value=FILL_FLOAT,
)
ref_var[:] = np.ma.masked_invalid(np.where(sss_ref_valid, sss_ref, np.nan).astype(np.float32))
ref_var.long_name = "fixed 2008-2025 all-month mean 0-30 m Practical Salinity"
ref_var.units = "1"

handles["strat"] = create_file(
    OUT["strat"], "stratification_20_200m",
    "EN4 potential-density stratification between 20 and 200 m", "kg m-3",
    "1 for GEBCO depth >=200 m with complete monthly T/S through 200 m.",
    dates, lat, lon, water_depth,
    {
        "upper_depth": "20 m",
        "lower_depth": "200 m",
        "calculation": "TEOS-10 sigma0(200 m) minus sigma0(20 m); positive means stable stratification.",
        "seasonal_cycle_preserved": "Yes",
    },
)
handles["ohc"] = create_file(
    OUT["ohc"], "ocean_heat_content",
    "EN4 ocean heat content integrated over 0-1000 m", "J m-2",
    "1 for GEBCO depth >=1000 m with complete monthly T/S through 1000 m.",
    dates, lat, lon, water_depth,
    {
        "integration_layer": "0-1000 m",
        "reference_conservative_temperature_C": OHC_REFERENCE_CT_C,
        "heat_capacity_constant_J_kg_K": CP0,
        "calculation": "Integral[rho(SA,CT,p) * cp0 * (CT - CT_reference)] dz.",
        "seasonal_cycle_preserved": "Yes",
    },
)

# 8. Process month-by-month
summary = []
try:
    for i in range(nt):
        date = pd.Timestamp(dates[i])
        print(f"[{i+1:03d}/{nt:03d}] {date:%Y-%m}")

        t_native_k = temp.isel(time=i).load().values.astype(float)
        s_native = sal.isel(time=i).load().values.astype(float)
        t_k = interpolate_depth(t_native_k, native_depth, analysis_depth)
        SP = interpolate_depth(s_native, native_depth, analysis_depth)
        pt_c = t_k - K_TO_C

        complete1000 = np.isfinite(pt_c[mask1000]).all(axis=0) & np.isfinite(SP[mask1000]).all(axis=0)
        complete200_ts = np.isfinite(pt_c[mask200]).all(axis=0) & np.isfinite(SP[mask200]).all(axis=0)
        complete200_s = np.isfinite(SP[mask200]).all(axis=0)
        complete30_s = np.isfinite(SP[mask30]).all(axis=0)

        valid1000 = bathy1000 & ref_complete1000 & complete1000
        valid200_ts = bathy200 & complete200_ts
        valid200_s = bathy200 & complete200_s
        valid30_s = bathy30 & complete30_s

        # Full monthly TEOS-10 state
        SA = gsw.SA_from_SP(
            SP,
            p3,
            lon3,
            lat3,
        )

        CT = gsw.CT_from_pt(
            SA,
            pt_c,
        )

        nu = gsw.specvol(
            SA,
            CT,
            p3,
        )

        # Total steric height
        # Monthly temperature and monthly salinity both vary.
        total = integrate_steric(
            nu - nu_ref,
            p3,
            g3,
        )

        # Corrected thermosteric height
        #
        # Hold Absolute Salinity fixed at SA_ref. Because
        # Conservative Temperature depends weakly on salinity,
        # recalculate CT using SA_ref and monthly potential
        # temperature before evaluating specific volume.
        CT_thermo = gsw.CT_from_pt(
            SA_ref,
            pt_c,
        )

        nu_thermo = gsw.specvol(
            SA_ref,
            CT_thermo,
            p3,
        )

        thermo = integrate_steric(
            nu_thermo - nu_ref,
            p3,
            g3,
        )

        # Residual partition preserves exact closure in float64.
        halo = total - thermo

        closure_error = (
            total
            - thermo
            - halo
        )

        fwc = integrate_depth((FWC_REFERENCE_SALINITY - SP[mask200]) / FWC_REFERENCE_SALINITY, depth200)
        sss = layer_mean(SP[mask30], depth30, MAX_SSS_DEPTH)
        sss_anom = sss - sss_ref
        valid_sss_anom = valid30_s & sss_ref_valid

        sigma0 = gsw.sigma0(SA, CT)
        strat = sigma0[i200] - sigma0[i20]

        rho = gsw.rho(SA, CT, p3)
        ohc = integrate_depth(
            rho[mask1000] * CP0 * (CT[mask1000] - OHC_REFERENCE_CT_C),
            depth1000,
        )

        write_month(handles["total"], i, total, valid1000)
        write_month(handles["thermo"], i, thermo, valid1000)
        write_month(handles["halo"], i, halo, valid1000)
        write_month(handles["fwc"], i, fwc, valid200_s)
        write_month(handles["sss"], i, sss, valid30_s)
        write_month(handles["sss_anom"], i, sss_anom, valid_sss_anom)
        write_month(handles["strat"], i, strat, valid200_ts)
        write_month(handles["ohc"], i, ohc, valid1000)

        summary.append({
            "time": date.strftime("%Y-%m-%d"),
            "valid_1000m_cells": int(valid1000.sum()),
            "valid_200m_TS_cells": int(valid200_ts.sum()),
            "valid_200m_salinity_cells": int(valid200_s.sum()),
            "valid_30m_salinity_cells": int(valid30_s.sum()),
            "total_steric_min_m": float(np.nanmin(np.where(valid1000, total, np.nan))) if valid1000.any() else np.nan,
            "total_steric_max_m": float(np.nanmax(np.where(valid1000, total, np.nan))) if valid1000.any() else np.nan,
            "FWC_min_m": float(np.nanmin(np.where(valid200_s, fwc, np.nan))) if valid200_s.any() else np.nan,
            "FWC_max_m": float(
                np.nanmax(
                    np.where(
                        valid200_s,
                        fwc,
                        np.nan,
                    )
                )
            ) if valid200_s.any() else np.nan,
            "thermosteric_min_m": float(
                np.nanmin(
                    np.where(
                        valid1000,
                        thermo,
                        np.nan,
                    )
                )
            ) if valid1000.any() else np.nan,
            "thermosteric_max_m": float(
                np.nanmax(
                    np.where(
                        valid1000,
                        thermo,
                        np.nan,
                    )
                )
            ) if valid1000.any() else np.nan,
            "halosteric_min_m": float(
                np.nanmin(
                    np.where(
                        valid1000,
                        halo,
                        np.nan,
                    )
                )
            ) if valid1000.any() else np.nan,
            "halosteric_max_m": float(
                np.nanmax(
                    np.where(
                        valid1000,
                        halo,
                        np.nan,
                    )
                )
            ) if valid1000.any() else np.nan,
            "maximum_absolute_steric_closure_error_m": float(
                np.nanmax(
                    np.abs(
                        np.where(
                            valid1000,
                            closure_error,
                            np.nan,
                        )
                    )
                )
            ) if valid1000.any() else np.nan,
        })

        del (
            t_native_k,
            s_native,
            t_k,
            SP,
            pt_c,
            SA,
            CT,
            CT_thermo,
            nu,
            nu_thermo,
            total,
            thermo,
            halo,
            closure_error,
            fwc,
            sss,
            sss_anom,
            sigma0,
            strat,
            rho,
            ohc,
        )
        gc.collect()
finally:
    for h in handles.values():
        try:
            h["root"].close()
        except Exception:
            pass

pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)

# 9. Verification
lines = [
    "=" * 100,
    "FINAL CORRECTED EN4 DERIVED-PRODUCT VERIFICATION",
    "=" * 100,
]

for key, path in OUT.items():

    if not path.exists():

        lines.append(
            f"\nMISSING: {path}"
        )

        continue

    with xr.open_dataset(
        path,
        decode_times=True,
        mask_and_scale=True,
    ) as check:

        science = [
            variable
            for variable in check.data_vars
            if variable not in {
                "valid_layer_mask",
                "gebco_water_depth",
                "surface_salinity_reference",
            }
        ][0]

        dates_check = pd.to_datetime(
            check.time.values
        )

        valid_count = int(
            (
                check.valid_layer_mask == 1
            )
            .sum()
            .values
        )

        science_values = check[
            science
        ]

        finite_count = int(
            science_values.notnull()
            .sum()
            .values
        )

        temporal_standard_deviation = float(
            science_values.std(
                dim="time",
                skipna=True,
            )
            .mean(
                skipna=True
            )
            .values
        )

        lines.extend(
            [
                "",
                f"Product: {key}",
                f"File: {path}",
                f"Dimensions: {dict(check.sizes)}",
                f"Variable: {science}",
                (
                    "Units: "
                    f"{check[science].attrs.get('units', '')}"
                ),
                f"First month: {dates_check[0]}",
                f"Last month: {dates_check[-1]}",
                f"Months: {len(dates_check)}",
                (
                    "Valid monthly grid values: "
                    f"{valid_count:,}"
                ),
                (
                    "Finite science values: "
                    f"{finite_count:,}"
                ),
                (
                    "Mean grid-cell temporal standard deviation: "
                    f"{temporal_standard_deviation:.8g}"
                ),
                (
                    "File size: "
                    f"{path.stat().st_size / 1024**3:.3f} GB"
                ),
            ]
        )
      
# Cross-file steric closure after float32 storage
with xr.open_dataset(
    OUT["total"],
    decode_times=True,
    mask_and_scale=True,
) as total_check, xr.open_dataset(
    OUT["thermo"],
    decode_times=True,
    mask_and_scale=True,
) as thermo_check, xr.open_dataset(
    OUT["halo"],
    decode_times=True,
    mask_and_scale=True,
) as halo_check:

    common_valid = (
        total_check["valid_layer_mask"] == 1
    ) & (
        thermo_check["valid_layer_mask"] == 1
    ) & (
        halo_check["valid_layer_mask"] == 1
    )

    stored_closure = (
        total_check["total_steric_height"]
        - thermo_check["thermosteric_height"]
        - halo_check["halosteric_height"]
    ).where(
        common_valid
    )

    maximum_stored_closure_error = float(
        np.abs(
            stored_closure
        )
        .max(
            skipna=True
        )
        .values
    )

    mean_stored_closure_error = float(
        np.abs(
            stored_closure
        )
        .mean(
            skipna=True
        )
        .values
    )

    mask_agreement = bool(
        np.array_equal(
            total_check["valid_layer_mask"].values,
            thermo_check["valid_layer_mask"].values,
        )
        and np.array_equal(
            total_check["valid_layer_mask"].values,
            halo_check["valid_layer_mask"].values,
        )
    )

    lines.extend(
        [
            "",
            "-" * 100,
            "STERIC DECOMPOSITION QUALITY CONTROL",
            "-" * 100,
            (
                "Thermosteric method: monthly potential temperature "
                "with Absolute Salinity held at the fixed reference profile."
            ),
            (
                "CT_thermo calculation: "
                "gsw.CT_from_pt(SA_ref, pt_monthly)"
            ),
            (
                "Halosteric definition: "
                "total steric minus corrected thermosteric."
            ),
            (
                "Valid-mask agreement among total/thermo/halo: "
                f"{mask_agreement}"
            ),
            (
                "Maximum stored closure error |total - thermo - halo|: "
                f"{maximum_stored_closure_error:.10e} m"
            ),
            (
                "Mean stored closure error |total - thermo - halo|: "
                f"{mean_stored_closure_error:.10e} m"
            ),
        ]
    )

# Confirm expected monthly sequence.
expected_time = pd.date_range(
    f"{YEAR_START}-01-01",
    f"{YEAR_END}-12-01",
    freq="MS",
)

with xr.open_dataset(
    OUT["total"],
    decode_times=True,
) as time_check:

    output_time = pd.DatetimeIndex(
        pd.to_datetime(
            time_check.time.values
        )
    )

    complete_month_sequence = (
        len(output_time) == len(expected_time)
        and np.array_equal(
            output_time.values,
            expected_time.values,
        )
    )

lines.extend(
    [
        "",
        (
            "Complete January 2008–December 2025 monthly sequence: "
            f"{complete_month_sequence}"
        ),
        (
            "Expected months: "
            f"{len(expected_time)}"
        ),
        (
            "Output months: "
            f"{len(output_time)}"
        ),
    ]
)

VERIFY_TXT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(
    "\n".join(lines)
)

print("\nSummary:", SUMMARY_CSV)
print("Verification:", VERIFY_TXT)

if not complete_month_sequence:
    raise RuntimeError(
        "The output monthly sequence is incomplete."
    )

if not mask_agreement:
    raise RuntimeError(
        "The total, thermosteric and halosteric masks do not agree."
    )

if maximum_stored_closure_error > 1.0e-6:
    raise RuntimeError(
        "Stored steric closure error exceeds 1e-6 m."
    )

print(
    "\nALL REQUESTED FILES CREATED SUCCESSFULLY "
    "WITH THE CORRECTED THERMOSTERIC DECOMPOSITION."
)

print(
    "For later trend, EOF and correlation work, remove the "
    "calendar-month climatology from these outputs."
)

ds.close()
bg.close()

# Commented out IPython magic to ensure Python compatibility.


# DATA PRODUCT GROUP 02: FIGURE 6 SUPPORTING METRICS AND REGRESSION PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: Fig06_TS_steric_metrics_13seas_seasonal.csv
# File name: Fig06_TS_steric_metrics_13seas_seasonal.xlsx
# File name: Fig06_regression_statistics.csv

# FIGURE 6 — QUANTITATIVE INTEGRATION OF T–S-DERIVED
# HYDROGRAPHIC PROPERTIES AND STERIC-HEIGHT VARIABILITY
# ACROSS 13 ANTARCTIC SHELF SEAS
#
# Main period: 2008–2025
#
# Outputs:
#   1. Figure06_TS_steric_13seas_1080dpi.png
#   2. Figure06_TS_steric_13seas.pdf
#   3. Fig06_TS_steric_metrics_13seas_seasonal.csv
#   4. Fig06_TS_steric_metrics_13seas_seasonal.xlsx
#   5. Fig06_regression_statistics.csv
#
# IMPORTANT:
# This code uses the exact simplified water-mass rules used in
# Figures 4 and 5. It does not introduce new thresholds.

# Run once in a separate Google Colab cell:
# !pip -q install netCDF4 xarray scipy openpyxl matplotlib pandas numpy


# 0. INSTALL REQUIRED PACKAGES AUTOMATICALLY
import sys
import subprocess
import importlib.util


REQUIRED_PACKAGES = {
    "netCDF4": "netCDF4",
    "cftime": "cftime",
    "xarray": "xarray",
    "scipy": "scipy",
    "openpyxl": "openpyxl",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "numpy": "numpy",
}


missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]


if missing_packages:

    print(
        "Installing missing packages:",
        ", ".join(missing_packages)
    )

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing_packages,
        ]
    )

    print("Package installation completed.")

else:

    print("All required packages are already installed.")



# 1. IMPORTS
import gc
import os
import warnings

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from netCDF4 import Dataset
from scipy import stats



warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)

ARGO_NC = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "argo_SO_profiles_2001_2025_cleaned_gridded.nc"
)

THERMOSTERIC_NC = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "thermosteric_height_monthly_2008_2025_SO.nc"
)

HALOSTERIC_NC = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "halosteric_height_monthly_2008_2025_SO.nc"
)

FRESHWATER_NC = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "freshwater_content_monthly_2008_2025_SO.nc"
)


OUTPUT_DIRECTORY = (
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)




OUTPUT_METRICS_CSV = os.path.join(
    OUTPUT_DIRECTORY,
    "Fig06_TS_steric_metrics_13seas_seasonal.csv"
)

OUTPUT_METRICS_XLSX = os.path.join(
    OUTPUT_DIRECTORY,
    "Fig06_TS_steric_metrics_13seas_seasonal.xlsx"
)

OUTPUT_REGRESSION_CSV = os.path.join(
    OUTPUT_DIRECTORY,
    "Fig06_regression_statistics.csv"
)


# 3. ANALYSIS SETTINGS

YEAR_START = 2008
YEAR_END = 2025

LAT_MIN = -90.0
LAT_MAX = -60.0

# Argo quality-control ranges
MIN_TEMPERATURE = -3.0
MAX_TEMPERATURE = 20.0

MIN_SALINITY = 0.0
MAX_SALINITY = 42.0

# A profile is considered usable when it has at least this many
# valid paired temperature-salinity observations between 0–2000 dbar.
MIN_VALID_TS_LEVELS = 5

# Minimum profile support requested for each sea-season estimate.
MIN_PROFILES_PER_SEA_SEASON = 10

# Minimum number of valid T-S levels required for a profile-level
# diagnostic in each search interval.
MIN_LEVELS_WW_SEARCH = 3
MIN_LEVELS_SUBSURFACE_SEARCH = 5

# Gridded seasonal values are retained only when at least this many
# monthly sector means are available.
MIN_VALID_GRIDDED_MONTHS = 3

# Depth/pressure ranges
PROFILE_PRESSURE_MIN = 0.0
PROFILE_PRESSURE_MAX = 2000.0

WW_SEARCH_MIN = 50.0
WW_SEARCH_MAX = 300.0

SUBSURFACE_MIN = 200.0
SUBSURFACE_MAX = 1000.0

# Save resolution requested by the user
SAVE_DPI = 1080

# A moderate physical figure size prevents excessive memory use
# during the 1080-dpi PNG render.
FIGURE_SIZE_INCHES = (
    12.5,
    9.5
)


# 4. SEAS AND NON-OVERLAPPING LONGITUDE SECTORS

# Each interval is interpreted as [minimum, maximum), so profiles
# located exactly on a common boundary are not counted twice.
SEA_INFORMATION = [
    (
        "WED",
        "Weddell Sea",
        -60.0,
        -20.0
    ),
    (
        "KHV",
        "King Haakon VII Sea",
        -20.0,
        0.0
    ),
    (
        "RLS",
        "Riiser-Larsen Sea",
        0.0,
        10.0
    ),
    (
        "LAZ",
        "Lazarev Sea",
        10.0,
        30.0
    ),
    (
        "COS",
        "Cosmonauts Sea",
        30.0,
        50.0
    ),
    (
        "COO",
        "Cooperation Sea",
        50.0,
        70.0
    ),
    (
        "DAV",
        "Davis Sea",
        70.0,
        90.0
    ),
    (
        "MAW",
        "Mawson Sea",
        90.0,
        130.0
    ),
    (
        "DUR",
        "D'Urville Sea",
        130.0,
        150.0
    ),
    (
        "SOM",
        "Somov Sea",
        150.0,
        170.0
    ),
    (
        "ROS",
        "Ross Sea",
        170.0,
        -130.0
    ),
    (
        "AMU",
        "Amundsen Sea",
        -130.0,
        -100.0
    ),
    (
        "BEL",
        "Bellingshausen Sea",
        -100.0,
        -60.0
    ),
]


SEA_CODES = [
    item[0]
    for item in SEA_INFORMATION
]

SEA_NAMES = {
    item[0]: item[1]
    for item in SEA_INFORMATION
}

SEA_RANGES = {
    item[0]: (
        item[2],
        item[3]
    )
    for item in SEA_INFORMATION
}


# 5. AUSTRAL SEASONS AND COLOURS

SEASON_ORDER = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

SEASON_CODE = {
    "Spring": "SON",
    "Summer": "DJF",
    "Autumn": "MAM",
    "Winter": "JJA",
}

SEASON_MONTHS = {
    "Spring": [
        9,
        10,
        11
    ],
    "Summer": [
        12,
        1,
        2
    ],
    "Autumn": [
        3,
        4,
        5
    ],
    "Winter": [
        6,
        7,
        8
    ],
}

# Same colours are used in all four panels.
SEASON_COLOUR = {
    "Spring": "#D62728",
    "Summer": "#1F77B4",
    "Autumn": "#FF7F0E",
    "Winter": "#2CA02C",
}

# Small vertical offsets separate the four seasonal dots in panels a-b.
SEASON_Y_OFFSET = {
    "Spring": -0.24,
    "Summer": -0.08,
    "Autumn": 0.08,
    "Winter": 0.24,
}


# 6. GENERAL HELPER FUNCTIONS

def check_input_files(
    file_paths
):
    """
    Stop immediately and list any missing input files.
    """

    missing_files = [
        path
        for path in file_paths
        if not os.path.exists(
            path
        )
    ]

    if missing_files:

        missing_text = "\n".join(
            missing_files
        )

        raise FileNotFoundError(
            "The following required files were not found:\n"
            f"{missing_text}"
        )


def normalize_longitude(
    longitude
):
    """
    Convert valid longitude values to the interval [-180, 180).
    """

    longitude = np.asarray(
        longitude,
        dtype=np.float64
    )

    return (
        (
            longitude
            + 180.0
        )
         % 360.0
    ) - 180.0


def longitude_sector_mask(
    longitude,
    minimum_longitude,
    maximum_longitude
):
    """
    Create a non-overlapping longitude mask.

    Normal sector:
        minimum <= longitude < maximum

    Dateline-crossing sector:
        longitude >= minimum OR longitude < maximum
    """

    longitude = np.asarray(
        longitude
    )

    if minimum_longitude < maximum_longitude:

        return (
            (
                longitude
                >= minimum_longitude
            )
            &
            (
                longitude
                < maximum_longitude
            )
        )

    return (
        (
            longitude
            >= minimum_longitude
        )
        |
        (
            longitude
            < maximum_longitude
        )
    )


def month_to_season_name(
    month
):
    """
    Convert a month number to an austral season name.
    """

    for season_name, season_month_list in (
        SEASON_MONTHS.items()
    ):

        if month in season_month_list:

            return season_name

    return None


def safe_float(
    value
):
    """
    Convert a scalar to float, returning NaN when conversion fails.
    """

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return np.nan


def nanmedian_or_nan(
    values
):
    """
    Return the median of finite values or NaN.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return np.nan

    return float(
        np.median(
            values
        )
    )


def percentile_or_nan(
    values,
    percentile
):
    """
    Return a percentile of finite values or NaN.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return np.nan

    return float(
        np.percentile(
            values,
            percentile
        )
    )


# 7. EXACT WATER-MASS RULES USED IN FIGURES 4 AND 5
def classify_watermass_existing_criteria(
    salinity,
    temperature,
    pressure
):
    """
    Apply exactly the simplified water-mass hierarchy used in the
    supplied Figure 4 and Figure 5 scripts.

    Note:
    Salinity is retained as an input because this is a T-S diagnostic,
    although the supplied simplified thresholds do not impose a
    numerical salinity boundary.
    """

    if not (
        np.isfinite(
            salinity
        )
        and np.isfinite(
            temperature
        )
        and np.isfinite(
            pressure
        )
    ):

        return None

    # Antarctic Bottom Water
    if (
        pressure
        >= 1500.0
        and temperature
        <= 0.5
    ):

        return "AABW"

    # Surface waters
    if pressure < 200.0:

        if temperature <= -0.5:

            return "WW"

        return "AASW"

    # Subsurface and deep waters
    if pressure >= 200.0:

        if temperature >= 1.0:

            return "CDW"

        return "mCDW"

    return "mCDW"


def existing_mcdw_cdw_mask(
    temperature,
    pressure,
    valid_ts_mask
):
    """
    Vectorized equivalent of the existing classification for the
    combined mCDW/CDW categories.

    The AABW condition has priority and is therefore excluded.
    """

    pressure_2d = np.broadcast_to(
        pressure[
            None,
            :
        ],
        temperature.shape
    )

    aabw_mask = (
        valid_ts_mask
        &
        (
            pressure_2d
            >= 1500.0
        )
        &
        (
            temperature
            <= 0.5
        )
    )

    return (
        valid_ts_mask
        &
        (
            pressure_2d
            >= 200.0
        )
        &
        ~aabw_mask
    )


# 8. READ AND DECODE ARGO PROFILE DATA
def decode_argo_juld(
    julian_days
):
    """
    Decode Argo JULD.

    The source scripts use the standard Argo origin:
        1950-01-01 00:00:00
    """

    julian_days = np.asarray(
        julian_days,
        dtype=np.float64
    )

    output_time = np.full(
        julian_days.shape,
        np.datetime64(
            "NaT"
        ),
        dtype="datetime64[ns]"
    )

    valid_juld = (
        np.isfinite(
            julian_days
        )
        &
        (
            julian_days
            > 0.0
        )
        &
        (
            julian_days
            < 50000.0
        )
    )

    if np.any(
        valid_juld
    ):

        base_date = datetime(
            1950,
            1,
            1
        )

        output_time[
            valid_juld
        ] = np.array(
            [
                np.datetime64(
                    base_date
                    + timedelta(
                        days=float(
                            day
                        )
                    )
                )
                for day in julian_days[
                    valid_juld
                ]
            ],
            dtype="datetime64[ns]"
        )

    return output_time


def read_argo_profiles():
    """
    Read the Argo profile file and apply explicit physical-range masks.
    """

    print(
        "\nReading Argo profile data..."
    )

    with Dataset(
        ARGO_NC,
        mode="r"
    ) as dataset:

        pressure = np.ma.filled(
            dataset.variables[
                "PRES_GRID"
            ][
                :
            ],
            np.nan
        ).astype(
            np.float64
        )

        juld = np.ma.filled(
            dataset.variables[
                "JULD"
            ][
                :
            ],
            np.nan
        ).astype(
            np.float64
        )

        latitude = np.ma.filled(
            dataset.variables[
                "LATITUDE"
            ][
                :
            ],
            np.nan
        ).astype(
            np.float64
        )

        longitude_raw = np.ma.filled(
            dataset.variables[
                "LONGITUDE"
            ][
                :
            ],
            np.nan
        ).astype(
            np.float64
        )

        temperature = np.ma.filled(
            dataset.variables[
                "TEMP"
            ][
                :
            ],
            np.nan
        ).astype(
            np.float32
        )

        salinity = np.ma.filled(
            dataset.variables[
                "PSAL"
            ][
                :
            ],
            np.nan
        ).astype(
            np.float32
        )

    # Remove invalid pressure values.
    pressure[
        ~np.isfinite(
            pressure
        )
        |
        (
            pressure
            < PROFILE_PRESSURE_MIN
        )
        |
        (
            pressure
            > PROFILE_PRESSURE_MAX
        )
    ] = np.nan

    # Remove invalid positions before longitude normalization.
    latitude[
        ~np.isfinite(
            latitude
        )
        |
        (
            latitude
            < -90.0
        )
        |
        (
            latitude
            > 90.0
        )
    ] = np.nan

    longitude_valid = (
        np.isfinite(
            longitude_raw
        )
        &
        (
            longitude_raw
            >= -360.0
        )
        &
        (
            longitude_raw
            <= 360.0
        )
    )

    longitude = np.full(
        longitude_raw.shape,
        np.nan,
        dtype=np.float64
    )

    longitude[
        longitude_valid
    ] = normalize_longitude(
        longitude_raw[
            longitude_valid
        ]
    )

    # Explicitly remove 99999-style values and other physical outliers.
    temperature[
        ~np.isfinite(
            temperature
        )
        |
        (
            temperature
            <= MIN_TEMPERATURE
        )
        |
        (
            temperature
            >= MAX_TEMPERATURE
        )
    ] = np.nan

    salinity[
        ~np.isfinite(
            salinity
        )
        |
        (
            salinity
            <= MIN_SALINITY
        )
        |
        (
            salinity
            >= MAX_SALINITY
        )
    ] = np.nan

    time = decode_argo_juld(
        juld
    )

    time_index = pd.DatetimeIndex(
        pd.to_datetime(
            time
        )
    )

    year = time_index.year.to_numpy()
    month = time_index.month.to_numpy()

    season = np.array(
        [
            (
                month_to_season_name(
                    int(
                        month_value
                    )
                )
                if pd.notna(
                    month_value
                )
                else None
            )
            for month_value in month
        ],
        dtype=object
    )

    print(
        "Argo arrays loaded:",
        f"{temperature.shape[0]:,} profiles × "
        f"{temperature.shape[1]:,} pressure levels"
    )

    return {
        "pressure": pressure,
        "latitude": latitude,
        "longitude": longitude,
        "time": time,
        "year": year,
        "month": month,
        "season": season,
        "temperature": temperature,
        "salinity": salinity,
    }


# 9. CALCULATE PROFILE-BASED SEA-SEASON METRICS
def calculate_profile_metrics(
    argo_data
):
    """
    Calculate:
      - valid profile count
      - profile occurrence of mCDW/CDW
      - median WW core temperature and pressure
      - median profile-wise maximum temperature from 200–1000 dbar
    """

    pressure = argo_data[
        "pressure"
    ]

    latitude = argo_data[
        "latitude"
    ]

    longitude = argo_data[
        "longitude"
    ]

    year = argo_data[
        "year"
    ]

    season_array = argo_data[
        "season"
    ]

    temperature = argo_data[
        "temperature"
    ]

    salinity = argo_data[
        "salinity"
    ]

    pressure_in_profile_range = (
        np.isfinite(
            pressure
        )
        &
        (
            pressure
            >= PROFILE_PRESSURE_MIN
        )
        &
        (
            pressure
            <= PROFILE_PRESSURE_MAX
        )
    )

    ww_pressure_indices = np.where(
        np.isfinite(
            pressure
        )
        &
        (
            pressure
            >= WW_SEARCH_MIN
        )
        &
        (
            pressure
            <= WW_SEARCH_MAX
        )
    )[0]

    subsurface_pressure_indices = np.where(
        np.isfinite(
            pressure
        )
        &
        (
            pressure
            >= SUBSURFACE_MIN
        )
        &
        (
            pressure
            <= SUBSURFACE_MAX
        )
    )[0]

    if ww_pressure_indices.size == 0:

        raise ValueError(
            "No pressure levels were found in the WW search range."
        )

    if subsurface_pressure_indices.size == 0:

        raise ValueError(
            "No pressure levels were found in the 200–1000 dbar range."
        )

    base_profile_mask = (
        np.isfinite(
            latitude
        )
        &
        np.isfinite(
            longitude
        )
        &
        (
            latitude
            >= LAT_MIN
        )
        &
        (
            latitude
            <= LAT_MAX
        )
        &
        (
            year
            >= YEAR_START
        )
        &
        (
            year
            <= YEAR_END
        )
        &
        pd.notna(
            season_array
        )
    )

    print(
        "Profiles within 2008–2025 and 60–90°S:",
        f"{int(base_profile_mask.sum()):,}"
    )

    output_rows = []

    for sea_code in SEA_CODES:

        longitude_minimum, longitude_maximum = (
            SEA_RANGES[
                sea_code
            ]
        )

        sea_mask = longitude_sector_mask(
            longitude,
            longitude_minimum,
            longitude_maximum
        )

        for season_name in SEASON_ORDER:

            profile_mask = (
                base_profile_mask
                &
                sea_mask
                &
                (
                    season_array
                    == season_name
                )
            )

            profile_indices = np.where(
                profile_mask
            )[0]

            selected_profile_count = int(
                profile_indices.size
            )

            row = {
                "sea_abbreviation": sea_code,
                "sea_name": SEA_NAMES[
                    sea_code
                ],
                "season": season_name,
                "season_code": SEASON_CODE[
                    season_name
                ],
                "selected_profile_count": selected_profile_count,
                "valid_profile_count": 0,
                "mCDW_CDW_profile_count": 0,
                "mCDW_CDW_occurrence_raw_percent": np.nan,
                "mCDW_CDW_occurrence_percent": np.nan,
                "WW_core_profile_count": 0,
                "WW_core_temperature_median_degC": np.nan,
                "WW_core_temperature_p25_degC": np.nan,
                "WW_core_temperature_p75_degC": np.nan,
                "WW_core_pressure_median_dbar": np.nan,
                "WW_core_pressure_p25_dbar": np.nan,
                "WW_core_pressure_p75_dbar": np.nan,
                "subsurface_maximum_profile_count": 0,
                (
                    "maximum_subsurface_temperature_"
                    "200_1000dbar_median_degC"
                ): np.nan,
                (
                    "maximum_subsurface_temperature_"
                    "200_1000dbar_p25_degC"
                ): np.nan,
                (
                    "maximum_subsurface_temperature_"
                    "200_1000dbar_p75_degC"
                ): np.nan,
                "profile_support_status": "insufficient",
            }

            if selected_profile_count == 0:

                output_rows.append(
                    row
                )

                continue

            temperature_subset = temperature[
                profile_indices,
                :
            ].astype(
                np.float64,
                copy=False
            )

            salinity_subset = salinity[
                profile_indices,
                :
            ].astype(
                np.float64,
                copy=False
            )

            valid_ts = (
                np.isfinite(
                    temperature_subset
                )
                &
                np.isfinite(
                    salinity_subset
                )
                &
                pressure_in_profile_range[
                    None,
                    :
                ]
            )

            valid_level_count = valid_ts.sum(
                axis=1
            )

            valid_profile_mask = (
                valid_level_count
                >= MIN_VALID_TS_LEVELS
            )

            valid_profile_count = int(
                valid_profile_mask.sum()
            )

            row[
                "valid_profile_count"
            ] = valid_profile_count

            # mCDW/CDW profile occurrence
   

            mcdw_cdw_samples = existing_mcdw_cdw_mask(
                temperature_subset,
                pressure,
                valid_ts
            )

            profile_contains_mcdw_cdw = (
                valid_profile_mask
                &
                np.any(
                    mcdw_cdw_samples,
                    axis=1
                )
            )

            mcdw_cdw_profile_count = int(
                profile_contains_mcdw_cdw.sum()
            )

            row[
                "mCDW_CDW_profile_count"
            ] = mcdw_cdw_profile_count

            if valid_profile_count > 0:

                raw_occurrence = (
                    100.0
                    * mcdw_cdw_profile_count
                    / valid_profile_count
                )

                row[
                    "mCDW_CDW_occurrence_raw_percent"
                ] = float(
                    raw_occurrence
                )

                if (
                    valid_profile_count
                    >= MIN_PROFILES_PER_SEA_SEASON
                ):

                    row[
                        "mCDW_CDW_occurrence_percent"
                    ] = float(
                        raw_occurrence
                    )

            # Winter Water core
            ww_temperature = temperature_subset[
                :,
                ww_pressure_indices
            ]

            ww_salinity = salinity_subset[
                :,
                ww_pressure_indices
            ]

            ww_pressure = pressure[
                ww_pressure_indices
            ]

            ww_valid = (
                np.isfinite(
                    ww_temperature
                )
                &
                np.isfinite(
                    ww_salinity
                )
                &
                valid_profile_mask[
                    :,
                    None
                ]
            )

            ww_valid_level_count = ww_valid.sum(
                axis=1
            )

            ww_search_profile_mask = (
                ww_valid_level_count
                >= MIN_LEVELS_WW_SEARCH
            )

            ww_core_temperatures = []
            ww_core_pressures = []

            for local_profile_index in np.where(
                ww_search_profile_mask
            )[0]:

                profile_temperature = np.where(
                    ww_valid[
                        local_profile_index
                    ],
                    ww_temperature[
                        local_profile_index
                    ],
                    np.nan
                )

                minimum_local_index = int(
                    np.nanargmin(
                        profile_temperature
                    )
                )

                core_temperature = float(
                    ww_temperature[
                        local_profile_index,
                        minimum_local_index
                    ]
                )

                core_salinity = float(
                    ww_salinity[
                        local_profile_index,
                        minimum_local_index
                    ]
                )

                core_pressure = float(
                    ww_pressure[
                        minimum_local_index
                    ]
                )

                core_classification = (
                    classify_watermass_existing_criteria(
                        core_salinity,
                        core_temperature,
                        core_pressure
                    )
                )

                if core_classification == "WW":

                    ww_core_temperatures.append(
                        core_temperature
                    )

                    ww_core_pressures.append(
                        core_pressure
                    )

            ww_core_profile_count = len(
                ww_core_temperatures
            )

            row[
                "WW_core_profile_count"
            ] = ww_core_profile_count

            if (
                ww_core_profile_count
                >= MIN_PROFILES_PER_SEA_SEASON
            ):

                row[
                    "WW_core_temperature_median_degC"
                ] = nanmedian_or_nan(
                    ww_core_temperatures
                )

                row[
                    "WW_core_temperature_p25_degC"
                ] = percentile_or_nan(
                    ww_core_temperatures,
                    25
                )

                row[
                    "WW_core_temperature_p75_degC"
                ] = percentile_or_nan(
                    ww_core_temperatures,
                    75
                )

                row[
                    "WW_core_pressure_median_dbar"
                ] = nanmedian_or_nan(
                    ww_core_pressures
                )

                row[
                    "WW_core_pressure_p25_dbar"
                ] = percentile_or_nan(
                    ww_core_pressures,
                    25
                )

                row[
                    "WW_core_pressure_p75_dbar"
                ] = percentile_or_nan(
                    ww_core_pressures,
                    75
                )

            # Profile-wise maximum subsurface temperature

            subsurface_temperature = temperature_subset[
                :,
                subsurface_pressure_indices
            ]

            subsurface_salinity = salinity_subset[
                :,
                subsurface_pressure_indices
            ]

            subsurface_valid = (
                np.isfinite(
                    subsurface_temperature
                )
                &
                np.isfinite(
                    subsurface_salinity
                )
                &
                valid_profile_mask[
                    :,
                    None
                ]
            )

            subsurface_level_count = subsurface_valid.sum(
                axis=1
            )

            subsurface_profile_mask = (
                subsurface_level_count
                >= MIN_LEVELS_SUBSURFACE_SEARCH
            )

            profile_maximum_temperature = np.full(
                selected_profile_count,
                np.nan,
                dtype=np.float64
            )

            if np.any(
                subsurface_profile_mask
            ):

                valid_subsurface_temperature = np.where(
                    subsurface_valid[
                        subsurface_profile_mask
                    ],
                    subsurface_temperature[
                        subsurface_profile_mask
                    ],
                    np.nan
                )

                profile_maximum_temperature[
                    subsurface_profile_mask
                ] = np.nanmax(
                    valid_subsurface_temperature,
                    axis=1
                )

            finite_profile_maximum = (
                profile_maximum_temperature[
                    np.isfinite(
                        profile_maximum_temperature
                    )
                ]
            )

            subsurface_profile_count = int(
                finite_profile_maximum.size
            )

            row[
                "subsurface_maximum_profile_count"
            ] = subsurface_profile_count

            if (
                subsurface_profile_count
                >= MIN_PROFILES_PER_SEA_SEASON
            ):

                row[
                    (
                        "maximum_subsurface_temperature_"
                        "200_1000dbar_median_degC"
                    )
                ] = nanmedian_or_nan(
                    finite_profile_maximum
                )

                row[
                    (
                        "maximum_subsurface_temperature_"
                        "200_1000dbar_p25_degC"
                    )
                ] = percentile_or_nan(
                    finite_profile_maximum,
                    25
                )

                row[
                    (
                        "maximum_subsurface_temperature_"
                        "200_1000dbar_p75_degC"
                    )
                ] = percentile_or_nan(
                    finite_profile_maximum,
                    75
                )

            if (
                valid_profile_count
                >= MIN_PROFILES_PER_SEA_SEASON
            ):

                row[
                    "profile_support_status"
                ] = "sufficient"

            output_rows.append(
                row
            )

            del (
                temperature_subset,
                salinity_subset,
                valid_ts,
                mcdw_cdw_samples,
                ww_temperature,
                ww_salinity,
                subsurface_temperature,
                subsurface_salinity
            )

    output_dataframe = pd.DataFrame(
        output_rows
    )

    return output_dataframe


# 10. GRIDDED PRODUCT HELPERS

def mask_fill_values(
    data_array
):
    """
    Convert encoded fill values to NaN without assuming that xarray
    decoded every source file identically.
    """

    cleaned = data_array.astype(
        np.float64
    )

    fill_candidates = [
        data_array.attrs.get(
            "_FillValue"
        ),
        data_array.attrs.get(
            "missing_value"
        ),
        data_array.encoding.get(
            "_FillValue"
        ),
    ]

    for fill_value in fill_candidates:

        if fill_value is None:

            continue

        try:

            cleaned = cleaned.where(
                ~np.isclose(
                    cleaned,
                    float(
                        fill_value
                    )
                )
            )

        except (
            TypeError,
            ValueError
        ):

            pass

    cleaned = cleaned.where(
        np.isfinite(
            cleaned
        )
    )

    # The source inventory identifies -9999 as the fill value.
    cleaned = cleaned.where(
        cleaned
        > -9000.0
    )

    return cleaned


def prepare_gridded_dataset(
    netcdf_path,
    variable_name
):
    """
    Open one monthly gridded product, standardize longitude and mask
    fill values.
    """

    dataset = xr.open_dataset(
        netcdf_path,
        decode_times=True,
        mask_and_scale=True
    )

    required_names = {
        "time",
        "lat",
        "lon",
        variable_name,
    }

    missing_names = [
        name
        for name in required_names
        if name not in dataset
        and name not in dataset.coords
    ]

    if missing_names:

        dataset.close()

        raise KeyError(
            f"{netcdf_path} is missing: {missing_names}"
        )

    data_array = mask_fill_values(
        dataset[
            variable_name
        ]
    )

    longitude_normalized = normalize_longitude(
        dataset[
            "lon"
        ].values
    )

    data_array = data_array.assign_coords(
        lon=(
            "lon",
            longitude_normalized
        )
    ).sortby(
        "lon"
    )

    if "nprof" in dataset:

        nprof = dataset[
            "nprof"
        ].astype(
            np.float64
        )

        nprof = nprof.assign_coords(
            lon=(
                "lon",
                longitude_normalized
            )
        ).sortby(
            "lon"
        )

        data_array = data_array.where(
            nprof
            > 0.0
        )

    else:

        nprof = None

    data_array = data_array.sel(
        time=slice(
            f"{YEAR_START}-01-01",
            f"{YEAR_END}-12-31"
        )
    )

    if nprof is not None:

        nprof = nprof.sel(
            time=data_array[
                "time"
            ]
        )

    return (
        dataset,
        data_array,
        nprof
    )


def sector_monthly_area_weighted_mean(
    data_array,
    nprof,
    sea_code,
    season_name
):
    """
    Calculate monthly sector means using cosine-latitude weighting,
    then return the mean of the available monthly sector means.

    The nprof field is used as a validity mask and support diagnostic,
    not as a spatial weight. This avoids biasing the regional mean
    toward the most heavily sampled cells.
    """

    longitude_minimum, longitude_maximum = (
        SEA_RANGES[
            sea_code
        ]
    )

    longitude_values = data_array[
        "lon"
    ].values

    longitude_mask_values = longitude_sector_mask(
        longitude_values,
        longitude_minimum,
        longitude_maximum
    )

    longitude_mask = xr.DataArray(
        longitude_mask_values,
        dims=[
            "lon"
        ],
        coords={
            "lon": data_array[
                "lon"
            ]
        }
    )

    month_mask_values = np.isin(
        data_array[
            "time"
        ].dt.month.values,
        SEASON_MONTHS[
            season_name
        ]
    )

    month_mask = xr.DataArray(
        month_mask_values,
        dims=[
            "time"
        ],
        coords={
            "time": data_array[
                "time"
            ]
        }
    )

    selected_data = data_array.where(
        longitude_mask,
        drop=True
    ).where(
        month_mask,
        drop=True
    )

    if selected_data.sizes.get(
        "time",
        0
    ) == 0:

        return {
            "seasonal_mean": np.nan,
            "seasonal_standard_deviation": np.nan,
            "valid_month_count": 0,
            "valid_gridcell_month_count": 0,
            "product_profile_count": 0,
        }

    latitude_weight = xr.DataArray(
        np.cos(
            np.deg2rad(
                selected_data[
                    "lat"
                ].values
            )
        ),
        dims=[
            "lat"
        ],
        coords={
            "lat": selected_data[
                "lat"
            ]
        }
    )

    broadcast_weight = latitude_weight.broadcast_like(
        selected_data
    )

    valid_data = selected_data.notnull()

    weighted_numerator = (
        selected_data
        * broadcast_weight
    ).sum(
        dim=[
            "lat",
            "lon"
        ],
        skipna=True
    )

    weighted_denominator = broadcast_weight.where(
        valid_data
    ).sum(
        dim=[
            "lat",
            "lon"
        ],
        skipna=True
    )

    monthly_sector_mean = (
        weighted_numerator
        / weighted_denominator
    ).where(
        weighted_denominator
        > 0.0
    )

    monthly_values = np.asarray(
        monthly_sector_mean.values,
        dtype=np.float64
    )

    monthly_values = monthly_values[
        np.isfinite(
            monthly_values
        )
    ]

    valid_month_count = int(
        monthly_values.size
    )

    valid_gridcell_month_count = int(
        valid_data.sum().values
    )

    if nprof is not None:

        selected_nprof = nprof.where(
            longitude_mask,
            drop=True
        ).where(
            month_mask,
            drop=True
        )

        product_profile_count = int(
            np.nansum(
                selected_nprof.values
            )
        )

    else:

        product_profile_count = 0

    if (
        valid_month_count
        < MIN_VALID_GRIDDED_MONTHS
    ):

        seasonal_mean = np.nan
        seasonal_standard_deviation = np.nan

    else:

        seasonal_mean = float(
            np.mean(
                monthly_values
            )
        )

        seasonal_standard_deviation = float(
            np.std(
                monthly_values,
                ddof=1
            )
        ) if valid_month_count > 1 else 0.0

    return {
        "seasonal_mean": seasonal_mean,
        "seasonal_standard_deviation": (
            seasonal_standard_deviation
        ),
        "valid_month_count": valid_month_count,
        "valid_gridcell_month_count": (
            valid_gridcell_month_count
        ),
        "product_profile_count": product_profile_count,
    }


def calculate_gridded_product_metrics(
    netcdf_path,
    variable_name,
    output_prefix,
    conversion_factor=1.0
):
    """
    Calculate one seasonal metric for every sea-season combination.
    """

    print(
        f"\nReading gridded product: {variable_name}"
    )

    (
        dataset,
        data_array,
        nprof
    ) = prepare_gridded_dataset(
        netcdf_path,
        variable_name
    )

    output_rows = []

    for sea_code in SEA_CODES:

        for season_name in SEASON_ORDER:

            summary = sector_monthly_area_weighted_mean(
                data_array,
                nprof,
                sea_code,
                season_name
            )

            seasonal_mean = summary[
                "seasonal_mean"
            ]

            seasonal_standard_deviation = summary[
                "seasonal_standard_deviation"
            ]

            if np.isfinite(
                seasonal_mean
            ):

                seasonal_mean = (
                    seasonal_mean
                    * conversion_factor
                )

            if np.isfinite(
                seasonal_standard_deviation
            ):

                seasonal_standard_deviation = (
                    seasonal_standard_deviation
                    * conversion_factor
                )

            output_rows.append(
                {
                    "sea_abbreviation": sea_code,
                    "season": season_name,
                    output_prefix: seasonal_mean,
                    (
                        output_prefix
                        + "_monthly_sd"
                    ): seasonal_standard_deviation,
                    (
                        output_prefix
                        + "_valid_month_count"
                    ): summary[
                        "valid_month_count"
                    ],
                    (
                        output_prefix
                        + "_valid_gridcell_month_count"
                    ): summary[
                        "valid_gridcell_month_count"
                    ],
                    (
                        output_prefix
                        + "_product_profile_count"
                    ): summary[
                        "product_profile_count"
                    ],
                }
            )

    dataset.close()

    return pd.DataFrame(
        output_rows
    )


# 11. REGRESSION AND CONFIDENCE-INTERVAL FUNCTIONS
def calculate_regression_statistics(
    dataframe,
    x_column,
    y_column,
    panel_name,
    slope_unit
):
    """
    Calculate Pearson correlation, Spearman correlation and ordinary
    least-squares slope with a two-sided 95% confidence interval.
    """

    valid_data = dataframe[
        [
            x_column,
            y_column
        ]
    ].replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    ).dropna()

    x = valid_data[
        x_column
    ].to_numpy(
        dtype=np.float64
    )

    y = valid_data[
        y_column
    ].to_numpy(
        dtype=np.float64
    )

    n_observations = int(
        x.size
    )

    output = {
        "panel": panel_name,
        "x_variable": x_column,
        "y_variable": y_column,
        "n": n_observations,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "regression_slope": np.nan,
        "slope_unit": slope_unit,
        "slope_ci95_lower": np.nan,
        "slope_ci95_upper": np.nan,
        "intercept": np.nan,
        "regression_p": np.nan,
        "r_squared": np.nan,
    }

    if (
        n_observations
        < 3
        or np.nanstd(
            x
        )
        == 0.0
        or np.nanstd(
            y
        )
        == 0.0
    ):

        return output

    pearson_result = stats.pearsonr(
        x,
        y
    )

    spearman_result = stats.spearmanr(
        x,
        y
    )

    regression_result = stats.linregress(
        x,
        y
    )

    degrees_of_freedom = (
        n_observations
        - 2
    )

    t_critical = stats.t.ppf(
        0.975,
        degrees_of_freedom
    )

    slope_margin = (
        t_critical
        * regression_result.stderr
    )

    output.update(
        {
            "pearson_r": float(
                pearson_result.statistic
            ),
            "pearson_p": float(
                pearson_result.pvalue
            ),
            "spearman_rho": float(
                spearman_result.statistic
            ),
            "spearman_p": float(
                spearman_result.pvalue
            ),
            "regression_slope": float(
                regression_result.slope
            ),
            "slope_ci95_lower": float(
                regression_result.slope
                - slope_margin
            ),
            "slope_ci95_upper": float(
                regression_result.slope
                + slope_margin
            ),
            "intercept": float(
                regression_result.intercept
            ),
            "regression_p": float(
                regression_result.pvalue
            ),
            "r_squared": float(
                regression_result.rvalue
                ** 2
            ),
        }
    )

    return output


def regression_line_and_mean_ci(
    x,
    y,
    number_of_line_points=250
):
    """
    Return fitted values and the 95% confidence interval for the mean
    regression response.
    """

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    valid = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if (
        x.size
        < 3
        or np.std(
            x
        )
        == 0.0
    ):

        return None

    regression_result = stats.linregress(
        x,
        y
    )

    x_line = np.linspace(
        np.min(
            x
        ),
        np.max(
            x
        ),
        number_of_line_points
    )

    y_line = (
        regression_result.intercept
        + regression_result.slope
        * x_line
    )

    fitted_observations = (
        regression_result.intercept
        + regression_result.slope
        * x
    )

    residuals = (
        y
        - fitted_observations
    )

    degrees_of_freedom = (
        x.size
        - 2
    )

    residual_standard_error = np.sqrt(
        np.sum(
            residuals
            ** 2
        )
        / degrees_of_freedom
    )

    x_mean = np.mean(
        x
    )

    sum_squared_x = np.sum(
        (
            x
            - x_mean
        )
        ** 2
    )

    t_critical = stats.t.ppf(
        0.975,
        degrees_of_freedom
    )

    confidence_half_width = (
        t_critical
        * residual_standard_error
        * np.sqrt(
            (
                1.0
                / x.size
            )
            +
            (
                (
                    x_line
                    - x_mean
                )
                ** 2
                / sum_squared_x
            )
        )
    )

    return {
        "x_line": x_line,
        "y_line": y_line,
        "ci_lower": (
            y_line
            - confidence_half_width
        ),
        "ci_upper": (
            y_line
            + confidence_half_width
        ),
    }


# 13. RUN THE COMPLETE ANALYSIS

check_input_files(
    [
        ARGO_NC,
        THERMOSTERIC_NC,
        HALOSTERIC_NC,
        FRESHWATER_NC,
    ]
)


# 13.1 Argo metrics

argo_data = read_argo_profiles()

profile_metrics = calculate_profile_metrics(
    argo_data
)


# Free the two large profile matrices before the high-resolution render.
del argo_data

gc.collect()


# 13.2 Thermosteric seasonal sector means

thermosteric_metrics = (
    calculate_gridded_product_metrics(
        THERMOSTERIC_NC,
        "thermosteric_height",
        "thermosteric_height_anomaly_cm",
        conversion_factor=100.0
    )
)


# 13.3 Halosteric seasonal sector means

halosteric_metrics = (
    calculate_gridded_product_metrics(
        HALOSTERIC_NC,
        "halosteric_height",
        "halosteric_height_anomaly_cm",
        conversion_factor=100.0
    )
)


# 13.4 Freshwater seasonal sector means

freshwater_metrics = (
    calculate_gridded_product_metrics(
        FRESHWATER_NC,
        "freshwater_content",
        "freshwater_content_0_200dbar_m",
        conversion_factor=1.0
    )
)


# 13.5 Merge all sea-season metrics

metrics = profile_metrics.merge(
    thermosteric_metrics,
    on=[
        "sea_abbreviation",
        "season"
    ],
    how="left"
)

metrics = metrics.merge(
    halosteric_metrics,
    on=[
        "sea_abbreviation",
        "season"
    ],
    how="left"
)

metrics = metrics.merge(
    freshwater_metrics,
    on=[
        "sea_abbreviation",
        "season"
    ],
    how="left"
)


sea_order_map = {
    sea_code: index
    for index, sea_code in enumerate(
        SEA_CODES
    )
}

season_order_map = {
    season_name: index
    for index, season_name in enumerate(
        SEASON_ORDER
    )
}

metrics[
    "_sea_order"
] = metrics[
    "sea_abbreviation"
].map(
    sea_order_map
)

metrics[
    "_season_order"
] = metrics[
    "season"
].map(
    season_order_map
)

metrics = metrics.sort_values(
    [
        "_sea_order",
        "_season_order"
    ]
).drop(
    columns=[
        "_sea_order",
        "_season_order"
    ]
).reset_index(
    drop=True
)


# 13.6 Place the required columns first

required_columns_first = [
    "sea_name",
    "sea_abbreviation",
    "season",
    "season_code",
    "valid_profile_count",
    "mCDW_CDW_occurrence_percent",
    "WW_core_temperature_median_degC",
    "WW_core_pressure_median_dbar",
    (
        "maximum_subsurface_temperature_"
        "200_1000dbar_median_degC"
    ),
    "thermosteric_height_anomaly_cm",
    "freshwater_content_0_200dbar_m",
    "halosteric_height_anomaly_cm",
]

remaining_columns = [
    column
    for column in metrics.columns
    if column not in required_columns_first
]

metrics = metrics[
    required_columns_first
    + remaining_columns
]


# 13.7 Save supporting tables

metrics.to_csv(
    OUTPUT_METRICS_CSV,
    index=False,
    float_format="%.6f"
)

metrics.to_excel(
    OUTPUT_METRICS_XLSX,
    index=False
)


# 14. STATISTICAL ANALYSIS FOR PANELS C AND D

PANEL_C_X = (
    "maximum_subsurface_temperature_"
    "200_1000dbar_median_degC"
)

PANEL_C_Y = (
    "thermosteric_height_anomaly_cm"
)

PANEL_D_X = (
    "freshwater_content_0_200dbar_m"
)

PANEL_D_Y = (
    "halosteric_height_anomaly_cm"
)


panel_c_statistics = calculate_regression_statistics(
    metrics,
    PANEL_C_X,
    PANEL_C_Y,
    panel_name="Figure 6c",
    slope_unit="cm per °C"
)

panel_d_statistics = calculate_regression_statistics(
    metrics,
    PANEL_D_X,
    PANEL_D_Y,
    panel_name="Figure 6d",
    slope_unit="cm per m FWC"
)


regression_statistics = pd.DataFrame(
    [
        panel_c_statistics,
        panel_d_statistics,
    ]
)

regression_statistics.to_csv(
    OUTPUT_REGRESSION_CSV,
    index=False,
    float_format="%.8f"
)


# 15. DATA-SUPPORT DIAGNOSTICS

print(
    "\n"
    + "=" * 78
)

print(
    "FIGURE 6 SUPPORT SUMMARY"
)

print(
    "=" * 78
)

print(
    "\nValid sea-season mCDW/CDW points:",
    int(
        metrics[
            "mCDW_CDW_occurrence_percent"
        ].notna().sum()
    ),
    "/ 52"
)

print(
    "Valid sea-season WW-core points:",
    int(
        metrics[
            "WW_core_temperature_median_degC"
        ].notna().sum()
    ),
    "/ 52"
)

print(
    "Valid Figure 6c pairs:",
    panel_c_statistics[
        "n"
    ]
)

print(
    "Valid Figure 6d pairs:",
    panel_d_statistics[
        "n"
    ]
)


finite_occurrence = metrics[
    "mCDW_CDW_occurrence_percent"
].dropna()

if (
    not finite_occurrence.empty
    and np.nanmin(
        finite_occurrence
    )
    >= 99.0
):

    print(
        "\nIMPORTANT THRESHOLD DIAGNOSTIC:"
    )

    print(
        "All supported mCDW/CDW occurrence values are approximately "
        "100%. This follows directly from the existing Figure 4/5 "
        "rules, because almost every valid non-AABW observation at "
        "pressure >= 200 dbar is classified as mCDW or CDW."
    )

    print(
        "No alternative threshold has been introduced in this code."
    )




# DATA PRODUCT GROUP 03: FIGURE 7 COMMON-MASK / SEASONAL SLA–EN4 PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: Fig07_SLA_EN4_total_steric_statistics.csv
# File name: Fig07_fixed_common_mask_EN4_grid.nc
# File name: Fig07_SLA_EN4_seasonal_anomalies_common_mask.nc

# FIGURE 7 — OBSERVED SLA VERSUS EN4 TOTAL STERIC HEIGHT
# Southern Ocean, 2008–2025
#
# Scientific design
# • Eight panels arranged as four seasons × two variables.
# • Left column  : satellite-observed SLA seasonal anomaly.
# • Right column : EN4 total steric-height seasonal anomaly,
#                  integrated over 0–1000 m.
# • Both columns are placed on the native EN4 1° grid.
# • The same fixed common mask is used in both columns.
# • No Argo objective mapping and no Argo grid-cell dots.
# • Argo–EN4 validation belongs in supplementary material.
#
# Seasonal anomalies
# SLA'(season) = SLA seasonal climatology - SLA all-month mean
#
# EN4 steric'(season) =
#     EN4 seasonal climatology - EN4 all-month mean
#
# Fixed common mask
# A cell is eligible only when:
#   1. OSTIA identifies it as an ocean cell.
#   2. GEBCO water depth is at least 1000 m.
#   3. SLA is available during at least 70% of the 216 months.
#   4. EN4 valid_layer_mask is valid during every month.
#
# For each seasonal row, an exact finite intersection is then
# applied to both maps. Therefore the two maps in a row always
# display precisely the same cells.
#
# Statistics reported for every season
# • Spatial Pearson correlation
# • RMSD
# • Mean bias = EN4 total steric - observed SLA
# • Number of common cells
# • Spatial coverage = seasonal common cells / fixed-mask cells × 100
#
# Statistics are calculated on the common native EN4 grid.
#
# Output directory
# /content/drive/MyDrive/SAM_Thesis/paper2/


# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "cftime": "cftime",
    "dask": "dask[array]",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing_packages,
        ]
    )
    print("Package installation completed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)



# 3. INPUT AND OUTPUT PATHS
SLA_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "SLA_Antarctic_monthly_2008_2025.nc"
)

EN4_STERIC_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
)

SIC_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"
)

GEBCO_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "GEBCO_2024_CEC.nc"
    ),
    (
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "GEBCO_2024_CF.nc"
    ),
]

GEBCO_FILE = next(
    (
        candidate
        for candidate in GEBCO_CANDIDATES
        if os.path.exists(candidate)
    ),
    GEBCO_CANDIDATES[0],
)

OUTPUT_DIRECTORY = Path(
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)



OUT_STATS_CSV = (
    OUTPUT_DIRECTORY
    / "Fig07_SLA_EN4_total_steric_statistics.csv"
)

OUT_MASK_NC = (
    OUTPUT_DIRECTORY
    / "Fig07_fixed_common_mask_EN4_grid.nc"
)

OUT_SEASONAL_NC = (
    OUTPUT_DIRECTORY
    / "Fig07_SLA_EN4_seasonal_anomalies_common_mask.nc"
)


# 4. ANALYSIS SETTINGS
LAT_MIN = -90.0
LAT_MAX = -60.0

YEAR_START = 2008
YEAR_END = 2025

MINIMUM_SLA_VALID_FRACTION = 0.70

# EN4 must be valid in all months because the product was generated
# using a complete-layer 0–1000 m requirement.
MINIMUM_EN4_VALID_FRACTION = 1.00

MINIMUM_WATER_DEPTH_M = 1000.0

SEASON_ORDER = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

SEASON_MONTHS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Autumn": [3, 4, 5],
    "Winter": [6, 7, 8],
}

SEASON_ROW_LABEL = {
    "Spring": "Spring (SON)",
    "Summer": "Summer (DJF)",
    "Autumn": "Autumn (MAM)",
    "Winter": "Winter (JJA)",
}

MERIDIANS = np.arange(-180, 181, 30)
PARALLELS = [-60, -70, -80]

DRAW_1000_M_CONTOUR = True
BATHYMETRY_CONTOUR_LEVEL = -1000.0

# Use one shared symmetric scale across both columns for direct
# visual comparison. Set False to use separate column scales.
USE_SHARED_COLOR_LIMIT = True

SAVE_DPI = 1080


# 6. DATASET HELPERS
def safe_is_datetime(dtype):
    try:
        return np.issubdtype(dtype, np.datetime64)
    except TypeError:
        return False


def open_dataset_safely(file_path, chunks=None):
    """
    Open a NetCDF file using several xarray engines.
    """
    attempts = []

    for engine in [
        None,
        "netcdf4",
        "h5netcdf",
        "scipy",
    ]:

        for decode_times in [True, False]:

            try:

                kwargs = {
                    "decode_times": decode_times,
                    "mask_and_scale": True,
                }

                if engine is not None:
                    kwargs["engine"] = engine

                if chunks is not None:
                    kwargs["chunks"] = chunks

                dataset = xr.open_dataset(
                    file_path,
                    **kwargs,
                )

                engine_name = (
                    "xarray-default"
                    if engine is None
                    else engine
                )

                print(
                    f"Opened {os.path.basename(file_path)} "
                    f"with engine={engine_name}, "
                    f"decode_times={decode_times}"
                )

                return dataset

            except Exception as error:

                attempts.append(
                    f"engine={engine}, "
                    f"decode_times={decode_times}: {error}"
                )

    raise RuntimeError(
        f"Could not open:\n{file_path}\n\n"
        + "\n".join(attempts)
    )


def detect_coordinate(dataset, coordinate_type):
    """
    Detect time, latitude, and longitude coordinate names.
    """
    names = list(dataset.coords)

    names += [
        name
        for name in dataset.variables
        if name not in dataset.coords
    ]

    aliases = {
        "time": [
            "time",
            "date",
            "datetime",
            "month",
            "juld",
            "t",
        ],
        "lat": [
            "lat",
            "latitude",
            "nav_lat",
            "y",
        ],
        "lon": [
            "lon",
            "longitude",
            "nav_lon",
            "x",
        ],
    }

    for name in names:

        variable = dataset[name]

        lower_name = name.lower()

        standard_name = str(
            variable.attrs.get(
                "standard_name",
                "",
            )
        ).lower()

        axis = str(
            variable.attrs.get(
                "axis",
                "",
            )
        ).upper()

        units = str(
            variable.attrs.get(
                "units",
                "",
            )
        ).lower()

        if coordinate_type == "time":

            if (
                lower_name in aliases["time"]
                or standard_name == "time"
                or axis == "T"
                or " since " in units
                or safe_is_datetime(variable.dtype)
            ):
                return name

        elif coordinate_type == "lat":

            if (
                lower_name in aliases["lat"]
                or standard_name == "latitude"
                or axis == "Y"
                or "degree_north" in units
                or "degrees_north" in units
            ):
                return name

        elif coordinate_type == "lon":

            if (
                lower_name in aliases["lon"]
                or standard_name == "longitude"
                or axis == "X"
                or "degree_east" in units
                or "degrees_east" in units
            ):
                return name

    raise KeyError(
        f"Could not identify the {coordinate_type} coordinate."
    )


def choose_data_variable(
    dataset,
    preferred_names,
    excluded_names=None,
):
    """
    Select the intended science variable.
    """
    if excluded_names is None:
        excluded_names = []

    variables = [
        name
        for name in dataset.data_vars
        if name not in excluded_names
    ]

    for preferred_name in preferred_names:

        for variable in variables:

            if preferred_name.lower() in variable.lower():
                return variable

    if not variables:
        raise ValueError(
            "No suitable data variable was found."
        )

    return variables[0]


def standardize_dataset_coordinates(
    dataset,
    time_name=None,
    lat_name=None,
    lon_name=None,
):
    """
    Rename coordinates to canonical time, lat, and lon names before
    any interpolation.
    """
    rename_mapping = {}

    if (
        time_name is not None
        and time_name != "time"
    ):
        rename_mapping[time_name] = "time"

    if (
        lat_name is not None
        and lat_name != "lat"
    ):
        rename_mapping[lat_name] = "lat"

    if (
        lon_name is not None
        and lon_name != "lon"
    ):
        rename_mapping[lon_name] = "lon"

    if rename_mapping:
        dataset = dataset.rename(rename_mapping)

    return dataset


def normalize_and_sort_longitude(dataset):
    """
    Normalize longitude to -180 ... 180 and remove duplicate endpoints.
    """
    if "lon" not in dataset.coords:
        return dataset

    normalized_lon = (
        (
            dataset["lon"].astype(float)
            + 180.0
        )
         % 360.0
    ) - 180.0

    dataset = dataset.assign_coords(
        lon=normalized_lon
    )

    longitude_values = dataset["lon"].values

    _, unique_indices = np.unique(
        longitude_values,
        return_index=True,
    )

    dataset = dataset.isel(
        lon=np.sort(unique_indices)
    )

    return dataset.sortby("lon")


def subset_antarctic_latitudes(
    dataset,
    southern_limit=LAT_MIN,
    northern_limit=LAT_MAX,
):
    """
    Subset and sort a one-dimensional latitude coordinate.
    """
    latitude = dataset["lat"].values

    if latitude[0] <= latitude[-1]:

        subset = dataset.sel(
            lat=slice(
                southern_limit,
                northern_limit,
            )
        )

    else:

        subset = dataset.sel(
            lat=slice(
                northern_limit,
                southern_limit,
            )
        )

    return subset.sortby("lat")


def restrict_time_period(data_array):
    """
    Restrict a time-dependent variable to 2008–2025.
    """
    return data_array.sel(
        time=slice(
            f"{YEAR_START}-01-01",
            f"{YEAR_END}-12-31",
        )
    )


def regrid_boolean_mask(
    boolean_mask,
    target_latitude,
    target_longitude,
):
    """
    Regrid a Boolean mask using numeric nearest-neighbour interpolation.
    """
    numeric_mask = boolean_mask.astype(np.float32)

    interpolated = numeric_mask.interp(
        lat=target_latitude,
        lon=target_longitude,
        method="nearest",
    )

    return interpolated.fillna(0.0) >= 0.5


def convert_height_to_cm(data_array):
    """
    Convert metres to centimetres when required.
    """
    units = str(
        data_array.attrs.get(
            "units",
            "",
        )
    ).strip().lower()

    if units in [
        "cm",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
    ]:

        output = data_array.copy()
        output.attrs["units"] = "cm"
        return output

    if units in [
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    ]:

        output = data_array * 100.0
        output.attrs.update(data_array.attrs)
        output.attrs["units"] = "cm"
        return output

    # Conservative magnitude-based fallback.
    sample = data_array.isel(
        time=slice(
            0,
            min(
                12,
                data_array.sizes.get("time", 1),
            ),
        )
    )

    values = np.asarray(
        sample.values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if (
        values.size > 0
        and np.nanpercentile(
            np.abs(values),
            99,
        ) < 1.0
    ):

        output = data_array * 100.0
        output.attrs.update(data_array.attrs)
        output.attrs["units"] = "cm"
        return output

    output = data_array.copy()
    output.attrs["units"] = units if units else "unknown"
    return output


def seasonal_climatology(data_array):
    """
    Calculate austral seasonal climatological means.
    """
    output = {}

    for season in SEASON_ORDER:

        selected = data_array.where(
            data_array["time"].dt.month.isin(
                SEASON_MONTHS[season]
            ),
            drop=True,
        )

        output[season] = selected.mean(
            dim="time",
            skipna=True,
        )

    return output


def symmetric_limit(fields):
    """
    Calculate a robust, rounded symmetric color limit.
    """
    collected = []

    for field in fields:

        values = np.asarray(
            field.values,
            dtype=float,
        )

        values = values[np.isfinite(values)]

        if values.size > 0:
            collected.append(values)

    if not collected:
        return 1.0

    all_values = np.concatenate(collected)

    robust_maximum = float(
        np.nanpercentile(
            np.abs(all_values),
            99,
        )
    )

    if robust_maximum <= 2.0:
        interval = 0.5

    elif robust_maximum <= 5.0:
        interval = 1.0

    elif robust_maximum <= 15.0:
        interval = 2.0

    else:
        interval = 5.0

    return float(
        interval
        * np.ceil(
            robust_maximum / interval
        )
    )


def coordinate_mesh(data_array):
    """
    Return two-dimensional longitude and latitude arrays.
    """
    return np.meshgrid(
        data_array["lon"].values,
        data_array["lat"].values,
    )


def format_longitude(longitude):
    """
    Format longitude labels for the polar-map rim.
    """
    value = int(longitude)

    if value == 0:
        return "0°"

    if abs(value) == 180:
        return "180°"

    if value < 0:
        return f"{abs(value)}°W"

    return f"{value}°E"


# 7. STATISTICAL HELPERS
def unweighted_spatial_statistics(
    observed_sla,
    en4_steric,
    fixed_mask_count,
):
    """
    Calculate requested statistics on the exact common finite cells.

    Bias sign:
        EN4 total steric minus observed SLA
    """
    observed_values = np.asarray(
        observed_sla.values,
        dtype=float,
    )

    steric_values = np.asarray(
        en4_steric.values,
        dtype=float,
    )

    valid = (
        np.isfinite(observed_values)
        & np.isfinite(steric_values)
    )

    n_common = int(valid.sum())

    if n_common < 2:

        return {
            "spatial_correlation_r": np.nan,
            "RMSD_cm": np.nan,
            "mean_bias_EN4_minus_SLA_cm": np.nan,
            "n_common_cells": n_common,
            "spatial_coverage_percent": (
                100.0 * n_common / fixed_mask_count
                if fixed_mask_count > 0
                else np.nan
            ),
        }

    observed = observed_values[valid]
    modeled = steric_values[valid]

    difference = modeled - observed

    return {
        "spatial_correlation_r": float(
            np.corrcoef(
                observed,
                modeled,
            )[0, 1]
        ),
        "RMSD_cm": float(
            np.sqrt(
                np.mean(
                    difference ** 2
                )
            )
        ),
        "mean_bias_EN4_minus_SLA_cm": float(
            np.mean(difference)
        ),
        "n_common_cells": n_common,
        "spatial_coverage_percent": float(
            100.0
            * n_common
            / fixed_mask_count
        ) if fixed_mask_count > 0 else np.nan,
    }

# 8. VERIFY INPUT FILES
for file_path in [
    SLA_FILE,
    EN4_STERIC_FILE,
    SIC_FILE,
    GEBCO_FILE,
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Input file not found:\n{file_path}"
        )

print("\nUsing EN4 steric file:")
print(EN4_STERIC_FILE)

print("\nUsing GEBCO file:")
print(GEBCO_FILE)


# 9. OPEN INPUT DATASETS
ds_sla = open_dataset_safely(
    SLA_FILE,
    chunks="auto",
)

ds_en4 = open_dataset_safely(
    EN4_STERIC_FILE,
    chunks="auto",
)

ds_sic = open_dataset_safely(
    SIC_FILE,
    chunks="auto",
)

ds_gebco = open_dataset_safely(
    GEBCO_FILE,
    chunks="auto",
)


# 10. DETECT VARIABLES AND COORDINATES
sla_time_name = detect_coordinate(
    ds_sla,
    "time",
)
sla_lat_name = detect_coordinate(
    ds_sla,
    "lat",
)
sla_lon_name = detect_coordinate(
    ds_sla,
    "lon",
)

en4_time_name = detect_coordinate(
    ds_en4,
    "time",
)
en4_lat_name = detect_coordinate(
    ds_en4,
    "lat",
)
en4_lon_name = detect_coordinate(
    ds_en4,
    "lon",
)

sic_time_name = detect_coordinate(
    ds_sic,
    "time",
)
sic_lat_name = detect_coordinate(
    ds_sic,
    "lat",
)
sic_lon_name = detect_coordinate(
    ds_sic,
    "lon",
)

gebco_lat_name = detect_coordinate(
    ds_gebco,
    "lat",
)
gebco_lon_name = detect_coordinate(
    ds_gebco,
    "lon",
)

sla_variable_name = choose_data_variable(
    ds_sla,
    preferred_names=[
        "sla",
        "sea_level",
        "adt",
        "ssh",
    ],
)

en4_steric_variable_name = choose_data_variable(
    ds_en4,
    preferred_names=[
        "total_steric_height",
        "total_steric",
        "steric",
    ],
    excluded_names=[
        "valid_layer_mask",
        "gebco_water_depth",
    ],
)

sic_variable_name = choose_data_variable(
    ds_sic,
    preferred_names=[
        "SIF",
        "sea_ice_fraction",
        "ice_fraction",
        "sic",
    ],
)

gebco_variable_name = choose_data_variable(
    ds_gebco,
    preferred_names=[
        "elevation",
        "bathymetry",
        "depth",
        "z",
    ],
)

print("\nDetected variables:")
print(f"Observed SLA       : {sla_variable_name}")
print(f"EN4 total steric   : {en4_steric_variable_name}")
print(f"OSTIA sea ice      : {sic_variable_name}")
print(f"GEBCO bathymetry   : {gebco_variable_name}")


# 11. STANDARDIZE ALL COORDINATES
ds_sla = standardize_dataset_coordinates(
    ds_sla,
    time_name=sla_time_name,
    lat_name=sla_lat_name,
    lon_name=sla_lon_name,
)

ds_en4 = standardize_dataset_coordinates(
    ds_en4,
    time_name=en4_time_name,
    lat_name=en4_lat_name,
    lon_name=en4_lon_name,
)

ds_sic = standardize_dataset_coordinates(
    ds_sic,
    time_name=sic_time_name,
    lat_name=sic_lat_name,
    lon_name=sic_lon_name,
)

ds_gebco = standardize_dataset_coordinates(
    ds_gebco,
    lat_name=gebco_lat_name,
    lon_name=gebco_lon_name,
)

ds_sla = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_sla)
)

ds_en4 = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_en4)
)

ds_sic = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_sic)
)

ds_gebco = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_gebco),
    southern_limit=-90.0,
    northern_limit=-55.0,
)


# 12. PREPARE SCIENCE VARIABLES
sla = convert_height_to_cm(
    restrict_time_period(
        ds_sla[sla_variable_name]
    )
)

en4_steric = convert_height_to_cm(
    restrict_time_period(
        ds_en4[en4_steric_variable_name]
    )
)

sic = restrict_time_period(
    ds_sic[sic_variable_name]
)

gebco = ds_gebco[gebco_variable_name]

if "valid_layer_mask" in ds_en4.data_vars:

    en4_valid_layer_mask = restrict_time_period(
        ds_en4["valid_layer_mask"]
    )

else:

    en4_valid_layer_mask = xr.where(
        np.isfinite(en4_steric),
        1,
        0,
    )

print("\nUnits after conversion:")
print("Observed SLA     :", sla.attrs.get("units"))
print("EN4 total steric:", en4_steric.attrs.get("units"))


# 13. ALIGN THE COMMON MONTHLY PERIOD
common_times = np.intersect1d(
    pd.to_datetime(
        sla["time"].values
    ),
    pd.to_datetime(
        en4_steric["time"].values
    ),
)

if common_times.size == 0:

    raise ValueError(
        "No common SLA and EN4 steric months were found."
    )

sla = sla.sel(time=common_times)
en4_steric = en4_steric.sel(time=common_times)
en4_valid_layer_mask = en4_valid_layer_mask.sel(
    time=common_times
)

number_of_months = len(common_times)

print("\nCommon analysis period:")
print(pd.to_datetime(common_times[0]))
print("to")
print(pd.to_datetime(common_times[-1]))
print(f"Months = {number_of_months}")


# 14. OSTIA STATIC OCEAN MASK
sic_reference = sic.isel(time=0)

sic_ocean_native = xr.where(
    np.isfinite(sic_reference),
    True,
    False,
)

sic_ocean_native = (
    sic_ocean_native
    .astype(np.float32)
    .compute()
    >= 0.5
)

sic_ocean_on_sla = regrid_boolean_mask(
    boolean_mask=sic_ocean_native,
    target_latitude=sla["lat"],
    target_longitude=sla["lon"],
).compute()

sic_ocean_on_en4 = regrid_boolean_mask(
    boolean_mask=sic_ocean_native,
    target_latitude=en4_steric["lat"],
    target_longitude=en4_steric["lon"],
).compute()

print("\nOSTIA static ocean masks created.")


# 15. GEBCO DEPTH MASK ON THE EN4 GRID
gebco_on_en4 = gebco.interp(
    lat=en4_steric["lat"],
    lon=en4_steric["lon"],
    method="nearest",
).compute()

gebco_values_on_en4 = np.asarray(
    gebco_on_en4.values,
    dtype=float,
)

# GEBCO elevation is negative below sea level.
gebco_water_depth_on_en4 = xr.DataArray(
    np.where(
        gebco_values_on_en4 < 0.0,
        -gebco_values_on_en4,
        np.nan,
    ).astype(np.float32),
    dims=("lat", "lon"),
    coords={
        "lat": en4_steric["lat"],
        "lon": en4_steric["lon"],
    },
    name="gebco_water_depth",
)

deep_ocean_mask_en4 = (
    np.isfinite(gebco_water_depth_on_en4)
    & (
        gebco_water_depth_on_en4
        >= MINIMUM_WATER_DEPTH_M
    )
)


# 16. BUILD THE FIXED COMMON MASK
# SLA core validity on its native high-resolution grid.
sla_valid_fraction_native = (
    sla.notnull().sum(dim="time")
    / number_of_months
)

sla_core_mask_native = (
    sic_ocean_on_sla
    & (
        sla_valid_fraction_native
        >= MINIMUM_SLA_VALID_FRACTION
    )
)

# Transfer SLA validity to the EN4 grid.
sla_core_mask_on_en4 = regrid_boolean_mask(
    boolean_mask=sla_core_mask_native,
    target_latitude=en4_steric["lat"],
    target_longitude=en4_steric["lon"],
).compute()

# EN4 valid-layer fraction on the native EN4 grid.
en4_valid_fraction = (
    (
        en4_valid_layer_mask == 1
    )
    .sum(dim="time")
    / number_of_months
)

en4_fixed_valid_mask = (
    en4_valid_fraction
    >= (
        MINIMUM_EN4_VALID_FRACTION
        - 1.0e-10
    )
)

fixed_common_mask = (
    sic_ocean_on_en4
    & deep_ocean_mask_en4
    & sla_core_mask_on_en4
    & en4_fixed_valid_mask
)

fixed_common_mask = fixed_common_mask.compute()

fixed_common_cell_count = int(
    fixed_common_mask.sum().values
)

if fixed_common_cell_count == 0:

    raise RuntimeError(
        "The fixed SLA–EN4 common mask contains no cells."
    )

print("\nFixed common mask:")
print(
    f"SLA threshold = "
    f"{MINIMUM_SLA_VALID_FRACTION * 100:.0f}%"
)
print(
    f"EN4 threshold = "
    f"{MINIMUM_EN4_VALID_FRACTION * 100:.0f}%"
)
print(
    f"Minimum water depth = "
    f"{MINIMUM_WATER_DEPTH_M:.0f} m"
)
print(
    f"Fixed common cells = "
    f"{fixed_common_cell_count:,}"
)


# 17. CALCULATE SEASONAL ANOMALIES
# SLA anomalies are first calculated on the native SLA grid.
sla_all_month_mean_native = sla.mean(
    dim="time",
    skipna=True,
)

sla_seasonal_mean_native = seasonal_climatology(
    sla
)

# EN4 anomalies remain on the native EN4 grid.
en4_all_month_mean = en4_steric.mean(
    dim="time",
    skipna=True,
)

en4_seasonal_mean = seasonal_climatology(
    en4_steric
)

sla_anomaly_common = {}
en4_anomaly_common = {}
seasonal_common_mask = {}
statistics_rows = []

for season in SEASON_ORDER:

    sla_anomaly_native = (
        sla_seasonal_mean_native[season]
        - sla_all_month_mean_native
    ).where(
        sla_core_mask_native
    )

    # Regrid the already calculated SLA seasonal anomaly to EN4.
    sla_anomaly_en4 = sla_anomaly_native.interp(
        lat=en4_steric["lat"],
        lon=en4_steric["lon"],
        method="linear",
    )

    en4_anomaly = (
        en4_seasonal_mean[season]
        - en4_all_month_mean
    )

    # Exact finite seasonal intersection, applied to both columns.
    row_common_mask = (
        fixed_common_mask
        & np.isfinite(sla_anomaly_en4)
        & np.isfinite(en4_anomaly)
    )

    row_common_mask = row_common_mask.compute()

    sla_field = sla_anomaly_en4.where(
        row_common_mask
    ).compute()

    en4_field = en4_anomaly.where(
        row_common_mask
    ).compute()

    sla_anomaly_common[season] = sla_field
    en4_anomaly_common[season] = en4_field
    seasonal_common_mask[season] = row_common_mask

    statistics = unweighted_spatial_statistics(
        observed_sla=sla_field,
        en4_steric=en4_field,
        fixed_mask_count=fixed_common_cell_count,
    )

    statistics_rows.append(
        {
            "season": season,
            **statistics,
            "bias_definition": (
                "EN4 total steric minus observed SLA"
            ),
            "statistics_grid": (
                "Native EN4 grid"
            ),
            "fixed_common_mask_cells": (
                fixed_common_cell_count
            ),
        }
    )

statistics_table = pd.DataFrame(
    statistics_rows
)

statistics_table.to_csv(
    OUT_STATS_CSV,
    index=False,
)

print("\nSeasonal statistics:")
print(
    statistics_table.to_string(
        index=False
    )
)


# 18. SAVE THE FIXED AND SEASONAL COMMON MASKS
mask_output = xr.Dataset(
    {
        "fixed_common_mask": (
            fixed_common_mask.astype(np.int8)
        ),
        "sla_valid_fraction": (
            sla_valid_fraction_native.interp(
                lat=en4_steric["lat"],
                lon=en4_steric["lon"],
                method="linear",
            )
        ),
        "en4_valid_fraction": (
            en4_valid_fraction
        ),
        "gebco_water_depth": (
            gebco_water_depth_on_en4
        ),
    }
)

mask_output["fixed_common_mask"].attrs.update({
    "long_name": "fixed common SLA–EN4 analysis mask",
    "flag_values": np.array([0, 1], dtype=np.int8),
    "flag_meanings": "excluded included",
})

mask_output.attrs.update({
    "title": "Figure 7 fixed SLA–EN4 common mask",
    "SLA_valid_fraction_threshold": (
        MINIMUM_SLA_VALID_FRACTION
    ),
    "EN4_valid_fraction_threshold": (
        MINIMUM_EN4_VALID_FRACTION
    ),
    "minimum_GEBCO_water_depth_m": (
        MINIMUM_WATER_DEPTH_M
    ),
})

mask_output.rename(
    {
        "lat": "latitude",
        "lon": "longitude",
    }
).to_netcdf(
    OUT_MASK_NC,
    encoding={
        "fixed_common_mask": {
            "zlib": True,
            "complevel": 4,
        },
        "sla_valid_fraction": {
            "zlib": True,
            "complevel": 4,
        },
        "en4_valid_fraction": {
            "zlib": True,
            "complevel": 4,
        },
        "gebco_water_depth": {
            "zlib": True,
            "complevel": 4,
        },
    },
)


# 19. SAVE SEASONAL FIELDS USED IN THE FIGURE

season_coordinate = xr.DataArray(
    SEASON_ORDER,
    dims="season",
    name="season",
)

sla_stack = xr.concat(
    [
        sla_anomaly_common[season]
        for season in SEASON_ORDER
    ],
    dim=season_coordinate,
)

en4_stack = xr.concat(
    [
        en4_anomaly_common[season]
        for season in SEASON_ORDER
    ],
    dim=season_coordinate,
)

seasonal_mask_stack = xr.concat(
    [
        seasonal_common_mask[season].astype(np.int8)
        for season in SEASON_ORDER
    ],
    dim=season_coordinate,
)

seasonal_output = xr.Dataset(
    {
        "observed_SLA_anomaly": sla_stack,
        "EN4_total_steric_anomaly": en4_stack,
        "seasonal_common_mask": seasonal_mask_stack,
    }
)

seasonal_output[
    "observed_SLA_anomaly"
].attrs.update({
    "long_name": (
        "seasonal observed sea-level anomaly relative "
        "to the 2008-2025 all-month mean"
    ),
    "units": "cm",
})

seasonal_output[
    "EN4_total_steric_anomaly"
].attrs.update({
    "long_name": (
        "seasonal EN4 0-1000 m total steric-height anomaly "
        "relative to the 2008-2025 all-month mean"
    ),
    "units": "cm",
})

seasonal_output[
    "seasonal_common_mask"
].attrs.update({
    "long_name": (
        "season-specific exact common finite mask"
    ),
    "flag_values": np.array([0, 1], dtype=np.int8),
    "flag_meanings": "excluded included",
})

seasonal_output.attrs.update({
    "title": (
        "Figure 7 observed SLA and EN4 total steric "
        "seasonal anomalies"
    ),
    "bias_definition": (
        "EN4 total steric minus observed SLA"
    ),
})

seasonal_output.rename(
    {
        "lat": "latitude",
        "lon": "longitude",
    }
).to_netcdf(
    OUT_SEASONAL_NC,
    encoding={
        "observed_SLA_anomaly": {
            "zlib": True,
            "complevel": 4,
        },
        "EN4_total_steric_anomaly": {
            "zlib": True,
            "complevel": 4,
        },
        "seasonal_common_mask": {
            "zlib": True,
            "complevel": 4,
        },
    },
)





# DATA PRODUCT GROUP 04: FIGURE 8 EN4 STERIC DECOMPOSITION SUPPORT PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: Fig08_EN4_steric_decomposition_closure_metrics.csv
# File name: Fig08_EN4_steric_decomposition_seasonal_anomalies.nc

# FIGURE 8 — EN4 SEASONAL STERIC DECOMPOSITION
# Southern Ocean, 2008–2025
#
# Inputs
# EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc
# EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc
# EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc
# GEBCO_2024_CEC.nc
#
# Output
# /content/drive/MyDrive/SAM_Thesis/paper2/
#
# Main calculation
# For each product and each season:
#
# seasonal anomaly = seasonal climatology - all-month mean
#
# where the all-month mean is calculated from the full 2008–2025
# monthly series.
#
# Closure residual
# epsilon = total - (thermo + halo)
#
# Reported closure metrics are calculated from all four seasonal
# residual fields combined, using the common finite mask.
#
# Columns
# Column 1: Total steric height anomaly
# Column 2: Thermosteric height anomaly
# Column 3: Halosteric height anomaly
#
# One symmetric colour scale is used per column.

# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "cftime": "cftime",
    "dask": "dask[array]",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing_packages,
        ]
    )
    print("Package installation completed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)



# 3. INPUT AND OUTPUT PATHS
TOTAL_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
)

THERMO_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc"
)

HALO_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc"
)

GEBCO_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "GEBCO_2024_CEC.nc"
    ),
    (
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "GEBCO_2024_CF.nc"
    ),
]

GEBCO_FILE = next(
    (
        candidate
        for candidate in GEBCO_CANDIDATES
        if os.path.exists(candidate)
    ),
    GEBCO_CANDIDATES[0],
)

OUTPUT_DIRECTORY = Path(
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)



OUT_CLOSURE_CSV = (
    OUTPUT_DIRECTORY
    / "Fig08_EN4_steric_decomposition_closure_metrics.csv"
)

OUT_SEASONAL_NC = (
    OUTPUT_DIRECTORY
    / "Fig08_EN4_steric_decomposition_seasonal_anomalies.nc"
)

OUT_SUMMARY_TXT = (
    OUTPUT_DIRECTORY
    / "Fig08_EN4_steric_decomposition_summary.txt"
)


# 4. ANALYSIS SETTINGS
LAT_MIN = -90.0
LAT_MAX = -60.0

YEAR_START = 2008
YEAR_END = 2025

SEASON_ORDER = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

SEASON_MONTHS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Autumn": [3, 4, 5],
    "Winter": [6, 7, 8],
}

SEASON_ROW_LABEL = {
    "Spring": "Spring",
    "Summer": "Summer",
    "Autumn": "Autumn",
    "Winter": "Winter",
}

COLUMN_LABELS = [
    "Total steric height\nanomaly (cm)",
    "Thermosteric height\nanomaly (cm)",
    "Halosteric height\nanomaly (cm)",
]

PANEL_LETTERS = [
    ["(a)", "(b)", "(c)"],
    ["(d)", "(e)", "(f)"],
    ["(g)", "(h)", "(i)"],
    ["(j)", "(k)", "(l)"],
]

MERIDIANS = np.arange(-180, 181, 30)
PARALLELS = [-60, -70, -80]

DRAW_1000_M_CONTOUR = True
BATHYMETRY_CONTOUR_LEVEL = -1000.0

CLOSURE_TOLERANCE_CM = 1.0

SAVE_DPI = 1080


# 6. GENERIC HELPERS
def safe_is_datetime(dtype):
    try:
        return np.issubdtype(dtype, np.datetime64)
    except TypeError:
        return False


def open_dataset_safely(file_path, chunks=None):
    """
    Open a NetCDF file using several possible engines.
    """
    attempts = []

    for engine in [
        None,
        "netcdf4",
        "h5netcdf",
        "scipy",
    ]:

        for decode_times in [True, False]:

            try:
                kwargs = {
                    "decode_times": decode_times,
                    "mask_and_scale": True,
                }

                if engine is not None:
                    kwargs["engine"] = engine

                if chunks is not None:
                    kwargs["chunks"] = chunks

                dataset = xr.open_dataset(
                    file_path,
                    **kwargs,
                )

                engine_name = (
                    "xarray-default"
                    if engine is None
                    else engine
                )

                print(
                    f"Opened {os.path.basename(file_path)} "
                    f"with engine={engine_name}, "
                    f"decode_times={decode_times}"
                )

                return dataset

            except Exception as error:
                attempts.append(
                    f"engine={engine}, decode_times={decode_times}: {error}"
                )

    raise RuntimeError(
        f"Could not open:\n{file_path}\n\n"
        + "\n".join(attempts)
    )


def detect_coordinate(dataset, coordinate_type):
    """
    Detect time, latitude, and longitude coordinate names.
    """
    names = list(dataset.coords) + [
        name
        for name in dataset.variables
        if name not in dataset.coords
    ]

    aliases = {
        "time": [
            "time",
            "date",
            "datetime",
            "month",
            "juld",
            "t",
        ],
        "lat": [
            "lat",
            "latitude",
            "nav_lat",
            "y",
        ],
        "lon": [
            "lon",
            "longitude",
            "nav_lon",
            "x",
        ],
    }

    for name in names:

        variable = dataset[name]

        lower_name = name.lower()
        standard_name = str(
            variable.attrs.get("standard_name", "")
        ).lower()
        axis = str(
            variable.attrs.get("axis", "")
        ).upper()
        units = str(
            variable.attrs.get("units", "")
        ).lower()

        if coordinate_type == "time":
            if (
                lower_name in aliases["time"]
                or standard_name == "time"
                or axis == "T"
                or " since " in units
                or safe_is_datetime(variable.dtype)
            ):
                return name

        elif coordinate_type == "lat":
            if (
                lower_name in aliases["lat"]
                or standard_name == "latitude"
                or axis == "Y"
                or "degree_north" in units
                or "degrees_north" in units
            ):
                return name

        elif coordinate_type == "lon":
            if (
                lower_name in aliases["lon"]
                or standard_name == "longitude"
                or axis == "X"
                or "degree_east" in units
                or "degrees_east" in units
            ):
                return name

    raise KeyError(
        f"Could not identify the {coordinate_type} coordinate."
    )


def choose_data_variable(
    dataset,
    preferred_names,
    excluded_names=None,
):
    """
    Select the intended science variable.
    """
    if excluded_names is None:
        excluded_names = []

    variables = [
        name
        for name in dataset.data_vars
        if name not in excluded_names
    ]

    for preferred_name in preferred_names:
        for variable in variables:
            if preferred_name.lower() in variable.lower():
                return variable

    if not variables:
        raise ValueError(
            "No suitable science variable was found."
        )

    return variables[0]


def standardize_dataset_coordinates(
    dataset,
    time_name=None,
    lat_name=None,
    lon_name=None,
):
    """
    Rename coordinates to canonical time/lat/lon names.
    """
    rename_mapping = {}

    if time_name is not None and time_name != "time":
        rename_mapping[time_name] = "time"

    if lat_name is not None and lat_name != "lat":
        rename_mapping[lat_name] = "lat"

    if lon_name is not None and lon_name != "lon":
        rename_mapping[lon_name] = "lon"

    if rename_mapping:
        dataset = dataset.rename(rename_mapping)

    return dataset


def normalize_and_sort_longitude(dataset):
    """
    Normalize longitude to -180 ... 180 and remove duplicates.
    """
    if "lon" not in dataset.coords:
        return dataset

    normalized_lon = (
        (dataset["lon"].astype(float) + 180.0) % 360.0
    ) - 180.0

    dataset = dataset.assign_coords(lon=normalized_lon)

    longitude_values = dataset["lon"].values
    _, unique_indices = np.unique(
        longitude_values,
        return_index=True,
    )

    dataset = dataset.isel(
        lon=np.sort(unique_indices)
    )

    return dataset.sortby("lon")


def subset_antarctic_latitudes(
    dataset,
    southern_limit=LAT_MIN,
    northern_limit=LAT_MAX,
):
    """
    Subset a dataset to Antarctic latitudes and sort ascending.
    """
    latitude = dataset["lat"].values

    if latitude[0] <= latitude[-1]:
        subset = dataset.sel(
            lat=slice(
                southern_limit,
                northern_limit,
            )
        )
    else:
        subset = dataset.sel(
            lat=slice(
                northern_limit,
                southern_limit,
            )
        )

    return subset.sortby("lat")


def restrict_time_period(data_array):
    """
    Restrict the analysis period to 2008–2025.
    """
    return data_array.sel(
        time=slice(
            f"{YEAR_START}-01-01",
            f"{YEAR_END}-12-31",
        )
    )


def convert_height_to_cm(data_array):
    """
    Convert metres to centimetres when needed.
    """
    units = str(
        data_array.attrs.get("units", "")
    ).strip().lower()

    if units in [
        "cm",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
    ]:
        output = data_array.copy()
        output.attrs["units"] = "cm"
        return output

    if units in [
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    ]:
        output = data_array * 100.0
        output.attrs.update(data_array.attrs)
        output.attrs["units"] = "cm"
        return output

    sample = data_array.isel(
        time=slice(
            0,
            min(
                12,
                data_array.sizes.get("time", 1),
            )
        )
    )

    sample_values = np.asarray(
        sample.values,
        dtype=float,
    )
    sample_values = sample_values[
        np.isfinite(sample_values)
    ]

    if (
        sample_values.size > 0
        and np.nanpercentile(
            np.abs(sample_values),
            99,
        ) < 1.0
    ):
        output = data_array * 100.0
        output.attrs.update(data_array.attrs)
        output.attrs["units"] = "cm"
        return output

    output = data_array.copy()
    output.attrs["units"] = units if units else "unknown"
    return output


def seasonal_climatology(data_array):
    """
    Calculate austral seasonal climatological means.
    """
    output = {}

    for season in SEASON_ORDER:
        selected = data_array.where(
            data_array["time"].dt.month.isin(
                SEASON_MONTHS[season]
            ),
            drop=True,
        )

        output[season] = selected.mean(
            dim="time",
            skipna=True,
        )

    return output


def symmetric_limit(fields):
    """
    Calculate a robust rounded symmetric colour limit.
    """
    collected = []

    for field in fields:
        values = np.asarray(
            field.values,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size > 0:
            collected.append(values)

    if not collected:
        return 1.0

    all_values = np.concatenate(collected)

    robust_maximum = float(
        np.nanpercentile(
            np.abs(all_values),
            99,
        )
    )

    if robust_maximum <= 2.0:
        interval = 0.5
    elif robust_maximum <= 5.0:
        interval = 1.0
    elif robust_maximum <= 15.0:
        interval = 2.0
    else:
        interval = 5.0

    return float(
        interval * np.ceil(robust_maximum / interval)
    )


def coordinate_mesh(data_array):
    """
    Return 2-D longitude and latitude arrays.
    """
    return np.meshgrid(
        data_array["lon"].values,
        data_array["lat"].values,
    )


def format_longitude(longitude):
    """
    Format longitude for circular-map rim labels.
    """
    value = int(longitude)

    if value == 0:
        return "0°"

    if abs(value) == 180:
        return "180°"

    if value < 0:
        return f"{abs(value)}°W"

    return f"{value}°E"


# 7. VERIFY INPUT FILES
for file_path in [
    TOTAL_FILE,
    THERMO_FILE,
    HALO_FILE,
    GEBCO_FILE,
]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Input file not found:\n{file_path}"
        )

print("\nUsing GEBCO file:")
print(GEBCO_FILE)


# 8. OPEN DATASETS
ds_total = open_dataset_safely(
    TOTAL_FILE,
    chunks="auto",
)

ds_thermo = open_dataset_safely(
    THERMO_FILE,
    chunks="auto",
)

ds_halo = open_dataset_safely(
    HALO_FILE,
    chunks="auto",
)

ds_gebco = open_dataset_safely(
    GEBCO_FILE,
    chunks="auto",
)


# 9. DETECT COORDINATES AND VARIABLES
total_time_name = detect_coordinate(ds_total, "time")
total_lat_name = detect_coordinate(ds_total, "lat")
total_lon_name = detect_coordinate(ds_total, "lon")

thermo_time_name = detect_coordinate(ds_thermo, "time")
thermo_lat_name = detect_coordinate(ds_thermo, "lat")
thermo_lon_name = detect_coordinate(ds_thermo, "lon")

halo_time_name = detect_coordinate(ds_halo, "time")
halo_lat_name = detect_coordinate(ds_halo, "lat")
halo_lon_name = detect_coordinate(ds_halo, "lon")

gebco_lat_name = detect_coordinate(ds_gebco, "lat")
gebco_lon_name = detect_coordinate(ds_gebco, "lon")

total_variable_name = choose_data_variable(
    ds_total,
    preferred_names=[
        "total_steric_height",
        "total_steric",
        "steric",
    ],
    excluded_names=[
        "valid_layer_mask",
        "gebco_water_depth",
    ],
)

thermo_variable_name = choose_data_variable(
    ds_thermo,
    preferred_names=[
        "thermosteric_height",
        "thermosteric",
        "steric",
    ],
    excluded_names=[
        "valid_layer_mask",
        "gebco_water_depth",
    ],
)

halo_variable_name = choose_data_variable(
    ds_halo,
    preferred_names=[
        "halosteric_height",
        "halosteric",
        "steric",
    ],
    excluded_names=[
        "valid_layer_mask",
        "gebco_water_depth",
    ],
)

gebco_variable_name = choose_data_variable(
    ds_gebco,
    preferred_names=[
        "elevation",
        "bathymetry",
        "depth",
        "z",
    ],
)

print("\nDetected variables:")
print(f"Total steric : {total_variable_name}")
print(f"Thermosteric : {thermo_variable_name}")
print(f"Halosteric   : {halo_variable_name}")
print(f"Bathymetry   : {gebco_variable_name}")


# 10. STANDARDIZE COORDINATES
ds_total = standardize_dataset_coordinates(
    ds_total,
    time_name=total_time_name,
    lat_name=total_lat_name,
    lon_name=total_lon_name,
)

ds_thermo = standardize_dataset_coordinates(
    ds_thermo,
    time_name=thermo_time_name,
    lat_name=thermo_lat_name,
    lon_name=thermo_lon_name,
)

ds_halo = standardize_dataset_coordinates(
    ds_halo,
    time_name=halo_time_name,
    lat_name=halo_lat_name,
    lon_name=halo_lon_name,
)

ds_gebco = standardize_dataset_coordinates(
    ds_gebco,
    lat_name=gebco_lat_name,
    lon_name=gebco_lon_name,
)

ds_total = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_total)
)

ds_thermo = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_thermo)
)

ds_halo = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_halo)
)

ds_gebco = subset_antarctic_latitudes(
    normalize_and_sort_longitude(ds_gebco),
    southern_limit=-90.0,
    northern_limit=-55.0,
)


# 11. PREPARE SCIENCE VARIABLES
total = convert_height_to_cm(
    restrict_time_period(ds_total[total_variable_name])
)

thermo = convert_height_to_cm(
    restrict_time_period(ds_thermo[thermo_variable_name])
)

halo = convert_height_to_cm(
    restrict_time_period(ds_halo[halo_variable_name])
)

gebco = ds_gebco[gebco_variable_name]

print("\nUnits after conversion:")
print("Total steric :", total.attrs.get("units"))
print("Thermosteric :", thermo.attrs.get("units"))
print("Halosteric   :", halo.attrs.get("units"))


# 12. ALIGN THE COMMON MONTHLY PERIOD
common_times = np.intersect1d(
    pd.to_datetime(total["time"].values),
    np.intersect1d(
        pd.to_datetime(thermo["time"].values),
        pd.to_datetime(halo["time"].values),
    ),
)

if common_times.size == 0:
    raise ValueError(
        "No common monthly period among total, thermo, and halo files."
    )

total = total.sel(time=common_times)
thermo = thermo.sel(time=common_times)
halo = halo.sel(time=common_times)

number_of_months = len(common_times)

print("\nCommon analysis period:")
print(pd.to_datetime(common_times[0]))
print("to")
print(pd.to_datetime(common_times[-1]))
print(f"Months = {number_of_months}")


# 13. BUILD A COMMON STATIC MASK
# Use the intersection of finite monthly values across all three
# products during the full period.
total_valid_fraction = (
    total.notnull().sum(dim="time")
    / number_of_months
)

thermo_valid_fraction = (
    thermo.notnull().sum(dim="time")
    / number_of_months
)

halo_valid_fraction = (
    halo.notnull().sum(dim="time")
    / number_of_months
)

static_common_mask = (
    (total_valid_fraction >= 1.0 - 1e-10)
    & (thermo_valid_fraction >= 1.0 - 1e-10)
    & (halo_valid_fraction >= 1.0 - 1e-10)
).compute()

static_common_cell_count = int(
    static_common_mask.sum().values
)

if static_common_cell_count == 0:
    raise RuntimeError(
        "The common static mask contains no valid cells."
    )

print("\nStatic common mask cells:", f"{static_common_cell_count:,}")


# 14. CALCULATE SEASONAL ANOMALIES
total_all_month_mean = total.mean(
    dim="time",
    skipna=True,
)

thermo_all_month_mean = thermo.mean(
    dim="time",
    skipna=True,
)

halo_all_month_mean = halo.mean(
    dim="time",
    skipna=True,
)

total_season_mean = seasonal_climatology(total)
thermo_season_mean = seasonal_climatology(thermo)
halo_season_mean = seasonal_climatology(halo)

total_anomaly = {}
thermo_anomaly = {}
halo_anomaly = {}
seasonal_mask = {}
residual_field = {}
no_data_mask = {}

for season in SEASON_ORDER:

    total_field = (
        total_season_mean[season]
        - total_all_month_mean
    )

    thermo_field = (
        thermo_season_mean[season]
        - thermo_all_month_mean
    )

    halo_field = (
        halo_season_mean[season]
        - halo_all_month_mean
    )

    row_mask = (
        static_common_mask
        & np.isfinite(total_field)
        & np.isfinite(thermo_field)
        & np.isfinite(halo_field)
    ).compute()

    total_anomaly[season] = total_field.where(
        row_mask
    ).compute()

    thermo_anomaly[season] = thermo_field.where(
        row_mask
    ).compute()

    halo_anomaly[season] = halo_field.where(
        row_mask
    ).compute()

    seasonal_mask[season] = row_mask

    residual = (
        total_anomaly[season]
        - (
            thermo_anomaly[season]
            + halo_anomaly[season]
        )
    )

    residual_field[season] = residual.where(row_mask)

    no_data_mask[season] = xr.where(
        row_mask,
        np.nan,
        1.0,
    )


# 15. CLOSURE METRICS
all_residual_values = []

seasonal_closure_rows = []

for season in SEASON_ORDER:

    values = np.asarray(
        residual_field[season].values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if values.size > 0:
        all_residual_values.append(values)

        seasonal_closure_rows.append(
            {
                "season": season,
                "mean_residual_cm": float(np.mean(values)),
                "median_absolute_residual_cm": float(
                    np.median(np.abs(values))
                ),
                "RMSE_cm": float(
                    np.sqrt(np.mean(values ** 2))
                ),
                "maximum_absolute_residual_cm": float(
                    np.max(np.abs(values))
                ),
                "cells_within_tolerance_percent": float(
                    100.0
                    * np.mean(
                        np.abs(values)
                        <= CLOSURE_TOLERANCE_CM
                    )
                ),
                "n_cells": int(values.size),
            }
        )

if not all_residual_values:
    raise RuntimeError(
        "No finite residual values were available for closure analysis."
    )

combined_residual_values = np.concatenate(
    all_residual_values
)

overall_closure_metrics = {
    "scope": "All seasons combined",
    "tolerance_cm": CLOSURE_TOLERANCE_CM,
    "mean_residual_cm": float(
        np.mean(combined_residual_values)
    ),
    "median_absolute_residual_cm": float(
        np.median(
            np.abs(combined_residual_values)
        )
    ),
    "RMSE_cm": float(
        np.sqrt(
            np.mean(combined_residual_values ** 2)
        )
    ),
    "maximum_absolute_residual_cm": float(
        np.max(
            np.abs(combined_residual_values)
        )
    ),
    "cells_within_tolerance_percent": float(
        100.0
        * np.mean(
            np.abs(combined_residual_values)
            <= CLOSURE_TOLERANCE_CM
        )
    ),
    "n_cells": int(
        combined_residual_values.size
    ),
}

closure_metrics_table = pd.concat(
    [
        pd.DataFrame([overall_closure_metrics]),
        pd.DataFrame(seasonal_closure_rows),
    ],
    ignore_index=True,
)

closure_metrics_table.to_csv(
    OUT_CLOSURE_CSV,
    index=False,
)

print("\nClosure metrics:")
print(
    closure_metrics_table.to_string(index=False)
)


# 16. SAVE SEASONAL FIELDS
season_coordinate = xr.DataArray(
    SEASON_ORDER,
    dims="season",
    name="season",
)

total_stack = xr.concat(
    [total_anomaly[season] for season in SEASON_ORDER],
    dim=season_coordinate,
)

thermo_stack = xr.concat(
    [thermo_anomaly[season] for season in SEASON_ORDER],
    dim=season_coordinate,
)

halo_stack = xr.concat(
    [halo_anomaly[season] for season in SEASON_ORDER],
    dim=season_coordinate,
)

residual_stack = xr.concat(
    [residual_field[season] for season in SEASON_ORDER],
    dim=season_coordinate,
)

mask_stack = xr.concat(
    [
        seasonal_mask[season].astype(np.int8)
        for season in SEASON_ORDER
    ],
    dim=season_coordinate,
)

seasonal_output = xr.Dataset(
    {
        "total_steric_anomaly": total_stack,
        "thermosteric_anomaly": thermo_stack,
        "halosteric_anomaly": halo_stack,
        "closure_residual": residual_stack,
        "seasonal_common_mask": mask_stack,
    }
)

seasonal_output["total_steric_anomaly"].attrs.update({
    "long_name": (
        "seasonal total steric-height anomaly "
        "relative to the 2008-2025 all-month mean"
    ),
    "units": "cm",
})

seasonal_output["thermosteric_anomaly"].attrs.update({
    "long_name": (
        "seasonal thermosteric-height anomaly "
        "relative to the 2008-2025 all-month mean"
    ),
    "units": "cm",
})

seasonal_output["halosteric_anomaly"].attrs.update({
    "long_name": (
        "seasonal halosteric-height anomaly "
        "relative to the 2008-2025 all-month mean"
    ),
    "units": "cm",
})

seasonal_output["closure_residual"].attrs.update({
    "long_name": (
        "closure residual: total - (thermo + halo)"
    ),
    "units": "cm",
})

seasonal_output["seasonal_common_mask"].attrs.update({
    "long_name": "season-specific common finite mask",
    "flag_values": np.array([0, 1], dtype=np.int8),
    "flag_meanings": "excluded included",
})

seasonal_output.attrs.update({
    "title": "Figure 8 EN4 seasonal steric decomposition",
    "closure_definition": "total - (thermo + halo)",
    "time_period": f"{YEAR_START}-01 to {YEAR_END}-12",
})

seasonal_output.rename(
    {
        "lat": "latitude",
        "lon": "longitude",
    }
).to_netcdf(
    OUT_SEASONAL_NC,
    encoding={
        "total_steric_anomaly": {
            "zlib": True,
            "complevel": 4,
        },
        "thermosteric_anomaly": {
            "zlib": True,
            "complevel": 4,
        },
        "halosteric_anomaly": {
            "zlib": True,
            "complevel": 4,
        },
        "closure_residual": {
            "zlib": True,
            "complevel": 4,
        },
        "seasonal_common_mask": {
            "zlib": True,
            "complevel": 4,
        },
    },
)




# DATA PRODUCT GROUP 05: FIGURE 9 FRESHWATER / SSS / STRATIFICATION SEASONAL PRODUCT
# Source: so_sealevel_paper_fig.py
# File name: Fig09_EN4_freshwater_stratification_seasonal_fields.nc

# FIGURE 9 — EN4 FRESHWATER AND STRATIFICATION
# LOW-RAM / COLAB-SAFE WORKFLOW
# Southern Ocean, 2008–2025
#
# The script reads only ONE MONTH at a time.
# It never loads complete 216-month MLD or SIC arrays into memory.
#
# Columns
# -------
# 1. FWC seasonal climatology
# 2. SSS seasonal departure from the 2008–2025 all-month mean
# 3. Stratification seasonal climatology
#       + solid MLD contours
#       + dashed SIC 15% contour
#
# Output directory
# ----------------
# /content/drive/MyDrive/SAM_Thesis/paper2/


# 0. INSTALL REQUIRED PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "cftime": "cftime",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print(
        "Installing missing packages:",
        ", ".join(missing_packages),
    )

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing_packages,
        ]
    )

    print("Package installation completed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib

# Non-interactive backend uses less memory in Colab.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)



# 3. INPUT AND OUTPUT PATHS
FWC_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "EN4_FWC_0_200m_monthly_2008_2025_SO.nc"
    ),
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "EN4_NetCDF_inventory/"
        "EN4_freshwater_content_0_200m_monthly_2008_2025_SO.nc"
    ),
]

SSS_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "EN4_SSS_0_30m_monthly_2008_2025_SO.nc"
    ),
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "EN4_NetCDF_inventory/"
        "EN4_surface_salinity_0_30m_monthly_2008_2025_SO.nc"
    ),
]

STRATIFICATION_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "EN4_stratification_20_200m_monthly_2008_2025_SO.nc"
    ),
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "EN4_NetCDF_inventory/"
        "EN4_stratification_20_200m_monthly_2008_2025_SO.nc"
    ),
]

MLD_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Processed/"
        "MLD_monthly_2008_2025_SO.nc"
    ),
    (
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "MLD_monthly_2008_2025_SO.nc"
    ),
]

SIC_CANDIDATES = [
    (
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"
    ),
]


def first_existing_path(candidates):

    for candidate in candidates:

        if os.path.exists(
            candidate
        ):

            return candidate

    return candidates[0]


FWC_FILE = first_existing_path(
    FWC_CANDIDATES
)

SSS_FILE = first_existing_path(
    SSS_CANDIDATES
)

STRATIFICATION_FILE = first_existing_path(
    STRATIFICATION_CANDIDATES
)

MLD_FILE = first_existing_path(
    MLD_CANDIDATES
)

SIC_FILE = first_existing_path(
    SIC_CANDIDATES
)


OUTPUT_DIRECTORY = Path(
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)



OUT_SEASONAL_NC = (
    OUTPUT_DIRECTORY
    / "Fig09_EN4_freshwater_stratification_seasonal_fields.nc"
)

OUT_SUMMARY_TXT = (
    OUTPUT_DIRECTORY
    / "Fig09_EN4_freshwater_stratification_summary.txt"
)


# 4. ANALYSIS SETTINGS
YEAR_START = 2008
YEAR_END = 2025

LAT_MIN = -90.0
LAT_MAX = -60.0

SEASON_ORDER = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

SEASON_MONTHS = {
    "Spring": [
        9,
        10,
        11,
    ],
    "Summer": [
        12,
        1,
        2,
    ],
    "Autumn": [
        3,
        4,
        5,
    ],
    "Winter": [
        6,
        7,
        8,
    ],
}

SEASON_ROW_LABEL = {
    "Spring": "Spring",
    "Summer": "Summer",
    "Autumn": "Autumn",
    "Winter": "Winter",
}

COLUMN_LABELS = [
    "FWC\nclimatology",
    "SSS\nanomaly",
    "Stratification + MLD",
]

PANEL_LETTERS = [
    [
        "(a)",
        "(b)",
        "(c)",
    ],
    [
        "(d)",
        "(e)",
        "(f)",
    ],
    [
        "(g)",
        "(h)",
        "(i)",
    ],
    [
        "(j)",
        "(k)",
        "(l)",
    ],
]

MERIDIANS = np.arange(
    -180,
    181,
    30,
)

PARALLELS = [
    -60,
    -70,
    -80,
]

SIC_CONTOUR_PERCENT = 15.0

# Only levels falling within a seasonal field are plotted.
PREFERRED_MLD_LEVELS_M = [
    50,
    100,
    200,
    500,
    1000,
]

# The PDF is vector and should be used for publication.
# A 600-dpi PNG is the stable low-RAM option.

# Setting this to True may require substantially more RAM.
SAVE_OPTIONAL_1080_DPI = False



# 6. INPUT CHECK
for required_file in [
    FWC_FILE,
    SSS_FILE,
    STRATIFICATION_FILE,
    MLD_FILE,
    SIC_FILE,
]:

    if not os.path.exists(
        required_file
    ):

        raise FileNotFoundError(
            f"Required file was not found:\n"
            f"{required_file}"
        )


print(
    "\nUsing files:"
)

print(
    "FWC  :",
    FWC_FILE,
)

print(
    "SSS  :",
    SSS_FILE,
)

print(
    "STRAT:",
    STRATIFICATION_FILE,
)

print(
    "MLD  :",
    MLD_FILE,
)

print(
    "SIC  :",
    SIC_FILE,
)


# 7. DATASET HELPERS
def safe_is_datetime(
    dtype,
):

    try:

        return np.issubdtype(
            dtype,
            np.datetime64,
        )

    except TypeError:

        return False


def open_dataset_safely(
    file_path,
):
    """
    Open without Dask chunks.

    This is intentional: monthly slices are loaded explicitly one
    at a time, preventing Dask from building a large task graph.
    """
    attempts = []

    for engine in [
        None,
        "netcdf4",
        "h5netcdf",
        "scipy",
    ]:

        try:

            kwargs = {
                "decode_times": True,
                "mask_and_scale": True,
                "cache": False,
            }

            if engine is not None:

                kwargs[
                    "engine"
                ] = engine

            dataset = xr.open_dataset(
                file_path,
                **kwargs,
            )

            engine_name = (
                "xarray-default"
                if engine is None
                else engine
            )

            print(
                f"Opened {os.path.basename(file_path)} "
                f"with engine={engine_name}"
            )

            return dataset

        except Exception as error:

            attempts.append(
                f"{engine}: {error}"
            )

    raise RuntimeError(
        f"Could not open:\n"
        f"{file_path}\n\n"
        + "\n".join(
            attempts
        )
    )


def detect_coordinate(
    dataset,
    coordinate_type,
):
    """
    Detect time, latitude and longitude names.
    """
    aliases = {
        "time": [
            "time",
            "date",
            "datetime",
            "month",
            "valid_time",
            "t",
        ],
        "lat": [
            "lat",
            "latitude",
            "nav_lat",
            "y",
        ],
        "lon": [
            "lon",
            "longitude",
            "nav_lon",
            "x",
        ],
    }

    candidates = list(
        dataset.coords
    ) + [
        name
        for name in dataset.variables
        if name not in dataset.coords
    ]

    for name in candidates:

        variable = dataset[
            name
        ]

        lower_name = name.lower()

        standard_name = str(
            variable.attrs.get(
                "standard_name",
                "",
            )
        ).lower()

        axis = str(
            variable.attrs.get(
                "axis",
                "",
            )
        ).upper()

        units = str(
            variable.attrs.get(
                "units",
                "",
            )
        ).lower()

        if coordinate_type == "time":

            if (
                lower_name
                in aliases["time"]
                or standard_name
                == "time"
                or axis == "T"
                or " since "
                in units
                or safe_is_datetime(
                    variable.dtype
                )
            ):

                return name

        elif coordinate_type == "lat":

            if (
                lower_name
                in aliases["lat"]
                or standard_name
                == "latitude"
                or axis == "Y"
                or "degrees_north"
                in units
            ):

                return name

        elif coordinate_type == "lon":

            if (
                lower_name
                in aliases["lon"]
                or standard_name
                == "longitude"
                or axis == "X"
                or "degrees_east"
                in units
            ):

                return name

    raise KeyError(
        f"Could not detect "
        f"{coordinate_type}."
    )


def choose_data_variable(
    dataset,
    preferred_names,
):
    """
    Prefer named science variables and avoid known auxiliary fields.
    """
    excluded = {
        "valid_layer_mask",
        "gebco_water_depth",
        "time_bnds",
        "depth_bnds",
        "lat_bnds",
        "lon_bnds",
        "surface_salinity_reference",
    }

    available = [
        variable_name
        for variable_name
        in dataset.data_vars
        if variable_name
        not in excluded
    ]

    for preferred_name in preferred_names:

        for variable_name in available:

            if (
                preferred_name.lower()
                in variable_name.lower()
            ):

                return variable_name

    if not available:

        raise ValueError(
            "No suitable science variable was found."
        )

    return available[0]


def standardize_dataset(
    dataset,
):
    """
    Standardize names, longitude, latitude and time range lazily.
    """
    time_name = detect_coordinate(
        dataset,
        "time",
    )

    lat_name = detect_coordinate(
        dataset,
        "lat",
    )

    lon_name = detect_coordinate(
        dataset,
        "lon",
    )

    rename_mapping = {}

    if time_name != "time":

        rename_mapping[
            time_name
        ] = "time"

    if lat_name != "lat":

        rename_mapping[
            lat_name
        ] = "lat"

    if lon_name != "lon":

        rename_mapping[
            lon_name
        ] = "lon"

    if rename_mapping:

        dataset = dataset.rename(
            rename_mapping
        )

    normalized_longitude = (
        (
            dataset[
                "lon"
            ].astype(float)
            + 180.0
        )
         % 360.0
    ) - 180.0

    dataset = dataset.assign_coords(
        lon=normalized_longitude
    )

    longitude_values = np.asarray(
        dataset[
            "lon"
        ].values
    )

    _, unique_indices = np.unique(
        longitude_values,
        return_index=True,
    )

    dataset = dataset.isel(
        lon=np.sort(
            unique_indices
        )
    )

    dataset = dataset.sortby(
        "lon"
    )

    dataset = dataset.sortby(
        "lat"
    )

    dataset = dataset.sel(
        lat=slice(
            LAT_MIN,
            LAT_MAX,
        )
    )

    dataset = dataset.sel(
        time=slice(
            f"{YEAR_START}-01-01",
            f"{YEAR_END}-12-31",
        )
    )

    return dataset


def prepare_monthly_field(
    field,
):
    """
    Return one 2-D lat × lon monthly field.
    """
    field = field.squeeze(
        drop=True
    )

    extra_dimensions = [
        dimension
        for dimension
        in field.dims
        if dimension
        not in [
            "lat",
            "lon",
        ]
    ]

    if extra_dimensions:

        raise ValueError(
            "A monthly field still contains "
            f"unexpected dimensions: "
            f"{extra_dimensions}"
        )

    return field.transpose(
        "lat",
        "lon",
    )


def infer_multiplication_factor(
    data_array,
    role,
):
    """
    Infer only from metadata and ONE monthly slice.
    Never load the complete time series.
    """
    units = str(
        data_array.attrs.get(
            "units",
            "",
        )
    ).lower()

    if role == "mld":

        if (
            "kilomet" in units
            or units.strip()
            in [
                "km",
                "kilometer",
                "kilometers",
            ]
        ):

            return 1000.0

        return 1.0

    if role == "sic":

        if "%" in units:

            return 1.0

        first_slice = prepare_monthly_field(
            data_array.isel(
                time=0
            ).load()
        )

        sample_values = np.asarray(
            first_slice.values,
            dtype=np.float32,
        )

        finite_values = sample_values[
            np.isfinite(
                sample_values
            )
        ]

        del first_slice
        del sample_values

        gc.collect()

        if (
            finite_values.size > 0
            and np.nanmax(
                finite_values
            ) <= 1.01
        ):

            return 100.0

        return 1.0

    return 1.0


def grids_match(
    data_array,
    target_latitude,
    target_longitude,
):
    """
    Check whether a monthly field is already on the target grid.
    """
    return (
        data_array.sizes.get(
            "lat"
        )
        == target_latitude.size
        and data_array.sizes.get(
            "lon"
        )
        == target_longitude.size
        and np.allclose(
            data_array[
                "lat"
            ].values,
            target_latitude,
        )
        and np.allclose(
            data_array[
                "lon"
            ].values,
            target_longitude,
        )
    )


def monthly_field_to_target_numpy(
    monthly_field,
    target_latitude,
    target_longitude,
    multiplication_factor=1.0,
):
    """
    Load and, when required, interpolate ONE monthly field only.
    """
    monthly_field = prepare_monthly_field(
        monthly_field
    )

    if not grids_match(
        monthly_field,
        target_latitude,
        target_longitude,
    ):

        monthly_field = monthly_field.interp(
            lat=xr.DataArray(
                target_latitude,
                dims="lat",
                coords={
                    "lat": target_latitude,
                },
            ),
            lon=xr.DataArray(
                target_longitude,
                dims="lon",
                coords={
                    "lon": target_longitude,
                },
            ),
            method="linear",
        )

    values = np.asarray(
        monthly_field.values,
        dtype=np.float32,
    )

    if multiplication_factor != 1.0:

        values = (
            values
            * np.float32(
                multiplication_factor
            )
        )

    return values


def month_to_season(
    month,
):

    for season_name, season_months in (
        SEASON_MONTHS.items()
    ):

        if month in season_months:

            return season_name

    raise ValueError(
        f"Month {month} is not assigned "
        "to an austral season."
    )


def sequential_seasonal_aggregation(
    file_path,
    preferred_variables,
    target_latitude,
    target_longitude,
    role,
    calculate_all_month_mean=False,
):
    """
    Process one source file at a time and one month at a time.

    Memory at any instant contains:
      • one monthly source field;
      • one monthly target-grid field;
      • small target-grid accumulators.
    """
    dataset = open_dataset_safely(
        file_path
    )

    dataset = standardize_dataset(
        dataset
    )

    variable_name = choose_data_variable(
        dataset,
        preferred_variables,
    )

    data_array = dataset[
        variable_name
    ]

    multiplication_factor = (
        infer_multiplication_factor(
            data_array,
            role,
        )
    )

    times = pd.DatetimeIndex(
        pd.to_datetime(
            data_array[
                "time"
            ].values
        )
    )

    target_shape = (
        target_latitude.size,
        target_longitude.size,
    )

    seasonal_sum = {
        season: np.zeros(
            target_shape,
            dtype=np.float64,
        )
        for season in SEASON_ORDER
    }

    seasonal_count = {
        season: np.zeros(
            target_shape,
            dtype=np.uint16,
        )
        for season in SEASON_ORDER
    }

    if calculate_all_month_mean:

        all_month_sum = np.zeros(
            target_shape,
            dtype=np.float64,
        )

        all_month_count = np.zeros(
            target_shape,
            dtype=np.uint16,
        )

    else:

        all_month_sum = None
        all_month_count = None

    number_of_times = len(
        times
    )

    print(
        f"\nProcessing {role}: "
        f"{number_of_times} monthly records"
    )

    for time_index, timestamp in enumerate(
        times
    ):

        season = month_to_season(
            int(
                timestamp.month
            )
        )

        monthly_values = (
            monthly_field_to_target_numpy(
                data_array.isel(
                    time=time_index
                ),
                target_latitude,
                target_longitude,
                multiplication_factor,
            )
        )

        finite = np.isfinite(
            monthly_values
        )

        seasonal_sum[
            season
        ][finite] += monthly_values[
            finite
        ].astype(
            np.float64
        )

        seasonal_count[
            season
        ][finite] += 1

        if calculate_all_month_mean:

            all_month_sum[
                finite
            ] += monthly_values[
                finite
            ].astype(
                np.float64
            )

            all_month_count[
                finite
            ] += 1

        del monthly_values
        del finite

        if (
            (time_index + 1) % 12 == 0
            or time_index
            == number_of_times - 1
        ):

            print(
                f"  [{time_index + 1:03d}/"
                f"{number_of_times:03d}] "
                f"{timestamp:%Y-%m}"
            )

            gc.collect()

    seasonal_mean = {}

    for season in SEASON_ORDER:

        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):

            mean_values = np.divide(
                seasonal_sum[
                    season
                ],
                seasonal_count[
                    season
                ],
                out=np.full(
                    target_shape,
                    np.nan,
                    dtype=np.float64,
                ),
                where=(
                    seasonal_count[
                        season
                    ]
                    > 0
                ),
            )

        seasonal_mean[
            season
        ] = mean_values.astype(
            np.float32
        )

    if calculate_all_month_mean:

        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):

            all_month_mean = np.divide(
                all_month_sum,
                all_month_count,
                out=np.full(
                    target_shape,
                    np.nan,
                    dtype=np.float64,
                ),
                where=(
                    all_month_count
                    > 0
                ),
            ).astype(
                np.float32
            )

    else:

        all_month_mean = None

    units = str(
        data_array.attrs.get(
            "units",
            "",
        )
    )

    dataset.close()

    del dataset
    del data_array
    del seasonal_sum
    del seasonal_count
    del all_month_sum
    del all_month_count

    gc.collect()

    return {
        "variable_name": variable_name,
        "units": units,
        "seasonal_mean": seasonal_mean,
        "all_month_mean": all_month_mean,
        "multiplication_factor": (
            multiplication_factor
        ),
        "first_time": times[0],
        "last_time": times[-1],
        "number_of_months": len(times),
    }


def data_array_from_numpy(
    values,
    latitude,
    longitude,
    name,
    units,
):
    """
    Construct a small target-grid DataArray.
    """
    output = xr.DataArray(
        values,
        dims=(
            "lat",
            "lon",
        ),
        coords={
            "lat": latitude,
            "lon": longitude,
        },
        name=name,
    )

    output.attrs[
        "units"
    ] = units

    return output


# 8. READ TARGET EN4 GRID ONLY
target_dataset = open_dataset_safely(
    STRATIFICATION_FILE
)

target_dataset = standardize_dataset(
    target_dataset
)

TARGET_LATITUDE = np.asarray(
    target_dataset[
        "lat"
    ].values,
    dtype=np.float64,
)

TARGET_LONGITUDE = np.asarray(
    target_dataset[
        "lon"
    ].values,
    dtype=np.float64,
)

target_dataset.close()

del target_dataset

gc.collect()

print(
    "\nTarget EN4 grid:"
)

print(
    f"Latitude cells : "
    f"{TARGET_LATITUDE.size}"
)

print(
    f"Longitude cells: "
    f"{TARGET_LONGITUDE.size}"
)


# 9. SEQUENTIAL LOW-RAM AGGREGATION
fwc_result = sequential_seasonal_aggregation(
    file_path=FWC_FILE,
    preferred_variables=[
        "freshwater_content",
        "fwc",
    ],
    target_latitude=TARGET_LATITUDE,
    target_longitude=TARGET_LONGITUDE,
    role="fwc",
    calculate_all_month_mean=False,
)

sss_result = sequential_seasonal_aggregation(
    file_path=SSS_FILE,
    preferred_variables=[
        "surface_salinity",
        "sss",
        "salinity",
    ],
    target_latitude=TARGET_LATITUDE,
    target_longitude=TARGET_LONGITUDE,
    role="sss",
    calculate_all_month_mean=True,
)

stratification_result = (
    sequential_seasonal_aggregation(
        file_path=STRATIFICATION_FILE,
        preferred_variables=[
            "stratification_20_200m",
            "stratification",
            "density_difference",
        ],
        target_latitude=TARGET_LATITUDE,
        target_longitude=TARGET_LONGITUDE,
        role="stratification",
        calculate_all_month_mean=False,
    )
)

mld_result = sequential_seasonal_aggregation(
    file_path=MLD_FILE,
    preferred_variables=[
        "MLD",
        "mld",
        "mixed_layer_depth",
        "mlotst",
    ],
    target_latitude=TARGET_LATITUDE,
    target_longitude=TARGET_LONGITUDE,
    role="mld",
    calculate_all_month_mean=False,
)

sic_result = sequential_seasonal_aggregation(
    file_path=SIC_FILE,
    preferred_variables=[
        "SIF",
        "sif",
        "sea_ice_fraction",
        "sic",
    ],
    target_latitude=TARGET_LATITUDE,
    target_longitude=TARGET_LONGITUDE,
    role="sic",
    calculate_all_month_mean=False,
)


# 10. BUILD SMALL SEASONAL DATA ARRAYS
fwc_seasonal = {}
sss_anomaly_seasonal = {}
stratification_seasonal = {}
mld_seasonal = {}
sic_seasonal = {}

for season in SEASON_ORDER:

    fwc_seasonal[
        season
    ] = data_array_from_numpy(
        fwc_result[
            "seasonal_mean"
        ][
            season
        ],
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
        "freshwater_content",
        "m",
    )

    sss_anomaly_values = (
        sss_result[
            "seasonal_mean"
        ][
            season
        ]
        - sss_result[
            "all_month_mean"
        ]
    ).astype(
        np.float32
    )

    sss_anomaly_seasonal[
        season
    ] = data_array_from_numpy(
        sss_anomaly_values,
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
        "surface_salinity_anomaly",
        "1",
    )

    stratification_seasonal[
        season
    ] = data_array_from_numpy(
        stratification_result[
            "seasonal_mean"
        ][
            season
        ],
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
        "stratification_20_200m",
        "kg m-3",
    )

    mld_seasonal[
        season
    ] = data_array_from_numpy(
        mld_result[
            "seasonal_mean"
        ][
            season
        ],
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
        "mixed_layer_depth",
        "m",
    )

    sic_seasonal[
        season
    ] = data_array_from_numpy(
        sic_result[
            "seasonal_mean"
        ][
            season
        ],
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
        "sea_ice_fraction",
        "%",
    )


# Release the aggregation dictionaries.
del fwc_result
del sss_result
del stratification_result
del mld_result
del sic_result

gc.collect()


# 11. COLOUR LIMIT HELPERS
def collect_finite_values(
    data_dictionary,
):

    collected = []

    for season in SEASON_ORDER:

        values = np.asarray(
            data_dictionary[
                season
            ].values,
            dtype=np.float32,
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        if finite.size > 0:

            collected.append(
                finite
            )

    if not collected:

        return np.array(
            [
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

    return np.concatenate(
        collected
    )


def nice_upper_limit(
    value,
):

    if (
        not np.isfinite(
            value
        )
        or value <= 0
    ):

        return 1.0

    exponent = np.floor(
        np.log10(
            value
        )
    )

    scale = 10.0 ** exponent

    normalized = value / scale

    for candidate in [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
    ]:

        if normalized <= candidate:

            return candidate * scale

    return 10.0 * scale


fwc_values = collect_finite_values(
    fwc_seasonal
)

sss_values = collect_finite_values(
    sss_anomaly_seasonal
)

stratification_values = collect_finite_values(
    stratification_seasonal
)

# FWC can be slightly negative under the selected reference
# salinity. Preserve negative values when present.
fwc_lower_percentile = float(
    np.nanpercentile(
        fwc_values,
        1.0,
    )
)

fwc_upper_percentile = float(
    np.nanpercentile(
        fwc_values,
        99.0,
    )
)

FWC_VMIN = min(
    0.0,
    fwc_lower_percentile,
)

FWC_VMAX = nice_upper_limit(
    fwc_upper_percentile
)

SSS_LIMIT = nice_upper_limit(
    float(
        np.nanpercentile(
            np.abs(
                sss_values
            ),
            99.0,
        )
    )
)

STRATIFICATION_VMIN = min(
    0.0,
    float(
        np.nanpercentile(
            stratification_values,
            1.0,
        )
    ),
)

STRATIFICATION_VMAX = nice_upper_limit(
    float(
        np.nanpercentile(
            stratification_values,
            99.0,
        )
    )
)

print(
    "\nFinal colour limits:"
)

print(
    f"FWC            : "
    f"{FWC_VMIN:.4g} to "
    f"{FWC_VMAX:.4g} m"
)

print(
    f"SSS anomaly    : "
    f"±{SSS_LIMIT:.4g}"
)

print(
    f"Stratification : "
    f"{STRATIFICATION_VMIN:.4g} to "
    f"{STRATIFICATION_VMAX:.4g} "
    f"kg m-3"
)


# 12. SAVE SMALL SEASONAL NETCDF
season_coordinate = xr.DataArray(
    SEASON_ORDER,
    dims="season",
    name="season",
)

seasonal_output = xr.Dataset(
    {
        "freshwater_content_climatology": xr.concat(
            [
                fwc_seasonal[
                    season
                ]
                for season in SEASON_ORDER
            ],
            dim=season_coordinate,
        ),
        "surface_salinity_anomaly": xr.concat(
            [
                sss_anomaly_seasonal[
                    season
                ]
                for season in SEASON_ORDER
            ],
            dim=season_coordinate,
        ),
        "stratification_climatology": xr.concat(
            [
                stratification_seasonal[
                    season
                ]
                for season in SEASON_ORDER
            ],
            dim=season_coordinate,
        ),
        "mixed_layer_depth_climatology": xr.concat(
            [
                mld_seasonal[
                    season
                ]
                for season in SEASON_ORDER
            ],
            dim=season_coordinate,
        ),
        "sea_ice_fraction_climatology": xr.concat(
            [
                sic_seasonal[
                    season
                ]
                for season in SEASON_ORDER
            ],
            dim=season_coordinate,
        ),
    }
)

seasonal_output.attrs.update(
    {
        "title": (
            "Figure 9 EN4 freshwater and "
            "stratification seasonal fields"
        ),
        "period": (
            "2008-01 to 2025-12"
        ),
        "SSS_anomaly_reference": (
            "2008-2025 all-month grid-cell mean"
        ),
        "SIC_contour_level_percent": (
            SIC_CONTOUR_PERCENT
        ),
    }
)

encoding = {
    variable_name: {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(
            9.96921e36
        ),
    }
    for variable_name
    in seasonal_output.data_vars
}

seasonal_output.rename(
    {
        "lat": "latitude",
        "lon": "longitude",
    }
).to_netcdf(
    OUT_SEASONAL_NC,
    encoding=encoding,
)

seasonal_output.close()

del seasonal_output

gc.collect()





# DATA PRODUCT GROUP 06: FIGURE 10 SEA-WISE MONTHLY CLIMATOLOGICAL-CYCLE TABLE
# Source: so_sealevel_paper_fig.py
# File name: Fig10_sea_wise_monthly_climatological_cycles.csv

# FIGURE 10 — SEA-WISE MONTHLY CLIMATOLOGICAL CYCLES
#
# 10a: SLA
# 10b: EN4 total steric
# 10c: EN4 thermosteric
# 10d: EN4 halosteric
#
# Output:
#   Four vertically stacked heatmaps
#   X-axis: Jan–Dec
#   Y-axis: 13 Antarctic seas
#   Colour: monthly departure from each sea's annual mean
#   Unit: cm
#
# Required:
#   - common SLA–EN4 mask
#   - mark low-coverage SLA cells
#
# Results section:
#   3.8 Sea-wise monthly cycles



# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "matplotlib": "matplotlib",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
    print("Packages installed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})




# 3. INPUT / OUTPUT PATHS
SLA_FILE = "/content/drive/MyDrive/SAM_Thesis/Data/SLA_Antarctic_monthly_2008_2025.nc"
TOTAL_FILE = "/content/drive/MyDrive/SAM_Thesis/Processed/EN4_NetCDF_inventory/EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
THERMO_FILE = "/content/drive/MyDrive/SAM_Thesis/Processed/EN4_NetCDF_inventory/EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc"
HALO_FILE = "/content/drive/MyDrive/SAM_Thesis/Processed/EN4_NetCDF_inventory/EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc"

OUTPUT_DIR = "/content/drive/MyDrive/SAM_Thesis/paper2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(
    OUTPUT_DIR,
    "Fig10_sea_wise_monthly_climatological_cycles.csv",
)
OUT_TXT = os.path.join(
    OUTPUT_DIR,
    "Fig10_processing_summary.txt",
)


# 4. USER SETTINGS
ANALYSIS_START = "2008-01-01"
ANALYSIS_END   = "2025-12-31"

# SLA core-validity threshold for the common mask
SLA_CORE_VALID_FRACTION = 0.70

# A sea-month cell in the SLA panel is marked as "low coverage"
# if the valid coverage fraction for that sea and month is below this.
LOW_COVERAGE_THRESHOLD = 0.70

# Minimum number of valid years contributing to a month climatology
MIN_VALID_YEARS_PER_MONTH = 5

SAVE_DPI = 1080

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


# 5. SEA DEFINITIONS
# IMPORTANT:
# Replace these sector boundaries with the EXACT same sea-sector
# dictionary used in your Figures 4 and 5 if your earlier code uses
# slightly different boundaries.
#
# lon_min and lon_max are in degrees east, range [-180, 180].
# If lon_min > lon_max, the sector crosses the dateline.
SEA_SECTORS = OrderedDict([
    ("WED", {"name": "Weddell Sea",           "lon_min": -60,  "lon_max": -20}),
    ("KHV", {"name": "King Haakon VII Sea",   "lon_min": -20,  "lon_max":  10}),
    ("RLS", {"name": "Riiser-Larsen Sea",     "lon_min":  10,  "lon_max":  35}),
    ("LAZ", {"name": "Lazarev Sea",           "lon_min":  35,  "lon_max":  60}),
    ("COS", {"name": "Cosmonauts Sea",        "lon_min":  60,  "lon_max":  90}),
    ("COO", {"name": "Cooperation Sea",       "lon_min":  90,  "lon_max": 115}),
    ("DAV", {"name": "Davis Sea",             "lon_min": 115,  "lon_max": 130}),
    ("MAW", {"name": "Mawson Sea",            "lon_min": 130,  "lon_max": 150}),
    ("DUR", {"name": "D'Urville Sea",         "lon_min": 150,  "lon_max": 170}),
    ("SOM", {"name": "Somov Sea",             "lon_min": 170,  "lon_max": -160}),
    ("ROS", {"name": "Ross Sea",              "lon_min": -160, "lon_max": -130}),
    ("AMU", {"name": "Amundsen Sea",          "lon_min": -130, "lon_max": -100}),
    ("BEL", {"name": "Bellingshausen Sea",    "lon_min": -100, "lon_max":  -60}),
])

SEA_CODES = list(SEA_SECTORS.keys())
N_SEAS = len(SEA_CODES)


# 6. HELPER FUNCTIONS
def open_dataset_safely(file_path):
    attempts = []
    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = dict(decode_times=True, mask_and_scale=True)
            if engine is not None:
                kwargs["engine"] = engine
            ds = xr.open_dataset(file_path, **kwargs)
            engine_name = "xarray-default" if engine is None else engine
            print(f"Opened {os.path.basename(file_path)} with engine={engine_name}")
            return ds
        except Exception as e:
            attempts.append(f"{engine}: {e}")

    raise RuntimeError(
        f"Could not open {file_path}\n" + "\n".join(attempts)
    )


def detect_coord_name(ds, kind):
    aliases = {
        "time": ["time"],
        "lat": ["lat", "latitude", "nav_lat", "y"],
        "lon": ["lon", "longitude", "nav_lon", "x"],
    }

    for name in list(ds.coords) + list(ds.variables):
        lname = name.lower()
        var = ds[name]

        std = str(var.attrs.get("standard_name", "")).lower()
        axis = str(var.attrs.get("axis", "")).upper()
        units = str(var.attrs.get("units", "")).lower()

        if kind == "time":
            if lname in aliases["time"] or std == "time" or axis == "T":
                return name

        if kind == "lat":
            if lname in aliases["lat"] or std == "latitude" or axis == "Y" or "degrees_north" in units:
                return name

        if kind == "lon":
            if lname in aliases["lon"] or std == "longitude" or axis == "X" or "degrees_east" in units:
                return name

    raise KeyError(f"Could not detect {kind} coordinate.")


def standardize_dataset(ds):
    time_name = detect_coord_name(ds, "time")
    lat_name = detect_coord_name(ds, "lat")
    lon_name = detect_coord_name(ds, "lon")

    rename_map = {}
    if time_name != "time":
        rename_map[time_name] = "time"
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"

    if rename_map:
        ds = ds.rename(rename_map)

    # normalize longitude to [-180, 180)
    ds = ds.assign_coords(
        lon=(((ds["lon"].astype(float) + 180) % 360) - 180)
    )
    ds = ds.sortby("lon")
    ds = ds.sortby("lat")

    return ds


def choose_variable(ds, exact_names, contains_names=None):
    available = list(ds.data_vars)

    for name in exact_names:
        if name in available:
            return name

    if contains_names is not None:
        for token in contains_names:
            for name in available:
                if token.lower() in name.lower():
                    return name

    raise KeyError(
        f"Could not detect science variable. Available variables: {available}"
    )


def convert_to_cm(da):
    units = str(da.attrs.get("units", "")).strip().lower()

    if units in ["m", "meter", "metre", "meters", "metres"]:
        out = da * 100.0
        out.attrs["units"] = "cm"
        return out

    if units in ["cm", "centimeter", "centimetre", "centimeters", "centimetres"]:
        out = da.copy()
        out.attrs["units"] = "cm"
        return out

    # If no units are given, assume SLA and steric are already in meters only if values are small.
    vmax = float(np.nanmax(np.abs(da.values)))
    if vmax < 2.0:
        out = da * 100.0
        out.attrs["units"] = "cm"
        return out

    out = da.copy()
    out.attrs["units"] = "cm (assumed)"
    return out


def lon_sector_mask(lon_1d, lon_min, lon_max):
    lon_vals = np.asarray(lon_1d.values)

    if lon_min <= lon_max:
        mask = (lon_vals >= lon_min) & (lon_vals < lon_max)
    else:
        # dateline crossing sector
        mask = (lon_vals >= lon_min) | (lon_vals < lon_max)

    return xr.DataArray(mask, coords={"lon": lon_1d}, dims=["lon"])


def make_2d_sector_mask(lat_1d, lon_1d, lon_min, lon_max):
    lon_mask = lon_sector_mask(lon_1d, lon_min, lon_max)
    lat_mask = xr.DataArray(
        np.ones(lat_1d.size, dtype=bool),
        coords={"lat": lat_1d},
        dims=["lat"],
    )
    return lat_mask & lon_mask


def area_weights_2d(lat_1d, lon_1d):
    w_lat = np.cos(np.deg2rad(lat_1d))
    w2d = xr.DataArray(
        np.repeat(w_lat.values[:, None], lon_1d.size, axis=1),
        coords={"lat": lat_1d, "lon": lon_1d},
        dims=["lat", "lon"],
    )
    return w2d


def weighted_spatial_mean(data_3d, weights_2d, spatial_mask_2d):
    """
    data_3d: (time, lat, lon)
    weights_2d: (lat, lon)
    spatial_mask_2d: boolean (lat, lon)
    returns: time series (time)
    """
    masked = data_3d.where(spatial_mask_2d)
    valid_weights = weights_2d.where(spatial_mask_2d)

    numerator = (masked * valid_weights).sum(dim=("lat", "lon"), skipna=True)
    denominator = valid_weights.where(np.isfinite(masked)).sum(dim=("lat", "lon"), skipna=True)

    return numerator / denominator


def coverage_fraction(data_3d, weights_2d, spatial_mask_2d):
    """
    Coverage fraction for each time step inside a sea mask,
    relative to the total available weight of that sea in the fixed common mask.
    """
    sea_weights_total = weights_2d.where(spatial_mask_2d).sum(dim=("lat", "lon"), skipna=True)

    valid_weight = weights_2d.where(spatial_mask_2d & np.isfinite(data_3d))
    valid_weight_sum = valid_weight.sum(dim=("lat", "lon"), skipna=True)

    return valid_weight_sum / sea_weights_total


def monthly_climatology_and_anomaly(ts):
    """
    ts: time series (time)
    Returns:
      monthly climatology (month=1..12)
      anomaly from the sea's annual mean
      monthly sample count
    """
    monthly_clim = ts.groupby("time.month").mean(dim="time", skipna=True)
    monthly_count = ts.groupby("time.month").count(dim="time")

    annual_mean = monthly_clim.mean(dim="month", skipna=True)
    monthly_anom = monthly_clim - annual_mean

    return monthly_clim, monthly_anom, monthly_count


def nice_symmetric_limit(arrays, percentile=99.0):
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size > 0:
            vals.append(a)

    if not vals:
        return 1.0

    allvals = np.concatenate(vals)
    vmax = float(np.nanpercentile(np.abs(allvals), percentile))

    if vmax <= 1:
        candidates = [0.5, 1.0, 1.5, 2.0]
    elif vmax <= 5:
        candidates = [2.0, 3.0, 4.0, 5.0, 6.0]
    elif vmax <= 10:
        candidates = [6.0, 8.0, 10.0, 12.0]
    elif vmax <= 20:
        candidates = [12.0, 15.0, 20.0]
    else:
        candidates = [20.0, 25.0, 30.0, 40.0]

    for c in candidates:
        if vmax <= c:
            return c

    return np.ceil(vmax)


def draw_low_coverage_cells(ax, low_cov_mask_2d, nrows, ncols):
    """
    Overlay grey hatching on low-coverage SLA cells.
    low_cov_mask_2d shape: (nrows, ncols), True = low coverage
    """
    for i in range(nrows):
        for j in range(ncols):
            if low_cov_mask_2d[i, j]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=(0.75, 0.75, 0.75, 0.55),
                    edgecolor="none",
                    hatch=None,
                    zorder=5,
                )
                ax.add_patch(rect)


def style_heatmap_axis(ax, title, ylabels):
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTH_NAMES, fontsize=10, fontweight="bold")

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=11, fontweight="bold")

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(len(ylabels) - 0.5, -0.5)

    # cell grid
    for x in np.arange(-0.5, 12.5, 1):
        ax.axvline(x, color="white", linewidth=1.2, zorder=6)

    for y in np.arange(-0.5, len(ylabels) + 0.5, 1):
        ax.axhline(y, color="white", linewidth=1.2, zorder=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# 7. OPEN DATASETS
ds_sla = standardize_dataset(open_dataset_safely(SLA_FILE))
ds_tot = standardize_dataset(open_dataset_safely(TOTAL_FILE))
ds_the = standardize_dataset(open_dataset_safely(THERMO_FILE))
ds_hal = standardize_dataset(open_dataset_safely(HALO_FILE))

sla_var = choose_variable(ds_sla, exact_names=["sla"], contains_names=["sla"])
tot_var = choose_variable(ds_tot, exact_names=["total_steric_height"], contains_names=["total_steric"])
the_var = choose_variable(ds_the, exact_names=["thermosteric_height"], contains_names=["thermosteric"])
hal_var = choose_variable(ds_hal, exact_names=["halosteric_height"], contains_names=["halosteric"])

print("\nDetected variables:")
print("SLA         :", sla_var)
print("Total       :", tot_var)
print("Thermosteric:", the_var)
print("Halosteric  :", hal_var)


# 8. TIME SUBSET AND COMMON TIME PERIOD
sla = ds_sla[sla_var].sel(time=slice(ANALYSIS_START, ANALYSIS_END))
tot = ds_tot[tot_var].sel(time=slice(ANALYSIS_START, ANALYSIS_END))
the = ds_the[the_var].sel(time=slice(ANALYSIS_START, ANALYSIS_END))
hal = ds_hal[hal_var].sel(time=slice(ANALYSIS_START, ANALYSIS_END))

common_time = np.intersect1d(
    np.intersect1d(sla["time"].values, tot["time"].values),
    np.intersect1d(the["time"].values, hal["time"].values),
)

sla = sla.sel(time=common_time)
tot = tot.sel(time=common_time)
the = the.sel(time=common_time)
hal = hal.sel(time=common_time)

print("\nCommon analysis months:", sla.sizes["time"])


# 9. USE EN4 GRID AS THE COMMON GRID
# Restrict SLA to EN4 latitude range before interpolation
sla = sla.sel(
    lat=slice(float(tot["lat"].min()), float(tot["lat"].max()))
)

# Interpolate SLA to EN4 grid
sla_on_en4 = sla.interp(
    lat=tot["lat"],
    lon=tot["lon"],
    method="linear",
)

# Convert all to cm
sla_on_en4 = convert_to_cm(sla_on_en4)
tot = convert_to_cm(tot)
the = convert_to_cm(the)
hal = convert_to_cm(hal)

print("Units after conversion:")
print("SLA         :", sla_on_en4.attrs.get("units", ""))
print("Total       :", tot.attrs.get("units", ""))
print("Thermosteric:", the.attrs.get("units", ""))
print("Halosteric  :", hal.attrs.get("units", ""))


# 10. BUILD FIXED COMMON SLA–EN4 MASK
nmonths = int(sla_on_en4.sizes["time"])
min_valid_months = int(np.ceil(SLA_CORE_VALID_FRACTION * nmonths))

sla_core_mask = np.isfinite(sla_on_en4).sum(dim="time") >= min_valid_months

# EN4 common validity across all three steric variables
en4_core_mask = (
    (np.isfinite(tot).sum(dim="time") >= min_valid_months) &
    (np.isfinite(the).sum(dim="time") >= min_valid_months) &
    (np.isfinite(hal).sum(dim="time") >= min_valid_months)
)

common_mask = sla_core_mask & en4_core_mask

print("\nCommon fixed mask summary:")
print("Total grid cells on EN4 grid:", common_mask.size)
print("Common valid cells         :", int(common_mask.sum().values))


# 11. SEA-WISE MONTHLY SERIES
weights2d = area_weights_2d(tot["lat"], tot["lon"])

sla_matrix = np.full((N_SEAS, 12), np.nan)
tot_matrix = np.full((N_SEAS, 12), np.nan)
the_matrix = np.full((N_SEAS, 12), np.nan)
hal_matrix = np.full((N_SEAS, 12), np.nan)

sla_monthly_count_matrix = np.full((N_SEAS, 12), np.nan)
low_coverage_matrix = np.zeros((N_SEAS, 12), dtype=bool)

output_rows = []

for i, sea_code in enumerate(SEA_CODES):
    sector = SEA_SECTORS[sea_code]
    sea_mask = make_2d_sector_mask(
        tot["lat"],
        tot["lon"],
        sector["lon_min"],
        sector["lon_max"],
    )

    # final sea mask = sector ∩ common mask
    sea_common_mask = sea_mask & common_mask

    # if a sea has zero valid cells, skip
    if int(sea_common_mask.sum().values) == 0:
        print(f"WARNING: {sea_code} has zero common valid cells.")
        continue

    # weighted monthly time series for each variable
    sla_ts = weighted_spatial_mean(sla_on_en4, weights2d, sea_common_mask)
    tot_ts = weighted_spatial_mean(tot,      weights2d, sea_common_mask)
    the_ts = weighted_spatial_mean(the,      weights2d, sea_common_mask)
    hal_ts = weighted_spatial_mean(hal,      weights2d, sea_common_mask)

    # SLA coverage fraction per time step inside this sea
    sla_cov_ts = coverage_fraction(sla_on_en4, weights2d, sea_common_mask)

    # monthly climatology and anomaly from annual mean
    sla_clim, sla_anom, sla_count = monthly_climatology_and_anomaly(sla_ts)
    tot_clim, tot_anom, tot_count = monthly_climatology_and_anomaly(tot_ts)
    the_clim, the_anom, the_count = monthly_climatology_and_anomaly(the_ts)
    hal_clim, hal_anom, hal_count = monthly_climatology_and_anomaly(hal_ts)

    sla_cov_clim = sla_cov_ts.groupby("time.month").mean(dim="time", skipna=True)

    for m in range(1, 13):
        if int(sla_count.sel(month=m).values) >= MIN_VALID_YEARS_PER_MONTH:
            sla_matrix[i, m-1] = float(sla_anom.sel(month=m).values)
            sla_monthly_count_matrix[i, m-1] = float(sla_count.sel(month=m).values)
        else:
            sla_matrix[i, m-1] = np.nan

        if int(tot_count.sel(month=m).values) >= MIN_VALID_YEARS_PER_MONTH:
            tot_matrix[i, m-1] = float(tot_anom.sel(month=m).values)

        if int(the_count.sel(month=m).values) >= MIN_VALID_YEARS_PER_MONTH:
            the_matrix[i, m-1] = float(the_anom.sel(month=m).values)

        if int(hal_count.sel(month=m).values) >= MIN_VALID_YEARS_PER_MONTH:
            hal_matrix[i, m-1] = float(hal_anom.sel(month=m).values)

        cov_val = float(sla_cov_clim.sel(month=m).values)
        low_coverage_matrix[i, m-1] = np.isfinite(cov_val) and (cov_val < LOW_COVERAGE_THRESHOLD)

        output_rows.append({
            "sea_code": sea_code,
            "sea_name": sector["name"],
            "month": m,
            "month_name": MONTH_NAMES[m-1],
            "sla_anomaly_cm": sla_matrix[i, m-1],
            "total_steric_anomaly_cm": tot_matrix[i, m-1],
            "thermosteric_anomaly_cm": the_matrix[i, m-1],
            "halosteric_anomaly_cm": hal_matrix[i, m-1],
            "sla_valid_years": sla_monthly_count_matrix[i, m-1],
            "sla_low_coverage_flag": int(low_coverage_matrix[i, m-1]),
        })


# 12. SAVE OUTPUT TABLE
df_out = pd.DataFrame(output_rows)
df_out.to_csv(OUT_CSV, index=False)
print("\nSaved processed table:")
print(OUT_CSV)




# DATA PRODUCT GROUP 07: FIGURE 11 TREND / ANNUAL-ANOMALY DATA PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: Fig11_gridcell_trend_statistics_2008_2025.nc
# File name: Fig11_13sea_Sen_slopes_bootstrap_CI.csv
# File name: Fig11_deseasonalized_annual_anomalies_common_grid.nc

# FIGURE 11 — STUDY-PERIOD TRENDS DURING 2008–2025
#
# Structure
# ---------
# Top row:
#   (a) SLA trend map
#   (b) Total steric trend map
#   (c) Thermosteric trend map
#   (d) Halosteric trend map
#
# Bottom row:
#   (e) 13-sea SLA Sen slopes
#   (f) 13-sea total steric Sen slopes
#   (g) 13-sea thermosteric Sen slopes
#   (h) 13-sea halosteric Sen slopes
#
# Trend method
# ------------
# 1. Interpolate monthly SLA to the EN4 grid.
# 2. Apply one fixed SLA–EN4 common mask.
# 3. Remove each grid cell's 2008–2025 calendar-month climatology.
# 4. Average the deseasonalized monthly anomalies by calendar year.
# 5. Estimate Sen's slope from the annual anomalies.
# 6. Test significance with the Hamed–Rao modified Mann–Kendall test.
# 7. Correct grid-cell and sector p values using Benjamini–Hochberg FDR.
# 8. Estimate sector 95% confidence intervals using a 3-year
#    circular moving-block bootstrap of residuals.
#
# Units
# -----
# cm decade^-1
#
# Results section
# ---------------
# 3.9 Study-period trends and regional contrasts
#
# Output directory
# ----------------
# /content/drive/MyDrive/SAM_Thesis/paper2/



# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print(
        "Installing missing packages:",
        ", ".join(missing_packages),
    )

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing_packages,
        ]
    )

    print("Package installation completed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import gc
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import norm, rankdata

import matplotlib

# Non-interactive backend reduces notebook display memory.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from cartopy.util import add_cyclic_point

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)



# 3. INPUT AND OUTPUT PATHS
SLA_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "SLA_Antarctic_monthly_2008_2025.nc"
)

TOTAL_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
)

THERMO_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc"
)

HALO_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/"
    "EN4_NetCDF_inventory/"
    "EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc"
)

OUTPUT_DIRECTORY = Path(
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)



OUT_GRID_NC = (
    OUTPUT_DIRECTORY
    / "Fig11_gridcell_trend_statistics_2008_2025.nc"
)

OUT_SECTOR_CSV = (
    OUTPUT_DIRECTORY
    / "Fig11_13sea_Sen_slopes_bootstrap_CI.csv"
)

OUT_ANNUAL_NC = (
    OUTPUT_DIRECTORY
    / "Fig11_deseasonalized_annual_anomalies_common_grid.nc"
)

OUT_SUMMARY = (
    OUTPUT_DIRECTORY
    / "Fig11_processing_summary.txt"
)


# 4. ANALYSIS SETTINGS
YEAR_START = 2008
YEAR_END = 2025

ANALYSIS_START = f"{YEAR_START}-01-01"
ANALYSIS_END = f"{YEAR_END}-12-31"

LAT_MIN = -90.0
LAT_MAX = -60.0

SLA_CORE_VALID_FRACTION = 0.70
EN4_CORE_VALID_FRACTION = 1.00

# At least this many valid monthly anomalies are required to form
# a grid-cell annual anomaly.
MIN_VALID_MONTHS_PER_YEAR = 8

# At least this many valid annual anomalies are required for trend.
MIN_VALID_YEARS = 12

FDR_ALPHA = 0.05
MK_ALPHA = 0.05

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH_YEARS = 3
BOOTSTRAP_SEED = 4217

DRAW_SECTOR_BOUNDARIES = True

# A compact physical figure permits 1080-dpi export without the
# enormous canvas created by a 15–18 inch figure.
FIGSIZE = (
    10.2,
    7.0,
)

SAVE_DPI = 1080


# 5. 13 ANTARCTIC SEA-SECTOR DEFINITIONS
# Use the same sector order and longitude limits used in the
# preceding sea-wise analysis figures.
#
# Longitudes use the -180 to 180 convention. A sector with
# lon_min > lon_max crosses the dateline.
SEA_SECTORS = OrderedDict([
    (
        "WED",
        {
            "name": "Weddell Sea",
            "lon_min": -60.0,
            "lon_max": -20.0,
        },
    ),
    (
        "KHV",
        {
            "name": "King Haakon VII Sea",
            "lon_min": -20.0,
            "lon_max": 10.0,
        },
    ),
    (
        "RLS",
        {
            "name": "Riiser-Larsen Sea",
            "lon_min": 10.0,
            "lon_max": 35.0,
        },
    ),
    (
        "LAZ",
        {
            "name": "Lazarev Sea",
            "lon_min": 35.0,
            "lon_max": 60.0,
        },
    ),
    (
        "COS",
        {
            "name": "Cosmonauts Sea",
            "lon_min": 60.0,
            "lon_max": 90.0,
        },
    ),
    (
        "COO",
        {
            "name": "Cooperation Sea",
            "lon_min": 90.0,
            "lon_max": 115.0,
        },
    ),
    (
        "DAV",
        {
            "name": "Davis Sea",
            "lon_min": 115.0,
            "lon_max": 130.0,
        },
    ),
    (
        "MAW",
        {
            "name": "Mawson Sea",
            "lon_min": 130.0,
            "lon_max": 150.0,
        },
    ),
    (
        "DUR",
        {
            "name": "D'Urville Sea",
            "lon_min": 150.0,
            "lon_max": 170.0,
        },
    ),
    (
        "SOM",
        {
            "name": "Somov Sea",
            "lon_min": 170.0,
            "lon_max": -160.0,
        },
    ),
    (
        "ROS",
        {
            "name": "Ross Sea",
            "lon_min": -160.0,
            "lon_max": -130.0,
        },
    ),
    (
        "AMU",
        {
            "name": "Amundsen Sea",
            "lon_min": -130.0,
            "lon_max": -100.0,
        },
    ),
    (
        "BEL",
        {
            "name": "Bellingshausen Sea",
            "lon_min": -100.0,
            "lon_max": -60.0,
        },
    ),
])

SEA_CODES = list(
    SEA_SECTORS.keys()
)

SEA_NAMES = [
    SEA_SECTORS[
        code
    ][
        "name"
    ]
    for code in SEA_CODES
]


# 7. FILE AND DATASET HELPERS
for required_file in [
    SLA_FILE,
    TOTAL_FILE,
    THERMO_FILE,
    HALO_FILE,
]:

    if not os.path.exists(
        required_file
    ):

        raise FileNotFoundError(
            f"Required file was not found:\n"
            f"{required_file}"
        )


def safe_is_datetime(
    dtype,
):

    try:

        return np.issubdtype(
            dtype,
            np.datetime64,
        )

    except TypeError:

        return False


def open_dataset_safely(
    file_path,
):
    """
    Open without Dask. The SLA file is read month-by-month, while
    the EN4 fields are already small on the 24 × 360 target grid.
    """
    attempts = []

    for engine in [
        None,
        "netcdf4",
        "h5netcdf",
        "scipy",
    ]:

        try:

            kwargs = {
                "decode_times": True,
                "mask_and_scale": True,
                "cache": False,
            }

            if engine is not None:

                kwargs[
                    "engine"
                ] = engine

            dataset = xr.open_dataset(
                file_path,
                **kwargs,
            )

            engine_name = (
                "xarray-default"
                if engine is None
                else engine
            )

            print(
                f"Opened {os.path.basename(file_path)} "
                f"with engine={engine_name}"
            )

            return dataset

        except Exception as error:

            attempts.append(
                f"engine={engine}: {error}"
            )

    raise RuntimeError(
        f"Could not open:\n"
        f"{file_path}\n\n"
        + "\n".join(
            attempts
        )
    )


def detect_coordinate(
    dataset,
    coordinate_type,
):
    """
    Detect time, latitude and longitude coordinate names.
    """
    aliases = {
        "time": [
            "time",
            "date",
            "datetime",
            "valid_time",
            "month",
            "t",
        ],
        "lat": [
            "lat",
            "latitude",
            "nav_lat",
            "y",
        ],
        "lon": [
            "lon",
            "longitude",
            "nav_lon",
            "x",
        ],
    }

    candidates = list(
        dataset.coords
    ) + [
        name
        for name in dataset.variables
        if name not in dataset.coords
    ]

    for name in candidates:

        variable = dataset[
            name
        ]

        lower_name = name.lower()

        standard_name = str(
            variable.attrs.get(
                "standard_name",
                "",
            )
        ).lower()

        axis = str(
            variable.attrs.get(
                "axis",
                "",
            )
        ).upper()

        units = str(
            variable.attrs.get(
                "units",
                "",
            )
        ).lower()

        if coordinate_type == "time":

            if (
                lower_name
                in aliases["time"]
                or standard_name
                == "time"
                or axis == "T"
                or " since "
                in units
                or safe_is_datetime(
                    variable.dtype
                )
            ):

                return name

        elif coordinate_type == "lat":

            if (
                lower_name
                in aliases["lat"]
                or standard_name
                == "latitude"
                or axis == "Y"
                or "degrees_north"
                in units
            ):

                return name

        elif coordinate_type == "lon":

            if (
                lower_name
                in aliases["lon"]
                or standard_name
                == "longitude"
                or axis == "X"
                or "degrees_east"
                in units
            ):

                return name

    raise KeyError(
        f"Could not detect "
        f"{coordinate_type}."
    )


def standardize_dataset(
    dataset,
):
    """
    Standardize coordinate names and longitude convention.
    """
    time_name = detect_coordinate(
        dataset,
        "time",
    )

    lat_name = detect_coordinate(
        dataset,
        "lat",
    )

    lon_name = detect_coordinate(
        dataset,
        "lon",
    )

    rename_mapping = {}

    if time_name != "time":

        rename_mapping[
            time_name
        ] = "time"

    if lat_name != "lat":

        rename_mapping[
            lat_name
        ] = "lat"

    if lon_name != "lon":

        rename_mapping[
            lon_name
        ] = "lon"

    if rename_mapping:

        dataset = dataset.rename(
            rename_mapping
        )

    normalized_longitude = (
        (
            dataset[
                "lon"
            ].astype(float)
            + 180.0
        )
         % 360.0
    ) - 180.0

    dataset = dataset.assign_coords(
        lon=normalized_longitude
    )

    longitude_values = np.asarray(
        dataset[
            "lon"
        ].values
    )

    _, unique_indices = np.unique(
        longitude_values,
        return_index=True,
    )

    dataset = dataset.isel(
        lon=np.sort(
            unique_indices
        )
    )

    dataset = dataset.sortby(
        "lon"
    )

    dataset = dataset.sortby(
        "lat"
    )

    dataset = dataset.sel(
        time=slice(
            ANALYSIS_START,
            ANALYSIS_END,
        )
    )

    return dataset


def choose_variable(
    dataset,
    exact_names,
    contains_names,
):
    """
    Select the requested science variable.
    """
    excluded_names = {
        "valid_layer_mask",
        "gebco_water_depth",
        "surface_salinity_reference",
        "time_bnds",
        "depth_bnds",
    }

    available = [
        variable_name
        for variable_name
        in dataset.data_vars
        if variable_name
        not in excluded_names
    ]

    for candidate in exact_names:

        if candidate in available:

            return candidate

    for token in contains_names:

        for variable_name in available:

            if (
                token.lower()
                in variable_name.lower()
            ):

                return variable_name

    raise KeyError(
        "Could not find the requested science variable.\n"
        f"Available variables: {available}"
    )


def unit_factor_to_cm(
    data_array,
):
    """
    Determine a multiplication factor without loading the full array.
    """
    units = str(
        data_array.attrs.get(
            "units",
            "",
        )
    ).strip().lower()

    if units in [
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    ]:

        return 100.0

    if units in [
        "cm",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
    ]:

        return 1.0

    sample = np.asarray(
        data_array.isel(
            time=0
        ).values,
        dtype=np.float32,
    )

    finite = sample[
        np.isfinite(
            sample
        )
    ]

    if (
        finite.size > 0
        and np.nanpercentile(
            np.abs(
                finite
            ),
            99.0,
        )
        < 1.0
    ):

        return 100.0

    return 1.0


def prepare_monthly_field(
    field,
):
    """
    Return a 2-D lat × lon monthly field.
    """
    field = field.squeeze(
        drop=True
    )

    extra_dimensions = [
        dimension
        for dimension
        in field.dims
        if dimension
        not in [
            "lat",
            "lon",
        ]
    ]

    if extra_dimensions:

        raise ValueError(
            "Unexpected dimensions remain in a monthly field: "
            f"{extra_dimensions}"
        )

    return field.transpose(
        "lat",
        "lon",
    )


# 8. LOAD EN4 TARGET GRID AND MONTHLY VARIABLES
total_dataset = standardize_dataset(
    open_dataset_safely(
        TOTAL_FILE
    )
)

total_variable = choose_variable(
    total_dataset,
    exact_names=[
        "total_steric_height",
    ],
    contains_names=[
        "total_steric",
    ],
)

target_latitude = np.asarray(
    total_dataset[
        "lat"
    ].values,
    dtype=np.float64,
)

target_longitude = np.asarray(
    total_dataset[
        "lon"
    ].values,
    dtype=np.float64,
)

target_times = pd.DatetimeIndex(
    pd.to_datetime(
        total_dataset[
            "time"
        ].values
    )
)

total_factor = unit_factor_to_cm(
    total_dataset[
        total_variable
    ]
)

total_monthly = (
    np.asarray(
        total_dataset[
            total_variable
        ].transpose(
            "time",
            "lat",
            "lon",
        ).values,
        dtype=np.float32,
    )
    * np.float32(
        total_factor
    )
)

if "valid_layer_mask" in total_dataset:

    total_layer_mask = np.asarray(
        total_dataset[
            "valid_layer_mask"
        ].transpose(
            "time",
            "lat",
            "lon",
        ).values,
        dtype=np.int8,
    )

else:

    total_layer_mask = np.isfinite(
        total_monthly
    ).astype(
        np.int8
    )

total_dataset.close()

del total_dataset

gc.collect()


def load_en4_variable(
    file_path,
    exact_names,
    contains_names,
    target_times,
    target_latitude,
    target_longitude,
):
    """
    Load one EN4 variable on the target grid and common monthly period.
    """
    dataset = standardize_dataset(
        open_dataset_safely(
            file_path
        )
    )

    variable_name = choose_variable(
        dataset,
        exact_names=exact_names,
        contains_names=contains_names,
    )

    factor = unit_factor_to_cm(
        dataset[
            variable_name
        ]
    )

    selected = dataset[
        variable_name
    ].sel(
        time=target_times
    )

    grids_match = (
        selected.sizes.get(
            "lat"
        )
        == target_latitude.size
        and selected.sizes.get(
            "lon"
        )
        == target_longitude.size
        and np.allclose(
            selected[
                "lat"
            ].values,
            target_latitude,
        )
        and np.allclose(
            selected[
                "lon"
            ].values,
            target_longitude,
        )
    )

    if not grids_match:

        selected = selected.interp(
            lat=xr.DataArray(
                target_latitude,
                dims="lat",
                coords={
                    "lat": target_latitude,
                },
            ),
            lon=xr.DataArray(
                target_longitude,
                dims="lon",
                coords={
                    "lon": target_longitude,
                },
            ),
            method="linear",
        )

    values = (
        np.asarray(
            selected.transpose(
                "time",
                "lat",
                "lon",
            ).values,
            dtype=np.float32,
        )
        * np.float32(
            factor
        )
    )

    dataset.close()

    del dataset
    del selected

    gc.collect()

    return (
        values,
        variable_name,
    )


thermosteric_monthly, thermo_variable = (
    load_en4_variable(
        THERMO_FILE,
        exact_names=[
            "thermosteric_height",
        ],
        contains_names=[
            "thermosteric",
        ],
        target_times=target_times,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )
)

halosteric_monthly, halo_variable = (
    load_en4_variable(
        HALO_FILE,
        exact_names=[
            "halosteric_height",
        ],
        contains_names=[
            "halosteric",
        ],
        target_times=target_times,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )
)


# 9. INTERPOLATE SLA MONTH-BY-MONTH TO THE EN4 GRID
sla_dataset = standardize_dataset(
    open_dataset_safely(
        SLA_FILE
    )
)

sla_variable = choose_variable(
    sla_dataset,
    exact_names=[
        "sla",
    ],
    contains_names=[
        "sea_level_anomaly",
        "sla",
    ],
)

sla_factor = unit_factor_to_cm(
    sla_dataset[
        sla_variable
    ]
)

sla_times = pd.DatetimeIndex(
    pd.to_datetime(
        sla_dataset[
            "time"
        ].values
    )
)

common_times = target_times.intersection(
    sla_times
)

if len(
    common_times
) != len(
    target_times
):

    raise ValueError(
        "The SLA and EN4 monthly sequences do not match exactly.\n"
        f"EN4 months: {len(target_times)}\n"
        f"Common months: {len(common_times)}"
    )

sla_monthly = np.full(
    (
        len(target_times),
        target_latitude.size,
        target_longitude.size,
    ),
    np.nan,
    dtype=np.float32,
)

print(
    "\nInterpolating SLA month-by-month to the EN4 grid..."
)

for time_index, timestamp in enumerate(
    target_times
):

    monthly_field = prepare_monthly_field(
        sla_dataset[
            sla_variable
        ].sel(
            time=timestamp
        )
    )

    monthly_field = monthly_field.interp(
        lat=xr.DataArray(
            target_latitude,
            dims="lat",
            coords={
                "lat": target_latitude,
            },
        ),
        lon=xr.DataArray(
            target_longitude,
            dims="lon",
            coords={
                "lon": target_longitude,
            },
        ),
        method="linear",
    )

    sla_monthly[
        time_index
    ] = (
        np.asarray(
            monthly_field.values,
            dtype=np.float32,
        )
        * np.float32(
            sla_factor
        )
    )

    del monthly_field

    if (
        (time_index + 1) % 12 == 0
        or time_index
        == len(target_times) - 1
    ):

        print(
            f"  [{time_index + 1:03d}/"
            f"{len(target_times):03d}] "
            f"{timestamp:%Y-%m}"
        )

        gc.collect()

sla_dataset.close()

del sla_dataset

gc.collect()


# 10. FIXED COMMON SLA–EN4 MASK
number_of_months = len(
    target_times
)

sla_valid_fraction = (
    np.isfinite(
        sla_monthly
    ).sum(
        axis=0
    )
    / number_of_months
)

total_valid_fraction = (
    (
        np.isfinite(
            total_monthly
        )
        & (
            total_layer_mask
            == 1
        )
    ).sum(
        axis=0
    )
    / number_of_months
)

thermo_valid_fraction = (
    np.isfinite(
        thermosteric_monthly
    ).sum(
        axis=0
    )
    / number_of_months
)

halo_valid_fraction = (
    np.isfinite(
        halosteric_monthly
    ).sum(
        axis=0
    )
    / number_of_months
)

common_mask = (
    (
        sla_valid_fraction
        >= SLA_CORE_VALID_FRACTION
    )
    & (
        total_valid_fraction
        >= (
            EN4_CORE_VALID_FRACTION
            - 1.0e-10
        )
    )
    & (
        thermo_valid_fraction
        >= (
            EN4_CORE_VALID_FRACTION
            - 1.0e-10
        )
    )
    & (
        halo_valid_fraction
        >= (
            EN4_CORE_VALID_FRACTION
            - 1.0e-10
        )
    )
)

common_cell_count = int(
    common_mask.sum()
)

if common_cell_count == 0:

    raise RuntimeError(
        "The fixed SLA–EN4 common mask contains no cells."
    )

print(
    "\nFixed common mask cells:",
    f"{common_cell_count:,}",
)


# 11. REMOVE MONTHLY CLIMATOLOGY AND FORM ANNUAL ANOMALIES
calendar_months = np.asarray(
    target_times.month,
    dtype=np.int16,
)

calendar_years = np.asarray(
    target_times.year,
    dtype=np.int16,
)

analysis_years = np.arange(
    YEAR_START,
    YEAR_END + 1,
    dtype=np.int16,
)


def deseasonalize_and_annualize(
    monthly_values,
    calendar_months,
    calendar_years,
    analysis_years,
    fixed_mask,
):
    """
    Remove the grid-cell calendar-month climatology, then average
    the deseasonalized monthly anomalies by calendar year.
    """
    monthly_values = np.asarray(
        monthly_values,
        dtype=np.float32,
    )

    monthly_climatology = np.full(
        (
            12,
            monthly_values.shape[1],
            monthly_values.shape[2],
        ),
        np.nan,
        dtype=np.float32,
    )

    for month in range(
        1,
        13,
    ):

        month_indices = (
            calendar_months
            == month
        )

        monthly_climatology[
            month - 1
        ] = np.nanmean(
            monthly_values[
                month_indices
            ],
            axis=0,
        ).astype(
            np.float32
        )

    deseasonalized = np.full_like(
        monthly_values,
        np.nan,
        dtype=np.float32,
    )

    for time_index, month in enumerate(
        calendar_months
    ):

        deseasonalized[
            time_index
        ] = (
            monthly_values[
                time_index
            ]
            - monthly_climatology[
                month - 1
            ]
        )

    deseasonalized[
        :,
        ~fixed_mask,
    ] = np.nan

    annual_values = np.full(
        (
            analysis_years.size,
            monthly_values.shape[1],
            monthly_values.shape[2],
        ),
        np.nan,
        dtype=np.float32,
    )

    for year_index, year in enumerate(
        analysis_years
    ):

        year_indices = (
            calendar_years
            == year
        )

        year_block = deseasonalized[
            year_indices
        ]

        valid_month_count = np.isfinite(
            year_block
        ).sum(
            axis=0
        )

        year_mean = np.nanmean(
            year_block,
            axis=0,
        ).astype(
            np.float32
        )

        year_mean[
            valid_month_count
            < MIN_VALID_MONTHS_PER_YEAR
        ] = np.nan

        annual_values[
            year_index
        ] = year_mean

    return (
        annual_values,
        monthly_climatology,
    )


print(
    "\nRemoving monthly climatology and forming annual anomalies..."
)

sla_annual, sla_monthly_climatology = (
    deseasonalize_and_annualize(
        sla_monthly,
        calendar_months,
        calendar_years,
        analysis_years,
        common_mask,
    )
)

total_annual, total_monthly_climatology = (
    deseasonalize_and_annualize(
        total_monthly,
        calendar_months,
        calendar_years,
        analysis_years,
        common_mask,
    )
)

thermo_annual, thermo_monthly_climatology = (
    deseasonalize_and_annualize(
        thermosteric_monthly,
        calendar_months,
        calendar_years,
        analysis_years,
        common_mask,
    )
)

halo_annual, halo_monthly_climatology = (
    deseasonalize_and_annualize(
        halosteric_monthly,
        calendar_months,
        calendar_years,
        analysis_years,
        common_mask,
    )
)

# Release monthly arrays before the trend calculations.
del sla_monthly
del total_monthly
del thermosteric_monthly
del halosteric_monthly
del total_layer_mask

gc.collect()


# 12. SEN SLOPE AND HAMED–RAO MODIFIED MANN–KENDALL
def sen_slope(
    values,
    times,
):
    """
    Median pairwise Sen slope.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            values
        )
        & np.isfinite(
            times
        )
    )

    values = values[
        valid
    ]

    times = times[
        valid
    ]

    number_of_values = values.size

    if number_of_values < 2:

        return np.nan

    upper_i, upper_j = np.triu_indices(
        number_of_values,
        k=1,
    )

    time_difference = (
        times[
            upper_j
        ]
        - times[
            upper_i
        ]
    )

    value_difference = (
        values[
            upper_j
        ]
        - values[
            upper_i
        ]
    )

    valid_pairs = (
        time_difference
        != 0.0
    )

    slopes = (
        value_difference[
            valid_pairs
        ]
        / time_difference[
            valid_pairs
        ]
    )

    if slopes.size == 0:

        return np.nan

    return float(
        np.nanmedian(
            slopes
        )
    )


def mann_kendall_score_and_variance(
    values,
):
    """
    Mann–Kendall S score and tie-corrected variance.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    number_of_values = values.size

    score = 0.0

    for index in range(
        number_of_values - 1
    ):

        score += np.sign(
            values[
                index + 1:
            ]
            - values[
                index
            ]
        ).sum()

    _, tie_counts = np.unique(
        values,
        return_counts=True,
    )

    tie_term = np.sum(
        tie_counts
        * (
            tie_counts
            - 1
        )
        * (
            2
            * tie_counts
            + 5
        )
    )

    variance = (
        number_of_values
        * (
            number_of_values
            - 1
        )
        * (
            2
            * number_of_values
            + 5
        )
        - tie_term
    ) / 18.0

    return (
        float(
            score
        ),
        float(
            variance
        ),
    )


def autocorrelation(
    values,
    lag,
):
    """
    Pearson autocorrelation at one lag.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if lag <= 0 or lag >= values.size:

        return np.nan

    first = values[
        :-lag
    ]

    second = values[
        lag:
    ]

    first = first - np.mean(
        first
    )

    second = second - np.mean(
        second
    )

    denominator = np.sqrt(
        np.sum(
            first ** 2
        )
        * np.sum(
            second ** 2
        )
    )

    if denominator <= 0.0:

        return 0.0

    return float(
        np.sum(
            first
            * second
        )
        / denominator
    )


def hamed_rao_modified_mk(
    values,
    times,
    alpha=MK_ALPHA,
):
    """
    Hamed–Rao variance-corrected Mann–Kendall test.

    The autocorrelation correction is calculated from ranks of the
    Sen-slope-detrended series. Only rank autocorrelations outside
    the approximate 95% white-noise bounds contribute to the
    correction factor.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            values
        )
        & np.isfinite(
            times
        )
    )

    values = values[
        valid
    ]

    times = times[
        valid
    ]

    number_of_values = values.size

    if number_of_values < 4:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    score, variance = (
        mann_kendall_score_and_variance(
            values
        )
    )

    if variance <= 0.0:

        return (
            0.0,
            1.0,
            1.0,
        )

    slope = sen_slope(
        values,
        times,
    )

    intercept = np.nanmedian(
        values
        - slope
        * times
    )

    detrended = (
        values
        - (
            intercept
            + slope
            * times
        )
    )

    ranked = rankdata(
        detrended,
        method="average",
    )

    confidence_limit = (
        norm.ppf(
            1.0
            - alpha
            / 2.0
        )
        / np.sqrt(
            number_of_values
        )
    )

    weighted_autocorrelation_sum = 0.0

    for lag in range(
        1,
        number_of_values,
    ):

        rank_autocorrelation = (
            autocorrelation(
                ranked,
                lag,
            )
        )

        if (
            np.isfinite(
                rank_autocorrelation
            )
            and abs(
                rank_autocorrelation
            )
            > confidence_limit
        ):

            weighted_autocorrelation_sum += (
                (
                    number_of_values
                    - lag
                )
                * (
                    number_of_values
                    - lag
                    - 1
                )
                * (
                    number_of_values
                    - lag
                    - 2
                )
                * rank_autocorrelation
            )

    denominator = (
        number_of_values
        * (
            number_of_values
            - 1
        )
        * (
            number_of_values
            - 2
        )
    )

    correction_factor = (
        1.0
        + (
            2.0
            * weighted_autocorrelation_sum
            / denominator
        )
    )

    correction_factor = max(
        correction_factor,
        1.0e-6,
    )

    modified_variance = (
        variance
        * correction_factor
    )

    if score > 0.0:

        z_score = (
            score
            - 1.0
        ) / np.sqrt(
            modified_variance
        )

    elif score < 0.0:

        z_score = (
            score
            + 1.0
        ) / np.sqrt(
            modified_variance
        )

    else:

        z_score = 0.0

    p_value = (
        2.0
        * (
            1.0
            - norm.cdf(
                abs(
                    z_score
                )
            )
        )
    )

    return (
        float(
            z_score
        ),
        float(
            p_value
        ),
        float(
            correction_factor
        ),
    )


def benjamini_hochberg_qvalues(
    p_values,
):
    """
    Benjamini–Hochberg FDR-adjusted q values.
    """
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    q_values = np.full_like(
        p_values,
        np.nan,
        dtype=np.float64,
    )

    finite_mask = np.isfinite(
        p_values
    )

    finite_p = p_values[
        finite_mask
    ]

    number_of_tests = finite_p.size

    if number_of_tests == 0:

        return q_values

    order = np.argsort(
        finite_p
    )

    sorted_p = finite_p[
        order
    ]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=np.float64,
    )

    sorted_q = (
        sorted_p
        * number_of_tests
        / ranks
    )

    sorted_q = np.minimum.accumulate(
        sorted_q[
            ::-1
        ]
    )[
        ::-1
    ]

    sorted_q = np.clip(
        sorted_q,
        0.0,
        1.0,
    )

    finite_q = np.empty_like(
        sorted_q
    )

    finite_q[
        order
    ] = sorted_q

    q_values[
        finite_mask
    ] = finite_q

    return q_values


def grid_trend_statistics(
    annual_values,
    years,
    fixed_mask,
    variable_label,
):
    """
    Calculate grid-cell Sen slope and Hamed–Rao p values.

    Slopes are returned in cm decade^-1.
    """
    latitude_count = annual_values.shape[1]
    longitude_count = annual_values.shape[2]

    slope_map = np.full(
        (
            latitude_count,
            longitude_count,
        ),
        np.nan,
        dtype=np.float32,
    )

    p_map = np.full(
        (
            latitude_count,
            longitude_count,
        ),
        np.nan,
        dtype=np.float32,
    )

    correction_map = np.full(
        (
            latitude_count,
            longitude_count,
        ),
        np.nan,
        dtype=np.float32,
    )

    valid_positions = np.argwhere(
        fixed_mask
    )

    print(
        f"\nCalculating {variable_label} grid trends "
        f"for {valid_positions.shape[0]:,} cells..."
    )

    for position_index, (
        latitude_index,
        longitude_index,
    ) in enumerate(
        valid_positions
    ):

        series = annual_values[
            :,
            latitude_index,
            longitude_index,
        ]

        valid_years = np.isfinite(
            series
        )

        if valid_years.sum() < MIN_VALID_YEARS:

            continue

        selected_values = series[
            valid_years
        ]

        selected_years = years[
            valid_years
        ].astype(
            np.float64
        )

        slope_per_year = sen_slope(
            selected_values,
            selected_years,
        )

        _, p_value, correction_factor = (
            hamed_rao_modified_mk(
                selected_values,
                selected_years,
            )
        )

        slope_map[
            latitude_index,
            longitude_index,
        ] = np.float32(
            slope_per_year
            * 10.0
        )

        p_map[
            latitude_index,
            longitude_index,
        ] = np.float32(
            p_value
        )

        correction_map[
            latitude_index,
            longitude_index,
        ] = np.float32(
            correction_factor
        )

        if (
            (position_index + 1) % 500 == 0
            or position_index
            == valid_positions.shape[0] - 1
        ):

            print(
                f"  [{position_index + 1:4d}/"
                f"{valid_positions.shape[0]:4d}]"
            )

    q_map = np.full_like(
        p_map,
        np.nan,
        dtype=np.float32,
    )

    q_values = benjamini_hochberg_qvalues(
        p_map[
            fixed_mask
        ]
    )

    q_map[
        fixed_mask
    ] = q_values.astype(
        np.float32
    )

    significant_map = (
        np.isfinite(
            q_map
        )
        & (
            q_map
            < FDR_ALPHA
        )
    )

    return {
        "slope": slope_map,
        "p": p_map,
        "q": q_map,
        "significant": significant_map,
        "hamed_rao_factor": correction_map,
    }


sla_grid_statistics = grid_trend_statistics(
    sla_annual,
    analysis_years.astype(
        np.float64
    ),
    common_mask,
    "SLA",
)

total_grid_statistics = grid_trend_statistics(
    total_annual,
    analysis_years.astype(
        np.float64
    ),
    common_mask,
    "total steric",
)

thermo_grid_statistics = grid_trend_statistics(
    thermo_annual,
    analysis_years.astype(
        np.float64
    ),
    common_mask,
    "thermosteric",
)

halo_grid_statistics = grid_trend_statistics(
    halo_annual,
    analysis_years.astype(
        np.float64
    ),
    common_mask,
    "halosteric",
)


# 13. SECTOR MASKS, WEIGHTS AND BLOCK-BOOTSTRAP CIs
longitude_2d, latitude_2d = np.meshgrid(
    target_longitude,
    target_latitude,
)

area_weights = np.cos(
    np.deg2rad(
        latitude_2d
    )
).astype(
    np.float64
)


def longitude_sector_mask(
    longitude_values,
    minimum_longitude,
    maximum_longitude,
):
    """
    One-dimensional sector mask, including dateline-crossing sectors.
    """
    if (
        minimum_longitude
        <= maximum_longitude
    ):

        return (
            (
                longitude_values
                >= minimum_longitude
            )
            & (
                longitude_values
                < maximum_longitude
            )
        )

    return (
        (
            longitude_values
            >= minimum_longitude
        )
        | (
            longitude_values
            < maximum_longitude
        )
    )


def sector_mask(
    sector_information,
):
    """
    Two-dimensional sector mask on the common EN4 grid.
    """
    longitude_mask = longitude_sector_mask(
        target_longitude,
        sector_information[
            "lon_min"
        ],
        sector_information[
            "lon_max"
        ],
    )

    return (
        np.broadcast_to(
            longitude_mask[
                None,
                :
            ],
            common_mask.shape,
        )
        & common_mask
    )


def area_weighted_sector_series(
    annual_values,
    mask,
):
    """
    Area-weighted annual sector mean.
    """
    number_of_years = annual_values.shape[0]

    output = np.full(
        number_of_years,
        np.nan,
        dtype=np.float64,
    )

    fixed_weights = np.where(
        mask,
        area_weights,
        np.nan,
    )

    for year_index in range(
        number_of_years
    ):

        field = annual_values[
            year_index
        ]

        valid = (
            mask
            & np.isfinite(
                field
            )
        )

        if not np.any(
            valid
        ):

            continue

        denominator = np.nansum(
            fixed_weights[
                valid
            ]
        )

        if denominator <= 0.0:

            continue

        output[
            year_index
        ] = (
            np.nansum(
                field[
                    valid
                ]
                * fixed_weights[
                    valid
                ]
            )
            / denominator
        )

    return output


def moving_block_bootstrap_ci(
    annual_values,
    years,
    block_length=BOOTSTRAP_BLOCK_LENGTH_YEARS,
    number_of_bootstraps=BOOTSTRAP_REPLICATES,
    random_seed=BOOTSTRAP_SEED,
):
    """
    Residual circular moving-block bootstrap confidence interval.

    A Sen trend is fitted first. Residuals are sampled in contiguous
    circular blocks and added to the fitted trend. Each bootstrap
    realization is assigned the original sequential year positions.
    """
    annual_values = np.asarray(
        annual_values,
        dtype=np.float64,
    )

    years = np.asarray(
        years,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            annual_values
        )
        & np.isfinite(
            years
        )
    )

    annual_values = annual_values[
        valid
    ]

    years = years[
        valid
    ]

    number_of_years = annual_values.size

    if number_of_years < MIN_VALID_YEARS:

        return (
            np.nan,
            np.nan,
        )

    point_slope = sen_slope(
        annual_values,
        years,
    )

    intercept = np.nanmedian(
        annual_values
        - point_slope
        * years
    )

    fitted = (
        intercept
        + point_slope
        * years
    )

    residuals = (
        annual_values
        - fitted
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    bootstrap_slopes = np.full(
        number_of_bootstraps,
        np.nan,
        dtype=np.float64,
    )

    relative_time = np.arange(
        number_of_years,
        dtype=np.float64,
    )

    for bootstrap_index in range(
        number_of_bootstraps
    ):

        sampled_indices = []

        while len(
            sampled_indices
        ) < number_of_years:

            block_start = int(
                random_generator.integers(
                    0,
                    number_of_years,
                )
            )

            sampled_indices.extend(
                [
                    (
                        block_start
                        + offset
                    )
#                     % number_of_years
                    for offset in range(
                        block_length
                    )
                ]
            )

        sampled_indices = np.asarray(
            sampled_indices[
                :number_of_years
            ],
            dtype=np.int64,
        )

        bootstrap_values = (
            fitted
            + residuals[
                sampled_indices
            ]
        )

        bootstrap_slopes[
            bootstrap_index
        ] = (
            sen_slope(
                bootstrap_values,
                relative_time,
            )
            * 10.0
        )

    finite_slopes = bootstrap_slopes[
        np.isfinite(
            bootstrap_slopes
        )
    ]

    if finite_slopes.size == 0:

        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            np.nanpercentile(
                finite_slopes,
                2.5,
            )
        ),
        float(
            np.nanpercentile(
                finite_slopes,
                97.5,
            )
        ),
    )


annual_field_dictionary = OrderedDict([
    (
        "SLA",
        sla_annual,
    ),
    (
        "Total steric",
        total_annual,
    ),
    (
        "Thermosteric",
        thermo_annual,
    ),
    (
        "Halosteric",
        halo_annual,
    ),
])

sector_rows = []

print(
    "\nCalculating 13-sea trends and 3-year block-bootstrap CIs..."
)

for variable_index, (
    variable_label,
    annual_values,
) in enumerate(
    annual_field_dictionary.items()
):

    variable_rows = []

    for sea_index, sea_code in enumerate(
        SEA_CODES
    ):

        sea_information = SEA_SECTORS[
            sea_code
        ]

        mask = sector_mask(
            sea_information
        )

        annual_series = (
            area_weighted_sector_series(
                annual_values,
                mask,
            )
        )

        valid_year_mask = np.isfinite(
            annual_series
        )

        selected_values = annual_series[
            valid_year_mask
        ]

        selected_years = analysis_years[
            valid_year_mask
        ].astype(
            np.float64
        )

        if selected_values.size < MIN_VALID_YEARS:

            slope_decade = np.nan
            p_value = np.nan
            correction_factor = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        else:

            slope_decade = (
                sen_slope(
                    selected_values,
                    selected_years,
                )
                * 10.0
            )

            _, p_value, correction_factor = (
                hamed_rao_modified_mk(
                    selected_values,
                    selected_years,
                )
            )

            ci_lower, ci_upper = (
                moving_block_bootstrap_ci(
                    selected_values,
                    selected_years,
                    block_length=(
                        BOOTSTRAP_BLOCK_LENGTH_YEARS
                    ),
                    number_of_bootstraps=(
                        BOOTSTRAP_REPLICATES
                    ),
                    random_seed=(
                        BOOTSTRAP_SEED
                        + variable_index
                        * 100
                        + sea_index
                    ),
                )
            )

        variable_rows.append(
            {
                "variable": variable_label,
                "sea_code": sea_code,
                "sea_name": sea_information[
                    "name"
                ],
                "sen_slope_cm_decade": slope_decade,
                "ci_lower_cm_decade": ci_lower,
                "ci_upper_cm_decade": ci_upper,
                "hamed_rao_p": p_value,
                "hamed_rao_factor": correction_factor,
                "valid_years": int(
                    selected_values.size
                ),
                "common_grid_cells": int(
                    mask.sum()
                ),
            }
        )

    variable_p_values = np.asarray(
        [
            row[
                "hamed_rao_p"
            ]
            for row in variable_rows
        ],
        dtype=np.float64,
    )

    variable_q_values = (
        benjamini_hochberg_qvalues(
            variable_p_values
        )
    )

    for row, q_value in zip(
        variable_rows,
        variable_q_values,
    ):

        row[
            "fdr_q"
        ] = q_value

        row[
            "fdr_significant"
        ] = bool(
            np.isfinite(
                q_value
            )
            and q_value
            < FDR_ALPHA
        )

        sector_rows.append(
            row
        )

sector_table = pd.DataFrame(
    sector_rows
)

sector_table.to_csv(
    OUT_SECTOR_CSV,
    index=False,
)

print(
    "Saved sector statistics:",
    OUT_SECTOR_CSV,
)


# 14. SAVE GRID AND ANNUAL OUTPUTS
grid_output = xr.Dataset(
    coords={
        "latitude": (
            "latitude",
            target_latitude.astype(
                np.float32
            ),
        ),
        "longitude": (
            "longitude",
            target_longitude.astype(
                np.float32
            ),
        ),
    }
)

grid_statistics_dictionary = OrderedDict([
    (
        "sla",
        sla_grid_statistics,
    ),
    (
        "total_steric",
        total_grid_statistics,
    ),
    (
        "thermosteric",
        thermo_grid_statistics,
    ),
    (
        "halosteric",
        halo_grid_statistics,
    ),
])

for short_name, statistics in (
    grid_statistics_dictionary.items()
):

    grid_output[
        f"{short_name}_sen_slope"
    ] = (
        (
            "latitude",
            "longitude",
        ),
        statistics[
            "slope"
        ],
    )

    grid_output[
        f"{short_name}_hamed_rao_p"
    ] = (
        (
            "latitude",
            "longitude",
        ),
        statistics[
            "p"
        ],
    )

    grid_output[
        f"{short_name}_fdr_q"
    ] = (
        (
            "latitude",
            "longitude",
        ),
        statistics[
            "q"
        ],
    )

    grid_output[
        f"{short_name}_fdr_significant"
    ] = (
        (
            "latitude",
            "longitude",
        ),
        statistics[
            "significant"
        ].astype(
            np.int8
        ),
    )

    grid_output[
        f"{short_name}_hamed_rao_factor"
    ] = (
        (
            "latitude",
            "longitude",
        ),
        statistics[
            "hamed_rao_factor"
        ],
    )

    grid_output[
        f"{short_name}_sen_slope"
    ].attrs[
        "units"
    ] = "cm decade-1"

grid_output[
    "fixed_common_mask"
] = (
    (
        "latitude",
        "longitude",
    ),
    common_mask.astype(
        np.int8
    ),
)

grid_output.attrs.update(
    {
        "title": (
            "Figure 11 study-period grid-cell trends"
        ),
        "study_period": (
            "2008-2025"
        ),
        "trend_phrase": (
            "Study-period trend during 2008–2025"
        ),
        "deseasonalization": (
            "Calendar-month climatology removed before annual averaging"
        ),
        "trend_estimator": (
            "Sen slope"
        ),
        "significance_test": (
            "Hamed-Rao modified Mann-Kendall"
        ),
        "multiple_testing": (
            "Benjamini-Hochberg FDR"
        ),
        "FDR_alpha": FDR_ALPHA,
    }
)

grid_encoding = {
    variable_name: {
        "zlib": True,
        "complevel": 4,
    }
    for variable_name
    in grid_output.data_vars
}

grid_output.to_netcdf(
    OUT_GRID_NC,
    encoding=grid_encoding,
)

grid_output.close()

del grid_output

annual_output = xr.Dataset(
    {
        "sla_annual_anomaly": (
            (
                "year",
                "latitude",
                "longitude",
            ),
            sla_annual,
        ),
        "total_steric_annual_anomaly": (
            (
                "year",
                "latitude",
                "longitude",
            ),
            total_annual,
        ),
        "thermosteric_annual_anomaly": (
            (
                "year",
                "latitude",
                "longitude",
            ),
            thermo_annual,
        ),
        "halosteric_annual_anomaly": (
            (
                "year",
                "latitude",
                "longitude",
            ),
            halo_annual,
        ),
        "fixed_common_mask": (
            (
                "latitude",
                "longitude",
            ),
            common_mask.astype(
                np.int8
            ),
        ),
    },
    coords={
        "year": analysis_years,
        "latitude": target_latitude.astype(
            np.float32
        ),
        "longitude": target_longitude.astype(
            np.float32
        ),
    },
)

for variable_name in [
    "sla_annual_anomaly",
    "total_steric_annual_anomaly",
    "thermosteric_annual_anomaly",
    "halosteric_annual_anomaly",
]:

    annual_output[
        variable_name
    ].attrs[
        "units"
    ] = "cm"

annual_output.attrs.update(
    {
        "title": (
            "Deseasonalized annual anomalies used in Figure 11"
        ),
        "method": (
            "Grid-cell monthly climatology removed, then annual mean calculated"
        ),
    }
)

annual_encoding = {
    variable_name: {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
    }
    for variable_name
    in annual_output.data_vars
    if variable_name
    != "fixed_common_mask"
}

annual_encoding[
    "fixed_common_mask"
] = {
    "dtype": "int8",
    "zlib": True,
    "complevel": 4,
}

annual_output.to_netcdf(
    OUT_ANNUAL_NC,
    encoding=annual_encoding,
)

annual_output.close()

del annual_output

gc.collect()




# DATA PRODUCT GROUP 08: FIGURE 12 SECTOR-EOF NUMERICAL SUPPORT PRODUCTS (SOURCE WORKFLOW)
# Source: so_sealevel_paper_fig.py
# File name: Fig12_SLA_sector_matrix_216x13.csv
# File name: Fig12_steric_sector_matrix_216x13.csv
# File name: Fig12_EOF_sector_loadings_summary.csv
# File name: Fig12_EOF_PC_timeseries_summary.csv
# File name: Fig12_correlation_EOF_sensitivity_summary.csv

# FIGURE 12 — SECTOR-BASED EOF ANALYSIS (CORRECTED MAP VERSION)
#
# Main changes requested:
#   1) EOF panels are plotted as south-polar maps, not circular ring charts.
#   2) Antarctica / land mask comes from GEBCO bathymetry.
#   3) EOF loadings are displayed on a gridded ocean mesh by broadcasting
#      each sector loading across the bathymetry ocean cells that belong
#      to that sector.
#   4) No sea labels are drawn on the EOF maps.
#   5) PC-panel x ticks are rotated 45 degrees.
#   6) Layout spacing is tightened so labels and text do not overlap.
#
# Scientific settings:
#   • Input matrices: 216 months × 13 sectors
#   • Remove monthly climatology
#   • Detrend each sector
#   • Covariance EOFs = main analysis
#   • Correlation EOFs = supplementary sensitivity output
#   • PCs shown as 13-month running means
#
# Output folder:
#   /content/drive/MyDrive/SAM_Thesis/paper2



# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
    print("Package installation completed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import gc
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.path import Path as MplPath

import cartopy.crs as ccrs

warnings.filterwarnings("ignore", category=RuntimeWarning)



# 3. INPUT / OUTPUT PATHS
SLA_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "SLA_Antarctic_monthly_2008_2025.nc"
)

STERIC_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Processed/EN4_NetCDF_inventory/"
    "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
)

GEBCO_CANDIDATES = [
    "/content/drive/MyDrive/SAM_Thesis/Data/GEBCO_2024_CEC.nc",
    "/content/drive/MyDrive/SAM_Thesis/Data/GEBCO_2024_CF.nc",
]

OUTPUT_DIR = Path("/content/drive/MyDrive/SAM_Thesis/paper2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


OUT_SLA_MATRIX = OUTPUT_DIR / "Fig12_SLA_sector_matrix_216x13.csv"
OUT_STERIC_MATRIX = OUTPUT_DIR / "Fig12_steric_sector_matrix_216x13.csv"
OUT_EOF_LOADINGS = OUTPUT_DIR / "Fig12_EOF_sector_loadings_summary.csv"
OUT_EOF_PCS = OUTPUT_DIR / "Fig12_EOF_PC_timeseries_summary.csv"
OUT_SUPP_CORR = OUTPUT_DIR / "Fig12_correlation_EOF_sensitivity_summary.csv"
OUT_LOG = OUTPUT_DIR / "Fig12_sector_EOF_processing_log.txt"


# ================================================================
# 5. 13 SECTOR DEFINITIONS
# ================================================================
SEA_SECTORS = OrderedDict([
    ("WED", {"name": "Weddell Sea",         "lon_min":  -60.0, "lon_max":  -20.0}),
    ("KHV", {"name": "King Haakon VII Sea", "lon_min":  -20.0, "lon_max":   10.0}),
    ("RLS", {"name": "Riiser-Larsen Sea",   "lon_min":   10.0, "lon_max":   35.0}),
    ("LAZ", {"name": "Lazarev Sea",         "lon_min":   35.0, "lon_max":   60.0}),
    ("COS", {"name": "Cosmonauts Sea",      "lon_min":   60.0, "lon_max":   90.0}),
    ("COO", {"name": "Cooperation Sea",     "lon_min":   90.0, "lon_max":  115.0}),
    ("DAV", {"name": "Davis Sea",           "lon_min":  115.0, "lon_max":  130.0}),
    ("MAW", {"name": "Mawson Sea",          "lon_min":  130.0, "lon_max":  150.0}),
    ("DUR", {"name": "D'Urville Sea",       "lon_min":  150.0, "lon_max":  170.0}),
    ("SOM", {"name": "Somov Sea",           "lon_min":  170.0, "lon_max": -160.0}),
    ("ROS", {"name": "Ross Sea",            "lon_min": -160.0, "lon_max": -130.0}),
    ("AMU", {"name": "Amundsen Sea",        "lon_min": -130.0, "lon_max": -100.0}),
    ("BEL", {"name": "Bellingshausen Sea",  "lon_min": -100.0, "lon_max":  -60.0}),
])
SEA_CODES = list(SEA_SECTORS.keys())


# 6. HELPERS
def open_dataset_safely(file_path):
    attempts = []
    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = {
                "decode_times": True,
                "mask_and_scale": True,
                "cache": False,
            }
            if engine is not None:
                kwargs["engine"] = engine
            ds = xr.open_dataset(file_path, **kwargs)
            engine_name = "xarray-default" if engine is None else engine
            print(f"Opened {os.path.basename(file_path)} with engine={engine_name}")
            return ds
        except Exception as error:
            attempts.append(f"engine={engine}: {error}")

    raise RuntimeError(
        "Could not open dataset:\n"
        f"{file_path}\n\n"
        + "\n".join(attempts)
    )


def detect_coordinate(dataset, coordinate_type):
    aliases = {
        "lat": ["lat", "latitude", "nav_lat", "y"],
        "lon": ["lon", "longitude", "nav_lon", "x"],
        "time": ["time"],
    }

    candidates = list(dataset.coords) + list(dataset.variables)

    for name in candidates:
        lower = name.lower()
        da = dataset[name]
        standard_name = str(da.attrs.get("standard_name", "")).lower()
        axis = str(da.attrs.get("axis", "")).upper()
        units = str(da.attrs.get("units", "")).lower()

        if coordinate_type == "lat":
            if (
                lower in aliases["lat"]
                or standard_name == "latitude"
                or axis == "Y"
                or "degrees_north" in units
            ):
                return name

        elif coordinate_type == "lon":
            if (
                lower in aliases["lon"]
                or standard_name == "longitude"
                or axis == "X"
                or "degrees_east" in units
            ):
                return name

        elif coordinate_type == "time":
            if lower in aliases["time"] or np.issubdtype(da.dtype, np.datetime64):
                return name

    raise KeyError(f"Could not detect coordinate: {coordinate_type}")


def standardize_dataset(dataset):
    rename_map = {}

    lat_name = detect_coordinate(dataset, "lat")
    lon_name = detect_coordinate(dataset, "lon")
    time_name = None
    try:
        time_name = detect_coordinate(dataset, "time")
    except Exception:
        time_name = None

    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if time_name is not None and time_name != "time":
        rename_map[time_name] = "time"

    if rename_map:
        dataset = dataset.rename(rename_map)

    lon_values = (((dataset["lon"].astype(float) + 180.0) % 360.0) - 180.0)
    dataset = dataset.assign_coords(lon=lon_values)

    lon_values = np.asarray(dataset["lon"].values)
    _, unique_indices = np.unique(lon_values, return_index=True)
    dataset = dataset.isel(lon=np.sort(unique_indices))

    dataset = dataset.sortby("lon").sortby("lat")
    if "time" in dataset.coords:
        dataset = dataset.sortby("time")

    return dataset


def detect_main_variable(dataset, preferred_names):
    for name in preferred_names:
        if name in dataset.data_vars:
            return name

    candidates = []
    for name in dataset.data_vars:
        dims = set(dataset[name].dims)
        if {"lat", "lon"}.issubset(dims):
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print("Candidate variables:", candidates)
        return candidates[0]

    raise KeyError("Could not detect main data variable.")


def convert_to_cm(data_array):
    units = str(data_array.attrs.get("units", "")).strip().lower()

    if units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        out = data_array.astype(np.float32)
        out.attrs["units"] = "cm"
        return out

    out = (data_array.astype(np.float32) * 100.0)
    out.attrs["units"] = "cm"
    return out


def monthly_anomaly(time_series):
    climatology = time_series.groupby("time.month").mean("time", skipna=True)
    anomaly = time_series.groupby("time.month") - climatology
    return anomaly


def detrend_series_1d(values):
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(values.size, dtype=np.float64)
    valid = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)

    if valid.sum() < 2:
        return out

    coef = np.polyfit(x[valid], values[valid], deg=1)
    fit = coef[0] * x + coef[1]
    out[valid] = values[valid] - fit[valid]
    return out


def fill_small_time_gaps(series):
    s = pd.Series(series)
    s = s.interpolate(method="linear", limit_direction="both")
    if s.isna().any():
        s = s.fillna(s.mean())
    if s.isna().any():
        s = s.fillna(0.0)
    return s.values.astype(np.float64)


def sector_mask(lon2d, lat2d, lon_min, lon_max):
    lat_cond = (lat2d >= LAT_MIN) & (lat2d <= LAT_MAX)

    if lon_max < lon_min:
        lon_cond = (lon2d >= lon_min) | (lon2d < lon_max)
    else:
        lon_cond = (lon2d >= lon_min) & (lon2d < lon_max)

    return lat_cond & lon_cond


def weighted_sector_mean(field_2d, mask_2d, lat_2d):
    valid = np.isfinite(field_2d) & mask_2d
    if not np.any(valid):
        return np.nan

    weights = np.cos(np.deg2rad(lat_2d))
    weights = np.where(valid, weights, np.nan)

    numerator = np.nansum(field_2d * weights)
    denominator = np.nansum(weights)

    if not np.isfinite(denominator) or denominator == 0.0:
        return np.nan

    return float(numerator / denominator)


def build_sector_matrix(data_array, validity_fraction_threshold=0.70):
    lon2d, lat2d = np.meshgrid(
        data_array["lon"].values.astype(np.float64),
        data_array["lat"].values.astype(np.float64),
    )

    core_validity = np.isfinite(data_array).mean("time") >= validity_fraction_threshold
    core_validity = core_validity.values.astype(bool)

    time_values = pd.to_datetime(data_array["time"].values)
    sector_df = pd.DataFrame(index=time_values, columns=SEA_CODES, dtype=float)

    for sea_code, sea_info in SEA_SECTORS.items():
        mask = sector_mask(
            lon2d,
            lat2d,
            sea_info["lon_min"],
            sea_info["lon_max"],
        ) & core_validity

        for it in range(data_array.sizes["time"]):
            field = data_array.isel(time=it).values.astype(np.float64)
            sector_df.iloc[it, sector_df.columns.get_loc(sea_code)] = (
                weighted_sector_mean(field, mask, lat2d)
            )

    return sector_df


def preprocess_sector_matrix(sector_df):
    da = xr.DataArray(
        sector_df.values.astype(np.float64),
        coords={"time": sector_df.index, "sector": sector_df.columns},
        dims=("time", "sector"),
    )

    anomaly = monthly_anomaly(da)

    out = np.full(anomaly.shape, np.nan, dtype=np.float64)
    for j in range(anomaly.shape[1]):
        out[:, j] = detrend_series_1d(anomaly[:, j].values)
        out[:, j] = fill_small_time_gaps(out[:, j])

    return pd.DataFrame(out, index=sector_df.index, columns=sector_df.columns)


def run_eof(matrix_df, n_modes=3, use_correlation=False):
    X = matrix_df.values.astype(np.float64)

    if use_correlation:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        X = (X - mean) / std
    else:
        X = X - X.mean(axis=0, keepdims=True)

    U, singular_values, VT = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (singular_values ** 2) / (X.shape[0] - 1)
    explained_variance = 100.0 * eigenvalues / eigenvalues.sum()

    eofs = VT[:n_modes, :].copy()
    pcs = (U[:, :n_modes] * singular_values[:n_modes]).copy()

    for mode in range(n_modes):
        idx = int(np.nanargmax(np.abs(eofs[mode])))
        if eofs[mode, idx] < 0:
            eofs[mode] *= -1.0
            pcs[:, mode] *= -1.0

    pcs_std = pcs.copy()
    for mode in range(n_modes):
        std = pcs_std[:, mode].std(ddof=1)
        if std == 0.0:
            std = 1.0
        pcs_std[:, mode] = (pcs_std[:, mode] - pcs_std[:, mode].mean()) / std

    return {
        "eofs": eofs,
        "pcs": pcs,
        "pcs_std": pcs_std,
        "explained_variance": explained_variance[:n_modes],
    }


def running_mean_13(values_2d, time_index):
    return pd.DataFrame(values_2d, index=time_index).rolling(
        window=13,
        center=True,
        min_periods=1,
    ).mean()


def normalize_loading_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    max_abs = np.nanmax(np.abs(vector))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0
    return vector / max_abs


def find_gebco_file():
    for candidate in GEBCO_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a GEBCO bathymetry file.\nChecked:\n"
        + "\n".join(GEBCO_CANDIDATES)
    )


def prepare_bathymetry_grid(gebco_file):
    bathy_ds = standardize_dataset(open_dataset_safely(gebco_file))
    bathy_var = detect_main_variable(
        bathy_ds,
        ["elevation", "z", "depth", "Band1"],
    )

    bathy = bathy_ds[bathy_var].sel(lat=slice(LAT_MIN, LAT_MAX))

    # Reduce memory if GEBCO is very high resolution.
    n_lat = bathy.sizes["lat"]
    n_lon = bathy.sizes["lon"]

    lat_step = max(1, int(np.ceil(n_lat / 420)))
    lon_step = max(1, int(np.ceil(n_lon / 720)))

    bathy = bathy.isel(
        lat=slice(None, None, lat_step),
        lon=slice(None, None, lon_step),
    )

    lon2d, lat2d = np.meshgrid(
        bathy["lon"].values.astype(np.float64),
        bathy["lat"].values.astype(np.float64),
    )

    elevation = bathy.values.astype(np.float32)
    ocean_mask = np.isfinite(elevation) & (elevation < 0.0)
    land_mask = np.isfinite(elevation) & (elevation >= 0.0)

    return {
        "dataset": bathy_ds,
        "variable": bathy_var,
        "bathy": bathy,
        "lon2d": lon2d,
        "lat2d": lat2d,
        "elevation": elevation,
        "ocean_mask": ocean_mask,
        "land_mask": land_mask,
    }


def broadcast_sector_loading_to_grid(loading_vector, lon2d, lat2d, ocean_mask):
    field = np.full(lon2d.shape, np.nan, dtype=np.float32)
    norm_load = normalize_loading_vector(loading_vector)

    for idx, sea_code in enumerate(SEA_CODES):
        info = SEA_SECTORS[sea_code]
        mask = sector_mask(
            lon2d,
            lat2d,
            info["lon_min"],
            info["lon_max"],
        ) & ocean_mask
        field[mask] = norm_load[idx]

    return field


def circular_boundary_path():
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    center = np.array([0.5, 0.5])
    radius = 0.5
    vertices = np.vstack(
        [np.sin(theta), np.cos(theta)]
    ).T * radius + center
    return MplPath(vertices)


def add_polar_labels(ax):
    # Longitude labels around outer boundary
    label_specs = [
        (-180, -62.3, "180°"),
        (-120, -61.3, "120°W"),
        (-60, -61.3, "60°W"),
        (0, -61.0, "0°"),
        (60, -61.3, "60°E"),
        (120, -61.3, "120°E"),
        (180, -62.3, "180°"),
    ]

    for lon, lat, txt in label_specs:
        ax.text(
            lon,
            lat,
            txt,
            transform=ccrs.PlateCarree(),
            fontsize=FONT_MAP_LABEL,
            fontweight="bold",
            ha="center",
            va="center",
            clip_on=False,
            zorder=30,
        )

    # Latitude labels
    for lat, lon, txt in [(-60, 0, "60°S"), (-70, 0, "70°S"), (-80, 0, "80°S")]:
        ax.text(
            lon,
            lat,
            txt,
            transform=ccrs.PlateCarree(),
            fontsize=FONT_MAP_LABEL,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=30,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.55),
        )

    ax.text(
        0.50,
        0.50,
        "90°S",
        transform=ax.transAxes,
        fontsize=FONT_MAP_LABEL + 0.2,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=31,
    )


def style_polar_map_axis(ax):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_boundary(circular_boundary_path(), transform=ax.transAxes)

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(-180, 181, 60),
        ylocs=np.arange(-80, -59, 10),
        linewidth=0.50,
        color="0.55",
        linestyle="--",
        alpha=0.65,
        zorder=2,
    )

    add_polar_labels(ax)


def plot_eof_map(ax, field, bathy_dict, title):
    lon2d = bathy_dict["lon2d"]
    lat2d = bathy_dict["lat2d"]
    elevation = bathy_dict["elevation"]
    land_mask = bathy_dict["land_mask"]

    style_polar_map_axis(ax)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        field,
        transform=ccrs.PlateCarree(),
        cmap=EOF_CMAP,
        norm=EOF_NORM,
        shading="auto",
        zorder=4,
    )

    # Antarctica / land from bathymetry
    land_binary = np.where(land_mask, 1.0, np.nan)
    ax.contourf(
        lon2d,
        lat2d,
        land_binary,
        levels=[0.5, 1.5],
        colors=["0.85"],
        transform=ccrs.PlateCarree(),
        zorder=8,
    )

    # Coastline and bathymetric context
    ax.contour(
        lon2d,
        lat2d,
        elevation,
        levels=[0.0],
        colors="0.35",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
        zorder=9,
    )

    ax.contour(
        lon2d,
        lat2d,
        elevation,
        levels=[-1000.0],
        colors="0.65",
        linewidths=0.45,
        linestyles="--",
        transform=ccrs.PlateCarree(),
        zorder=7,
    )

    ax.set_title(title, fontsize=FONT_PANEL, fontweight="bold", pad=9)
    return mesh


def plot_pc_panel(ax, time_index, pcs_running_mean, explained_variance, title):
    years = pd.to_datetime(time_index)

    for mode in range(3):
        ax.plot(
            years,
            pcs_running_mean.iloc[:, mode].values,
            color=PC_COLORS[mode],
            lw=1.25,
            label=f"PC{mode+1} ({explained_variance[mode]:.1f}%)",
        )

    ax.axhline(0.0, color="0.35", lw=0.9, ls="--")
    ax.grid(True, axis="x", linestyle=":", color="0.72", lw=0.8)

    ax.set_ylabel("Amplitude (std. dev.)", fontsize=FONT_AXIS, fontweight="bold")
    ax.set_xlabel("Year", fontsize=FONT_AXIS, fontweight="bold")
    ax.set_title(title, fontsize=FONT_PANEL, fontweight="bold", pad=8)

    ax.tick_params(axis="both", labelsize=FONT_TICK)

    # X-axis spacing and rotation to prevent overlap
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    y_max = float(np.nanmax(np.abs(pcs_running_mean.values)))
    if not np.isfinite(y_max) or y_max == 0:
        y_max = 1.0
    y_lim = max(2.0, np.ceil(y_max * 1.15 * 2) / 2)
    ax.set_ylim(-y_lim, y_lim)

    ax.legend(
        loc="upper right",
        fontsize=FONT_LEGEND,
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.3,
    )


# 7. VERIFY INPUT FILES
for path in [SLA_FILE, STERIC_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found:\n{path}")

GEBCO_FILE = find_gebco_file()

print("\nUsing input files:")
print("SLA   :", SLA_FILE)
print("Steric:", STERIC_FILE)
print("GEBCO :", GEBCO_FILE)

# 8. READ DATASETS
sla_ds = standardize_dataset(open_dataset_safely(SLA_FILE))
steric_ds = standardize_dataset(open_dataset_safely(STERIC_FILE))

sla_var = detect_main_variable(sla_ds, ["sla", "SLA", "adt", "sla_anomaly"])
steric_var = detect_main_variable(steric_ds, ["total_steric_height", "steric", "eta_total"])

print("\nDetected variables:")
print("SLA   :", sla_var)
print("Steric:", steric_var)

sla = convert_to_cm(sla_ds[sla_var]).sel(lat=slice(LAT_MIN, LAT_MAX))
steric = convert_to_cm(steric_ds[steric_var]).sel(lat=slice(LAT_MIN, LAT_MAX))

common_time = np.intersect1d(
    pd.to_datetime(sla["time"].values).values,
    pd.to_datetime(steric["time"].values).values,
)

sla = sla.sel(time=common_time)
steric = steric.sel(time=common_time)

print("\nCommon analysis period:")
print(pd.to_datetime(sla["time"].values[0]))
print("to")
print(pd.to_datetime(sla["time"].values[-1]))
print("Months =", sla.sizes["time"])


# 9. BUILD 216 × 13 SECTOR MATRICES
print("\nBuilding 216 × 13 SLA sector matrix...")
sla_sector_raw = build_sector_matrix(sla, validity_fraction_threshold=0.70)

print("Building 216 × 13 steric sector matrix...")
steric_sector_raw = build_sector_matrix(steric, validity_fraction_threshold=0.70)

print("Removing monthly climatology and detrending each sector...")
sla_sector_preprocessed = preprocess_sector_matrix(sla_sector_raw)
steric_sector_preprocessed = preprocess_sector_matrix(steric_sector_raw)

sla_sector_preprocessed.to_csv(OUT_SLA_MATRIX, index_label="time")
steric_sector_preprocessed.to_csv(OUT_STERIC_MATRIX, index_label="time")


# 10. EOF ANALYSIS
print("\nRunning covariance EOF analysis...")
sla_eof = run_eof(sla_sector_preprocessed, n_modes=3, use_correlation=False)
steric_eof = run_eof(steric_sector_preprocessed, n_modes=3, use_correlation=False)

print("Running correlation EOF sensitivity analysis...")
sla_eof_corr = run_eof(sla_sector_preprocessed, n_modes=3, use_correlation=True)
steric_eof_corr = run_eof(steric_sector_preprocessed, n_modes=3, use_correlation=True)

sla_pcs_rm = running_mean_13(
    sla_eof["pcs_std"],
    sla_sector_preprocessed.index,
)
steric_pcs_rm = running_mean_13(
    steric_eof["pcs_std"],
    steric_sector_preprocessed.index,
)


# 12. SAVE NUMERICAL OUTPUTS
loading_rows = []
for family_name, eof_result in [
    ("SLA_covariance", sla_eof),
    ("Steric_covariance", steric_eof),
    ("SLA_correlation", sla_eof_corr),
    ("Steric_correlation", steric_eof_corr),
]:
    for mode in range(3):
        normalized = normalize_loading_vector(eof_result["eofs"][mode])
        for sea_index, sea_code in enumerate(SEA_CODES):
            loading_rows.append(
                {
                    "family": family_name,
                    "mode": mode + 1,
                    "explained_variance_percent": eof_result["explained_variance"][mode],
                    "sea_code": sea_code,
                    "loading_raw": eof_result["eofs"][mode, sea_index],
                    "loading_normalized_for_plot": normalized[sea_index],
                }
            )
pd.DataFrame(loading_rows).to_csv(OUT_EOF_LOADINGS, index=False)

pc_rows = []
for family_name, time_index, eof_result, running_mean_df in [
    ("SLA_covariance", sla_sector_preprocessed.index, sla_eof, sla_pcs_rm),
    ("Steric_covariance", steric_sector_preprocessed.index, steric_eof, steric_pcs_rm),
]:
    for mode in range(3):
        for i, time_value in enumerate(time_index):
            pc_rows.append(
                {
                    "family": family_name,
                    "time": pd.to_datetime(time_value),
                    "mode": mode + 1,
                    "explained_variance_percent": eof_result["explained_variance"][mode],
                    "pc_standardized": eof_result["pcs_std"][i, mode],
                    "pc_13month_running_mean": running_mean_df.iloc[i, mode],
                }
            )
pd.DataFrame(pc_rows).to_csv(OUT_EOF_PCS, index=False)

supp_rows = []
for family_name, eof_result in [
    ("SLA_correlation", sla_eof_corr),
    ("Steric_correlation", steric_eof_corr),
]:
    for mode in range(3):
        supp_rows.append(
            {
                "family": family_name,
                "mode": mode + 1,
                "explained_variance_percent": eof_result["explained_variance"][mode],
            }
        )
pd.DataFrame(supp_rows).to_csv(OUT_SUPP_CORR, index=False)




# DATA PRODUCT GROUP 09: FIGURE 12 FINAL TRUE-GRIDDED EOF / TABLE S7 PRODUCTS — FINAL PUBLICATION VERSION
# Source: sla_data_thesis.py
# File name: Fig12_TRUE_GRIDDED_EOF_FINAL_patterns.nc
# File name: Fig12_TRUE_GRIDDED_EOF_FINAL_PC_timeseries.csv
# File name: Fig12_TRUE_GRIDDED_EOF_FINAL_mode_summary.csv
# File name: Fig12_TRUE_GRIDDED_EOF_FINAL_sector_loadings_TableS7.csv
# File name: Fig12_SLA_EOF_gap_reconstruction_diagnostics.csv
# File name: Fig12_EOF_mask_diagnostics.csv

# FIGURE 12 — FINAL TRUE GRIDDED SPATIAL EOF ANALYSIS
# DUACS SLA vs EN4 0–1000 m TOTAL STERIC HEIGHT
# January 2008–December 2025 (216 months)
#
# FINAL PUBLICATION VERSION
#
# LAYOUT IS UNCHANGED:
#
# TOP ROW
#   (a) SLA EOF1
#   (b) SLA EOF2
#   (c) SLA EOF3
#   (d) SLA principal components
#
# BOTTOM ROW
#   (e) Total steric EOF1
#   (f) Total steric EOF2
#   (g) Total steric EOF3
#   (h) Total-steric principal components
#
# SCIENTIFIC CORRECTIONS
# 1. True gridded EOF: time × individual grid cells.
# 2. NO 216 × 13 sector EOF.
# 3. NO broadcasting one loading over an entire marginal sea.
# 4. DUACS SLA interpolated to EN4 1° grid.
# 5. Identical spatial support for SLA and total steric.
# 6. GEBCO water depth >=1000 m.
# 7. DUACS SLA valid for >=80% of 216 months (>=173 months).
# 8. EOF-specific refinement: >=3 valid observations for EACH
#    calendar month over 2008–2025.
# 9. EN4 total steric must be complete over retained grid cells.
# 10. Missing SLA anomalies inside retained Mask C reconstructed
#     by iterative low-rank EOF reconstruction.
# 11. Original observed SLA values are NEVER overwritten.
# 12. Calendar-month climatology removed.
# 13. Grid-cell linear trend removed.
# 14. Grid-cell area weighting used before EOF decomposition.
# 15. Spatial maps = covariance/regression loadings in
#     cm per 1-standard-deviation PC.
# 16. Excluded ocean cells = light gray.
#     Valid near-zero EOF loading = white.
# 17. Separate common scales for SLA EOF1–3 and total-steric EOF1–3.
# 18. Monthly PCs shown as thin transparent lines.
# 19. 13-month running means shown as thick lines.
# 20. North et al. mode-separation diagnostics exported and printed.
# 21. 1080-dpi PNG + vector PDF.



# 0. INSTALL / IMPORT

import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "xarray": "xarray",
    "cartopy": "cartopy",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
}

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
    print("Installation complete.")
else:
    print("All required packages are installed.")


import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.path import Path as MplPath

import cartopy.crs as ccrs

warnings.filterwarnings("ignore")



# 2. OUTPUT DIRECTORY

PAPER2_DIR = Path(
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)

PAPER2_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 3. INPUT FILES — USER-CONFIRMED PATHS

SLA_FILE = Path(
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "SLA_Antarctic_monthly_2008_2025.nc"
)

STERIC_FILE = Path(
    "/content/drive/MyDrive/SAM_Thesis/Processed/EN4_NetCDF_inventory/"
    "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
)

GEBCO_CANDIDATES = [
    Path(
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "GEBCO_2024_CEC.nc"
    ),
    Path(
        "/content/drive/MyDrive/SAM_Thesis/Data/"
        "GEBCO_2024_CF.nc"
    ),
]


# 4. OUTPUT FILES



OUT_NC = PAPER2_DIR / (
    "Fig12_TRUE_GRIDDED_EOF_FINAL_patterns.nc"
)

OUT_PC_CSV = PAPER2_DIR / (
    "Fig12_TRUE_GRIDDED_EOF_FINAL_PC_timeseries.csv"
)

OUT_MODE_CSV = PAPER2_DIR / (
    "Fig12_TRUE_GRIDDED_EOF_FINAL_mode_summary.csv"
)

OUT_SECTOR_CSV = PAPER2_DIR / (
    "Fig12_TRUE_GRIDDED_EOF_FINAL_sector_loadings_TableS7.csv"
)

OUT_RECON_CSV = PAPER2_DIR / (
    "Fig12_SLA_EOF_gap_reconstruction_diagnostics.csv"
)

OUT_MASK_CSV = PAPER2_DIR / (
    "Fig12_EOF_mask_diagnostics.csv"
)

OUT_LOG = PAPER2_DIR / (
    "Figure12_TRUE_GRIDDED_EOF_FINAL_log.txt"
)


# 5. INPUT FILE CHECK

if not SLA_FILE.exists():
    raise FileNotFoundError(
        f"SLA file not found:\n{SLA_FILE}"
    )

if not STERIC_FILE.exists():
    raise FileNotFoundError(
        f"Total-steric file not found:\n{STERIC_FILE}"
    )


def find_gebco_file():

    for p in GEBCO_CANDIDATES:

        if p.exists():
            return p

    raise FileNotFoundError(
        "No GEBCO file found.\nChecked:\n"
        + "\n".join(str(p) for p in GEBCO_CANDIDATES)
    )


GEBCO_FILE = find_gebco_file()


print("\nINPUT FILES")
print("=" * 90)
print("SLA          :", SLA_FILE)
print("Total steric :", STERIC_FILE)
print("GEBCO        :", GEBCO_FILE)


# 6. ANALYSIS SETTINGS

START_DATE = "2008-01-01"
END_DATE = "2025-12-31"

EXPECTED_TIME = pd.date_range(
    "2008-01-01",
    "2025-12-01",
    freq="MS"
)

N_MONTHS = len(EXPECTED_TIME)

if N_MONTHS != 216:
    raise RuntimeError(
        f"Expected 216 months; obtained {N_MONTHS}."
    )


LAT_MIN = -90.0
LAT_MAX = -60.0

LON_MIN = -180.0
LON_MAX = 180.0


# EN4 main steric layer is 0–1000 m.
MIN_DEPTH_M = 1000.0


# Manuscript Mask C availability criterion

MIN_SLA_VALID_FRACTION = 0.80

MIN_SLA_VALID_MONTHS = int(
    np.ceil(
        MIN_SLA_VALID_FRACTION
        * N_MONTHS
    )
)

# 173 months for 216-month record.
print(
    f"Minimum SLA availability = "
    f"{MIN_SLA_VALID_MONTHS}/{N_MONTHS} months"
)


# EOF-specific refinement.
#
# Need enough data for each monthly climatological mean.

MIN_VALID_PER_CALENDAR_MONTH = 3


# Regridding

SLA_REGRID_METHOD = "linear"


# EOF modes

N_EOF_MODES = 3


# Iterative gap reconstruction settings
#
# These modes are ONLY for reconstruction.
# They are not the number of modes reported in the paper.

RECONSTRUCTION_MODES = 10

RECON_MAX_ITER = 40

RECON_TOL = 1.0e-5


# Unit conversion
#
# None = detect automatically.
#
# Set:
#   100.0 if source = metres
#   1.0   if source = centimetres

SLA_SCALE_TO_CM = None
STERIC_SCALE_TO_CM = None


# 8. ANTARCTIC MARGINAL-SEA DEFINITIONS
#
# NOT used to calculate EOFs.
#
# Used only for:
#   • subtle boundaries
#   • regional loading summary for Table S7

SEA_SECTORS = OrderedDict([

    ("WED", {
        "name": "Weddell Sea",
        "lon_min": -60.0,
        "lon_max": -20.0,
    }),

    ("KHV", {
        "name": "King Haakon VII Sea",
        "lon_min": -20.0,
        "lon_max": 0.0,
    }),

    ("RLS", {
        "name": "Riiser-Larsen Sea",
        "lon_min": 0.0,
        "lon_max": 10.0,
    }),

    ("LAZ", {
        "name": "Lazarev Sea",
        "lon_min": 10.0,
        "lon_max": 30.0,
    }),

    ("COS", {
        "name": "Cosmonauts Sea",
        "lon_min": 30.0,
        "lon_max": 50.0,
    }),

    ("COO", {
        "name": "Cooperation Sea",
        "lon_min": 50.0,
        "lon_max": 70.0,
    }),

    ("DAV", {
        "name": "Davis Sea",
        "lon_min": 70.0,
        "lon_max": 90.0,
    }),

    ("MAW", {
        "name": "Mawson Sea",
        "lon_min": 90.0,
        "lon_max": 130.0,
    }),

    ("DUR", {
        "name": "D'Urville Sea",
        "lon_min": 130.0,
        "lon_max": 150.0,
    }),

    ("SOM", {
        "name": "Somov Sea",
        "lon_min": 150.0,
        "lon_max": 170.0,
    }),

    ("ROS", {
        "name": "Ross Sea",
        "lon_min": 170.0,
        "lon_max": -130.0,
    }),

    ("AMU", {
        "name": "Amundsen Sea",
        "lon_min": -130.0,
        "lon_max": -100.0,
    }),

    ("BEL", {
        "name": "Bellingshausen Sea",
        "lon_min": -100.0,
        "lon_max": -60.0,
    }),

])


# 9. NETCDF OPENING

def open_dataset_safely(path):

    errors = []

    for engine in [
        None,
        "netcdf4",
        "h5netcdf",
        "scipy",
    ]:

        try:

            kwargs = dict(
                decode_times=True,
                mask_and_scale=True,
                cache=False,
            )

            if engine is not None:
                kwargs["engine"] = engine

            ds = xr.open_dataset(
                path,
                **kwargs
            )

            print(
                f"Opened {Path(path).name} "
                f"with {engine or 'xarray-default'}"
            )

            return ds

        except Exception as exc:

            errors.append(
                f"{engine or 'default'}: {exc}"
            )

    raise RuntimeError(
        f"Could not open:\n{path}\n\n"
        + "\n".join(errors)
    )


# 10. COORDINATE DETECTION

def detect_coordinate(ds, kind):

    aliases = {

        "lat": [
            "lat",
            "latitude",
            "nav_lat",
            "y",
        ],

        "lon": [
            "lon",
            "longitude",
            "nav_lon",
            "x",
        ],

        "time": [
            "time",
            "date",
            "datetime",
        ],
    }

    for name in list(ds.coords) + list(ds.variables):

        da = ds[name]

        lower = name.lower()

        standard_name = str(
            da.attrs.get(
                "standard_name",
                ""
            )
        ).lower()

        units = str(
            da.attrs.get(
                "units",
                ""
            )
        ).lower()

        axis = str(
            da.attrs.get(
                "axis",
                ""
            )
        ).upper()

        if kind == "lat":

            if (
                lower in aliases["lat"]
                or standard_name == "latitude"
                or axis == "Y"
                or "degrees_north" in units
            ):
                return name

        elif kind == "lon":

            if (
                lower in aliases["lon"]
                or standard_name == "longitude"
                or axis == "X"
                or "degrees_east" in units
            ):
                return name

        elif kind == "time":

            if (
                lower in aliases["time"]
                or standard_name == "time"
                or axis == "T"
            ):
                return name

    raise KeyError(
        f"Could not detect {kind} coordinate."
    )


# 11. STANDARDIZE DATASET

def standardize_dataset(
    ds,
    require_time=True,
):

    rename = {}

    lat_name = detect_coordinate(
        ds,
        "lat"
    )

    lon_name = detect_coordinate(
        ds,
        "lon"
    )

    if lat_name != "lat":
        rename[lat_name] = "lat"

    if lon_name != "lon":
        rename[lon_name] = "lon"

    if require_time:

        time_name = detect_coordinate(
            ds,
            "time"
        )

        if time_name != "time":
            rename[time_name] = "time"

    if rename:
        ds = ds.rename(rename)

    # Longitude to -180 ... +180.
    longitude = (
        (
            ds["lon"].astype(float)
            + 180.0
        )
         % 360.0
    ) - 180.0

    ds = ds.assign_coords(
        lon=longitude
    )

    # Remove duplicate longitudes.
    lon_values = np.asarray(
        ds["lon"].values
    )

    _, unique_idx = np.unique(
        lon_values,
        return_index=True
    )

    ds = ds.isel(
        lon=np.sort(
            unique_idx
        )
    )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    return ds


# 12. DATA VARIABLE DETECTION

SLA_VARIABLE_NAMES = [
    "sla",
    "SLA",
    "sea_level_anomaly",
    "sea_surface_height_anomaly",
    "zos",
]

STERIC_VARIABLE_NAMES = [
    "total_steric_height",
    "total_steric",
    "steric_total",
    "eta_total",
    "steric_height",
    "steric",
    "total",
]


def detect_time_lat_lon_variable(
    ds,
    preferred,
    label,
):

    for name in preferred:

        if name in ds.data_vars:

            if {
                "time",
                "lat",
                "lon",
            }.issubset(
                set(
                    ds[name].dims
                )
            ):

                print(
                    f"{label} variable: {name}"
                )

                return name

    candidates = []

    for name in ds.data_vars:

        if {
            "time",
            "lat",
            "lon",
        }.issubset(
            set(
                ds[name].dims
            )
        ):

            candidates.append(
                name
            )

    if len(candidates) == 1:

        print(
            f"{label} variable automatically selected: "
            f"{candidates[0]}"
        )

        return candidates[0]

    print(
        f"\nAvailable {label} variables:"
    )

    for name in ds.data_vars:

        print(
            " ",
            name,
            ds[name].dims,
            ds[name].attrs.get(
                "units",
                ""
            )
        )

    raise KeyError(
        f"Could not uniquely identify {label}. "
        f"Candidates={candidates}"
    )


# 13. REDUCE TO time × lat × lon

def squeeze_to_3d(da):

    extra_dims = [
        dim
        for dim in da.dims
        if dim not in [
            "time",
            "lat",
            "lon",
        ]
    ]

    for dim in extra_dims:

        if da.sizes[dim] == 1:

            da = da.isel(
                {
                    dim: 0
                },
                drop=True
            )

        else:

            raise ValueError(
                f"{da.name}: extra dimension "
                f"{dim!r} has size {da.sizes[dim]}."
            )

    return da.transpose(
        "time",
        "lat",
        "lon"
    )


# 14. EXACT MONTHLY SERIES

def make_exact_monthly(
    da,
    label,
):

    da = da.sel(
        time=slice(
            START_DATE,
            END_DATE
        )
    )

    da = da.resample(
        time="MS"
    ).mean(
        skipna=True
    )

    da = da.reindex(
        time=EXPECTED_TIME
    )

    if da.sizes["time"] != N_MONTHS:

        raise RuntimeError(
            f"{label}: expected {N_MONTHS} months, "
            f"found {da.sizes['time']}."
        )

    monthly_valid = (
        da.notnull()
        .sum(
            dim=[
                "lat",
                "lon"
            ]
        )
        .values
    )

    missing_entire_month = np.where(
        monthly_valid == 0
    )[0]

    if len(
        missing_entire_month
    ) > 0:

        dates = [
            EXPECTED_TIME[i].strftime(
                "%Y-%m"
            )
            for i in missing_entire_month
        ]

        raise RuntimeError(
            f"{label}: completely missing months:\n"
            + "\n".join(dates)
        )

    print(
        f"{label}: {da.sizes['time']} monthly fields"
    )

    return da


# 15. CONVERT TO CENTIMETRES

def convert_to_cm(
    da,
    label,
    manual_scale=None,
):

    da = da.astype(
        np.float64
    )

    units = str(
        da.attrs.get(
            "units",
            ""
        )
    ).strip().lower()

    print(
        f"{label}: source units = {units!r}"
    )

    if manual_scale is not None:

        result = (
            da
            * float(
                manual_scale
            )
        )

        result.attrs["units"] = "cm"

        return result

    if (
        units == "cm"
        or "centimeter" in units
        or "centimetre" in units
    ):

        result = da.copy()

        result.attrs["units"] = "cm"

        return result

    if units in [
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    ]:

        result = da * 100.0

        result.attrs["units"] = "cm"

        return result

    finite = np.asarray(
        da.values,
        dtype=float
    )

    finite = finite[
        np.isfinite(
            finite
        )
    ]

    if finite.size == 0:

        raise ValueError(
            f"{label}: no finite values."
        )

    p99 = np.nanpercentile(
        np.abs(
            finite
        ),
        99
    )

    if p99 < 2.0:

        print(
            f"{label}: ambiguous units; "
            "magnitude suggests metres -> converting to cm."
        )

        result = da * 100.0

    else:

        print(
            f"{label}: ambiguous units; "
            "magnitude suggests centimetres."
        )

        result = da.copy()

    result.attrs["units"] = "cm"

    return result


# 16. PERIODIC LONGITUDE EXTENSION

def periodic_longitude_extension(da):

    left = da.assign_coords(
        lon=da["lon"] - 360.0
    )

    right = da.assign_coords(
        lon=da["lon"] + 360.0
    )

    result = xr.concat(
        [
            left,
            da,
            right,
        ],
        dim="lon"
    )

    return result.sortby(
        "lon"
    )


# 17. REGRID SLA TO EN4 GRID

def regrid_to_target(
    source,
    target_lat,
    target_lon,
    method="linear",
):

    extended = periodic_longitude_extension(
        source
    )

    return extended.interp(
        lat=target_lat,
        lon=target_lon,
        method=method,
    ).transpose(
        "time",
        "lat",
        "lon"
    )


# 18. GEBCO

def detect_gebco_variable(ds):

    preferred = [
        "elevation",
        "z",
        "depth",
        "Band1",
    ]

    for name in preferred:

        if name in ds.data_vars:

            if {
                "lat",
                "lon"
            }.issubset(
                set(
                    ds[name].dims
                )
            ):

                return name

    candidates = [
        name
        for name in ds.data_vars
        if {
            "lat",
            "lon"
        }.issubset(
            set(
                ds[name].dims
            )
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Could not identify GEBCO variable. "
        f"Candidates={candidates}"
    )


def prepare_gebco(path):

    ds = standardize_dataset(
        open_dataset_safely(
            path
        ),
        require_time=False,
    )

    variable = detect_gebco_variable(
        ds
    )

    print(
        f"GEBCO variable: {variable}"
    )

    bathy = ds[
        variable
    ].sel(
        lat=slice(
            LAT_MIN,
            LAT_MAX
        )
    )

    # Reduced copy only for plotting.
    lat_step = max(
        1,
        int(
            np.ceil(
                bathy.sizes["lat"]
                / 420
            )
        )
    )

    lon_step = max(
        1,
        int(
            np.ceil(
                bathy.sizes["lon"]
                / 720
            )
        )
    )

    plot_bathy = bathy.isel(
        lat=slice(
            None,
            None,
            lat_step
        ),
        lon=slice(
            None,
            None,
            lon_step
        ),
    )

    lon2d, lat2d = np.meshgrid(
        plot_bathy["lon"].values,
        plot_bathy["lat"].values,
    )

    elevation = np.asarray(
        plot_bathy.values,
        dtype=np.float32
    )

    land_mask = (
        np.isfinite(
            elevation
        )
        & (
            elevation >= 0.0
        )
    )

    return {

        "full":
            bathy,

        "plot":
            plot_bathy,

        "lon2d":
            lon2d,

        "lat2d":
            lat2d,

        "elevation":
            elevation,

        "land_mask":
            land_mask,

        "variable":
            variable,
    }


# 19. GRID-CELL AREA

EARTH_RADIUS_M = 6_371_000.0


def coordinate_edges(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if values.size < 2:
        raise ValueError(
            "At least two coordinate centres required."
        )

    middle = (
        values[:-1]
        + values[1:]
    ) / 2.0

    first = (
        values[0]
        - (
            middle[0]
            - values[0]
        )
    )

    last = (
        values[-1]
        + (
            values[-1]
            - middle[-1]
        )
    )

    return np.concatenate(
        [
            [first],
            middle,
            [last],
        ]
    )


def calculate_cell_area(
    lat,
    lon,
):

    lat_edges = coordinate_edges(
        lat
    )

    lon_edges = coordinate_edges(
        lon
    )

    lat_edges = np.clip(
        lat_edges,
        -90.0,
        90.0
    )

    lat1 = np.deg2rad(
        lat_edges[:-1]
    )

    lat2 = np.deg2rad(
        lat_edges[1:]
    )

    dlon = np.abs(
        np.diff(
            np.deg2rad(
                lon_edges
            )
        )
    )

    area = (
        EARTH_RADIUS_M ** 2
        * np.abs(
            np.sin(
                lat2
            )
            - np.sin(
                lat1
            )
        )[:, None]
        * dlon[None, :]
    )

    return area


# 20. CALENDAR-MONTH SUPPORT

def calendar_month_support(
    values,
    times,
):

    months = pd.DatetimeIndex(
        times
    ).month.values

    support = []

    for month in range(
        1,
        13
    ):

        idx = (
            months == month
        )

        support.append(
            np.sum(
                np.isfinite(
                    values[
                        idx,
                        :,
                        :
                    ]
                ),
                axis=0,
            )
        )

    return np.stack(
        support,
        axis=0
    )


# 21. PREPROCESS WITH MISSING VALUES

def preprocess_with_missing(
    X,
    times,
):

    """
    Calendar-month climatology and linear trend are removed
    using only available observations.

    Missing entries remain NaN.
    """

    X = np.asarray(
        X,
        dtype=np.float64
    ).copy()

    ntime, nspace = X.shape

    months = pd.DatetimeIndex(
        times
    ).month.values


    # A. REMOVE CALENDAR-MONTH CLIMATOLOGY

    for month in range(
        1,
        13
    ):

        idx = (
            months == month
        )

        climatology = np.nanmean(
            X[
                idx,
                :
            ],
            axis=0,
        )

        X[
            idx,
            :
        ] = (
            X[
                idx,
                :
            ]
            - climatology[
                None,
                :
            ]
        )


    # B. LINEAR TREND USING AVAILABLE VALUES

    t = np.arange(
        ntime,
        dtype=np.float64
    )

    T = t[:, None]

    valid = np.isfinite(
        X
    )

    n = valid.sum(
        axis=0
    ).astype(
        np.float64
    )

    sum_t = np.sum(
        np.where(
            valid,
            T,
            0.0
        ),
        axis=0,
    )

    sum_x = np.nansum(
        X,
        axis=0
    )

    sum_tt = np.sum(
        np.where(
            valid,
            T ** 2,
            0.0
        ),
        axis=0,
    )

    sum_tx = np.nansum(
        X * T,
        axis=0,
    )

    denominator = (
        sum_tt
        - (
            sum_t ** 2
            / n
        )
    )

    slope = np.zeros(
        nspace,
        dtype=np.float64
    )

    good = (
        np.isfinite(
            denominator
        )
        & (
            denominator > 0
        )
    )

    slope[
        good
    ] = (
        (
            sum_tx[
                good
            ]
            - (
                sum_t[
                    good
                ]
                * sum_x[
                    good
                ]
                / n[
                    good
                ]
            )
        )
        / denominator[
            good
        ]
    )

    intercept = (
        sum_x
        - slope
        * sum_t
    ) / n

    trend = (
        intercept[
            None,
            :
        ]
        + T
        * slope[
            None,
            :
        ]
    )

    X = X - trend

    residual_mean = np.nanmean(
        X,
        axis=0
    )

    X = (
        X
        - residual_mean[
            None,
            :
        ]
    )

    return X


# 22. FINAL COMPLETE-DATA PREPROCESSING

def preprocess_complete(
    X,
    times,
):

    if not np.all(
        np.isfinite(
            X
        )
    ):

        raise ValueError(
            "Complete preprocessing received NaNs."
        )

    X = np.asarray(
        X,
        dtype=np.float64
    ).copy()

    months = pd.DatetimeIndex(
        times
    ).month.values


    # Monthly climatology
    for month in range(
        1,
        13
    ):

        idx = (
            months == month
        )

        X[
            idx,
            :
        ] -= np.mean(
            X[
                idx,
                :
            ],
            axis=0,
            keepdims=True,
        )


    # Linear trend
    t = np.arange(
        X.shape[0],
        dtype=np.float64
    )

    t -= np.mean(
        t
    )

    denominator = np.sum(
        t ** 2
    )

    slope = np.sum(
        t[
            :,
            None
        ]
        * X,
        axis=0,
    ) / denominator

    X -= (
        t[
            :,
            None
        ]
        * slope[
            None,
            :
        ]
    )

    X -= np.mean(
        X,
        axis=0,
        keepdims=True,
    )

    return X


# 23. ITERATIVE EOF RECONSTRUCTION OF MISSING SLA ANOMALIES

def iterative_eof_reconstruction(
    anomaly_matrix,
    area_vector,
    n_modes=10,
    max_iter=40,
    tolerance=1.0e-5,
):

    """
    Iterative low-rank area-weighted EOF reconstruction.

    ONLY originally missing anomaly entries are updated.

    Observed values remain fixed.
    """

    X = np.asarray(
        anomaly_matrix,
        dtype=np.float64
    )

    missing = ~np.isfinite(
        X
    )

    observed = ~missing

    n_missing = int(
        missing.sum()
    )

    if n_missing == 0:

        print(
            "No SLA gaps require EOF reconstruction."
        )

        return (
            X.copy(),
            {
                "iterations": 0,
                "converged": True,
                "relative_change": 0.0,
                "missing_values": 0,
                "reconstruction_modes": 0,
            }
        )


    Xfilled = X.copy()

    # Since this is already an anomaly matrix, initial missing anomaly = 0.
    Xfilled[
        missing
    ] = 0.0


    area_norm = (
        area_vector
        / np.nanmean(
            area_vector
        )
    )

    sqrt_area = np.sqrt(
        area_norm
    )


    observed_values = X[
        observed
    ]

    observed_std = np.nanstd(
        observed_values
    )

    if (
        not np.isfinite(
            observed_std
        )
        or observed_std <= 0
    ):

        observed_std = 1.0


    k = min(
        int(
            n_modes
        ),
        Xfilled.shape[0] - 1,
        Xfilled.shape[1],
    )


    print(
        "\nIterative SLA EOF reconstruction"
    )

    print(
        f"Missing anomaly entries : {n_missing:,}"
    )

    print(
        f"Reconstruction modes    : {k}"
    )


    converged = False

    last_relative_change = np.nan


    for iteration in range(
        1,
        max_iter + 1
    ):


        # Area-weighted matrix

        Xw = (
            Xfilled
            * sqrt_area[
                None,
                :
            ]
        )


        # Temporal covariance matrix:
        # efficient because ntime = 216

        covariance_time = (
            Xw
            @ Xw.T
        ) / (
            Xw.shape[0]
            - 1
        )


        eigvals, eigvecs = np.linalg.eigh(
            covariance_time
        )

        order = np.argsort(
            eigvals
        )[
            ::-1
        ]

        U = eigvecs[
            :,
            order[
                :k
            ]
        ]


        # Rank-k reconstruction in area-weighted space

        reconstructed_w = (
            U
            @ (
                U.T
                @ Xw
            )
        )


        reconstructed = (
            reconstructed_w
            / sqrt_area[
                None,
                :
            ]
        )


        previous = Xfilled[
            missing
        ].copy()


        # ONLY update missing entries.
        Xfilled[
            missing
        ] = reconstructed[
            missing
        ]


        change = np.sqrt(
            np.mean(
                (
                    Xfilled[
                        missing
                    ]
                    - previous
                ) ** 2
            )
        )


        relative_change = (
            change
            / observed_std
        )

        last_relative_change = float(
            relative_change
        )


        if (
            iteration == 1
            or iteration % 5 == 0
            or relative_change < tolerance
        ):

            print(
                f"Iteration {iteration:02d}: "
                f"relative change = "
                f"{relative_change:.8e}"
            )


        if relative_change < tolerance:

            converged = True

            print(
                f"Converged after {iteration} iterations."
            )

            break


    if not converged:

        print(
            "WARNING: reconstruction reached maximum iterations."
        )


    # Verify observed values remained unchanged.
    observed_difference = np.nanmax(
        np.abs(
            Xfilled[
                observed
            ]
            - X[
                observed
            ]
        )
    )

    if observed_difference > 1.0e-12:

        raise RuntimeError(
            "Observed SLA values changed during reconstruction."
        )


    diagnostics = {

        "iterations":
            iteration,

        "converged":
            bool(
                converged
            ),

        "relative_change":
            last_relative_change,

        "missing_values":
            n_missing,

        "reconstruction_modes":
            k,
    }


    return (
        Xfilled,
        diagnostics
    )


# 24. AREA-WEIGHTED COVARIANCE EOF

def calculate_gridded_eof(
    X,
    cell_area_vector,
    spatial_mask,
    nlat,
    nlon,
    family,
):

    """
    X = complete deseasonalized and detrended matrix:
        time × grid cells

    EOF weighting:
        Xw = X * sqrt(grid-cell area)

    Spatial pattern:
        covariance/regression of X on standardized PC

    Units:
        cm per 1-standard-deviation PC
    """

    if not np.all(
        np.isfinite(
            X
        )
    ):

        raise ValueError(
            f"{family}: EOF matrix contains NaN/Inf."
        )


    ntime, nspace = X.shape


    area_norm = (
        cell_area_vector
        / np.mean(
            cell_area_vector
        )
    )

    sqrt_area = np.sqrt(
        area_norm
    )


    Xw = (
        X
        * sqrt_area[
            None,
            :
        ]
    )


    # Temporal covariance EOF

    C = (
        Xw
        @ Xw.T
    ) / (
        ntime
        - 1
    )


    eigenvalues, U = np.linalg.eigh(
        C
    )


    order = np.argsort(
        eigenvalues
    )[
        ::-1
    ]


    eigenvalues = eigenvalues[
        order
    ]

    U = U[
        :,
        order
    ]


    eigenvalues = np.maximum(
        eigenvalues,
        0.0
    )


    explained = (
        eigenvalues
        / np.sum(
            eigenvalues
        )
        * 100.0
    )


    # PCs

    singular = np.sqrt(
        eigenvalues
        * (
            ntime - 1
        )
    )


    PCs_raw = (
        U[
            :,
            :N_EOF_MODES
        ]
        * singular[
            None,
            :N_EOF_MODES
        ]
    )


    PCs_std = (
        PCs_raw
        - np.mean(
            PCs_raw,
            axis=0,
            keepdims=True
        )
    )


    PCs_std /= np.std(
        PCs_std,
        axis=0,
        ddof=1,
        keepdims=True
    )


    # Physical covariance/regression patterns

    patterns = (
        X.T
        @ PCs_std
    ) / (
        ntime
        - 1
    )


    # Deterministic sign orientation
    #
    # EOF sign has no intrinsic physical meaning.

    for mode in range(
        N_EOF_MODES
    ):

        strongest_index = np.argmax(
            np.abs(
                patterns[
                    :,
                    mode
                ]
            )
        )

        if (
            patterns[
                strongest_index,
                mode
            ]
            < 0
        ):

            patterns[
                :,
                mode
            ] *= -1.0

            PCs_std[
                :,
                mode
            ] *= -1.0

            PCs_raw[
                :,
                mode
            ] *= -1.0


    # Restore spatial patterns

    flat_mask = spatial_mask.ravel()


    full = np.full(
        (
            N_EOF_MODES,
            nlat * nlon
        ),
        np.nan,
        dtype=np.float64
    )


    full[
        :,
        flat_mask
    ] = patterns.T


    patterns_grid = full.reshape(
        N_EOF_MODES,
        nlat,
        nlon
    )


    # North et al. sampling error
    #
    # Uses lag-1 effective sample size for each PC.

    summary_rows = []


    for mode in range(
        N_EOF_MODES
    ):


        pc = PCs_std[
            :,
            mode
        ]


        r1 = np.corrcoef(
            pc[:-1],
            pc[1:]
        )[
            0,
            1
        ]


        if not np.isfinite(
            r1
        ):

            r1 = 0.0


        r1_for_neff = np.clip(
            r1,
            -0.95,
            0.95
        )


        n_eff = (
            ntime
            * (
                1.0
                - r1_for_neff
            )
            / (
                1.0
                + r1_for_neff
            )
        )


        n_eff = float(
            np.clip(
                n_eff,
                3.0,
                ntime
            )
        )


        north_eigen_error = (
            eigenvalues[
                mode
            ]
            * np.sqrt(
                2.0
                / n_eff
            )
        )


        north_variance_error = (
            explained[
                mode
            ]
            * np.sqrt(
                2.0
                / n_eff
            )
        )


        summary_rows.append({

            "family":
                family,

            "mode":
                mode + 1,

            "explained_variance_percent":
                explained[
                    mode
                ],

            "eigenvalue":
                eigenvalues[
                    mode
                ],

            "pc_lag1_autocorrelation":
                r1,

            "effective_sample_size":
                n_eff,

            "north_eigenvalue_error":
                north_eigen_error,

            "north_explained_variance_error_percent":
                north_variance_error,

            "north_eigenvalue_lower":
                eigenvalues[
                    mode
                ]
                - north_eigen_error,

            "north_eigenvalue_upper":
                eigenvalues[
                    mode
                ]
                + north_eigen_error,

            "n_grid_cells":
                nspace,
        })


    print(
        f"\n{family} EOF variance:"
    )

    for mode in range(
        N_EOF_MODES
    ):

        print(
            f"EOF{mode+1}: "
            f"{explained[mode]:.4f}%"
        )


    return {

        "patterns":
            patterns_grid,

        "pcs":
            PCs_std,

        "pcs_raw":
            PCs_raw,

        "explained":
            explained,

        "eigenvalues":
            eigenvalues,

        "summary":
            summary_rows,
    }


# 25. NORTH MODE-SEPARATION CHECK

def add_north_separation_diagnostics(
    summary_df
):

    """
    Two neighboring modes are treated as clearly separated when
    their approximate North eigenvalue uncertainty intervals do
    not overlap.

    This is a conservative interpretive diagnostic.
    """

    output = summary_df.copy()

    output[
        "north_separated_from_next_mode"
    ] = np.nan

    output[
        "north_interval_overlap_with_next"
    ] = np.nan


    for family in output[
        "family"
    ].unique():


        family_rows = output[
            output[
                "family"
            ] == family
        ].sort_values(
            "mode"
        )


        for i in range(
            len(
                family_rows
            )
            - 1
        ):


            current_index = family_rows.index[
                i
            ]

            next_index = family_rows.index[
                i + 1
            ]


            lower_current = output.loc[
                current_index,
                "north_eigenvalue_lower"
            ]

            upper_current = output.loc[
                current_index,
                "north_eigenvalue_upper"
            ]


            lower_next = output.loc[
                next_index,
                "north_eigenvalue_lower"
            ]

            upper_next = output.loc[
                next_index,
                "north_eigenvalue_upper"
            ]


            overlap = (
                max(
                    lower_current,
                    lower_next
                )
                <=
                min(
                    upper_current,
                    upper_next
                )
            )


            output.loc[
                current_index,
                "north_interval_overlap_with_next"
            ] = bool(
                overlap
            )


            output.loc[
                current_index,
                "north_separated_from_next_mode"
            ] = bool(
                not overlap
            )


    return output


# 26. SECTOR FUNCTIONS — ONLY FOR TABLE S7

def sector_mask(
    lon2d,
    lat2d,
    lon_min,
    lon_max,
):

    lat_ok = (
        (lat2d >= LAT_MIN)
        & (lat2d <= LAT_MAX)
    )

    if lon_max < lon_min:

        lon_ok = (
            (lon2d >= lon_min)
            | (lon2d < lon_max)
        )

    else:

        lon_ok = (
            (lon2d >= lon_min)
            & (lon2d < lon_max)
        )

    return (
        lat_ok
        & lon_ok
    )


def weighted_mean(
    field,
    mask,
    area,
):

    valid = (
        mask
        & np.isfinite(
            field
        )
        & np.isfinite(
            area
        )
        & (
            area > 0
        )
    )

    if valid.sum() == 0:
        return np.nan

    return np.sum(
        field[
            valid
        ]
        * area[
            valid
        ]
    ) / np.sum(
        area[
            valid
        ]
    )


# 31. READ SLA

print(
    "\n"
    + "=" * 90
)

print(
    "READING DUACS SLA"
)

print(
    "=" * 90
)


sla_ds = standardize_dataset(
    open_dataset_safely(
        SLA_FILE
    )
)


sla_var = detect_time_lat_lon_variable(
    sla_ds,
    SLA_VARIABLE_NAMES,
    "SLA"
)


sla = squeeze_to_3d(
    sla_ds[
        sla_var
    ]
)


sla = sla.sel(
    lat=slice(
        LAT_MIN,
        LAT_MAX
    )
)


sla = make_exact_monthly(
    sla,
    "SLA"
)


sla = convert_to_cm(
    sla,
    "SLA",
    SLA_SCALE_TO_CM
)


print(
    "SLA grid:",
    sla.sizes[
        "lat"
    ],
    "×",
    sla.sizes[
        "lon"
    ]
)


# 32. READ TOTAL STERIC HEIGHT

print(
    "\n"
    + "=" * 90
)

print(
    "READING EN4 TOTAL STERIC HEIGHT"
)

print(
    "=" * 90
)


steric_ds = standardize_dataset(
    open_dataset_safely(
        STERIC_FILE
    )
)


steric_var = detect_time_lat_lon_variable(
    steric_ds,
    STERIC_VARIABLE_NAMES,
    "Total steric"
)


steric = squeeze_to_3d(
    steric_ds[
        steric_var
    ]
)


steric = steric.sel(
    lat=slice(
        LAT_MIN,
        LAT_MAX
    )
)


steric = make_exact_monthly(
    steric,
    "Total steric"
)


steric = convert_to_cm(
    steric,
    "Total steric",
    STERIC_SCALE_TO_CM
)


print(
    "Total-steric grid:",
    steric.sizes[
        "lat"
    ],
    "×",
    steric.sizes[
        "lon"
    ]
)


# 33. REGRID SLA TO EN4 GRID

print(
    "\n"
    + "=" * 90
)

print(
    "REGRIDDING SLA TO EN4 GRID"
)

print(
    "=" * 90
)


print(
    "Original SLA grid:",
    sla.sizes[
        "lat"
    ],
    "×",
    sla.sizes[
        "lon"
    ]
)


print(
    "EN4 target grid:",
    steric.sizes[
        "lat"
    ],
    "×",
    steric.sizes[
        "lon"
    ]
)


sla_on_en4 = regrid_to_target(
    source=sla,

    target_lat=
        steric[
            "lat"
        ],

    target_lon=
        steric[
            "lon"
        ],

    method=
        SLA_REGRID_METHOD,
)


# Make times exactly identical.
sla_on_en4 = sla_on_en4.assign_coords(
    time=
        steric[
            "time"
        ]
)


print(
    "Final common grid:",
    sla_on_en4.sizes[
        "lat"
    ],
    "×",
    sla_on_en4.sizes[
        "lon"
    ]
)


# 34. PREPARE GEBCO

print(
    "\n"
    + "=" * 90
)

print(
    "PREPARING GEBCO"
)

print(
    "=" * 90
)


gebco = prepare_gebco(
    GEBCO_FILE
)


# 35. BATHYMETRY ON EN4 GRID

bathy_target = gebco[
    "full"
].interp(
    lat=
        steric[
            "lat"
        ],

    lon=
        steric[
            "lon"
        ],

    method=
        "nearest",
)


bathy_values = np.asarray(
    bathy_target.values,
    dtype=float
)


depth_mask = (
    np.isfinite(
        bathy_values
    )
    & (
        bathy_values
        <= -MIN_DEPTH_M
    )
)


# 36. BUILD EOF-SPECIFIC REFINED MASK C

print(
    "\n"
    + "=" * 90
)

print(
    "BUILDING FIXED COMMON EOF MASK"
)

print(
    "=" * 90
)


sla_values = np.asarray(
    sla_on_en4.values,
    dtype=float
)


steric_values = np.asarray(
    steric.values,
    dtype=float
)


# A. DUACS >=80% valid months

sla_valid_count = np.sum(
    np.isfinite(
        sla_values
    ),
    axis=0
)


sla_valid_fraction = (
    sla_valid_count
    / float(
        N_MONTHS
    )
)


sla_80pct = (
    sla_valid_count
    >= MIN_SLA_VALID_MONTHS
)


# B. >=3 valid observations in EACH calendar month

calendar_support = calendar_month_support(
    sla_values,
    steric[
        "time"
    ].values
)


calendar_support_ok = np.all(
    calendar_support
    >= MIN_VALID_PER_CALENDAR_MONTH,
    axis=0
)


# C. EN4 must be complete

steric_complete = np.all(
    np.isfinite(
        steric_values
    ),
    axis=0
)


# D. Non-zero temporal variance

sla_std = np.nanstd(
    sla_values,
    axis=0
)


steric_std = np.nanstd(
    steric_values,
    axis=0
)


variance_ok = (
    np.isfinite(
        sla_std
    )
    & np.isfinite(
        steric_std
    )
    & (
        sla_std > 1.0e-12
    )
    & (
        steric_std > 1.0e-12
    )
)


# FINAL EOF MASK

common_mask = (
    depth_mask
    & sla_80pct
    & calendar_support_ok
    & steric_complete
    & variance_ok
)


N_DEPTH = int(
    depth_mask.sum()
)


N_SLA80 = int(
    (
        depth_mask
        & sla_80pct
    ).sum()
)


N_CALENDAR = int(
    (
        depth_mask
        & sla_80pct
        & calendar_support_ok
    ).sum()
)


N_COMMON = int(
    common_mask.sum()
)


COMMON_COVERAGE = (
    100.0
    * N_COMMON
    / max(
        N_DEPTH,
        1
    )
)


print(
    f"DUACS minimum valid months   : "
    f"{MIN_SLA_VALID_MONTHS}/{N_MONTHS}"
)

print(
    f"DUACS availability required  : "
    f">={MIN_SLA_VALID_FRACTION*100:.0f}%"
)

print(
    f"Min support/calendar month   : "
    f"{MIN_VALID_PER_CALENDAR_MONTH}"
)

print(
    f">=1000-m depth cells         : "
    f"{N_DEPTH:,}"
)

print(
    f"After >=80% SLA criterion    : "
    f"{N_SLA80:,}"
)

print(
    f"After calendar-month QC      : "
    f"{N_CALENDAR:,}"
)

print(
    f"Final common EOF cells       : "
    f"{N_COMMON:,}"
)

print(
    f"Retained depth-qualified area: "
    f"{COMMON_COVERAGE:.2f}%"
)


if N_COMMON < 100:

    raise RuntimeError(
        "Too few common EOF cells retained."
    )


# 37. SAVE MASK DIAGNOSTICS

mask_diagnostics = pd.DataFrame([{

    "n_months":
        N_MONTHS,

    "minimum_sla_valid_fraction":
        MIN_SLA_VALID_FRACTION,

    "minimum_sla_valid_months":
        MIN_SLA_VALID_MONTHS,

    "minimum_valid_per_calendar_month":
        MIN_VALID_PER_CALENDAR_MONTH,

    "minimum_depth_m":
        MIN_DEPTH_M,

    "depth_qualified_cells":
        N_DEPTH,

    "cells_after_sla_80pct":
        N_SLA80,

    "cells_after_calendar_month_qc":
        N_CALENDAR,

    "final_common_eof_cells":
        N_COMMON,

    "retained_depth_qualified_percent":
        COMMON_COVERAGE,

}])


mask_diagnostics.to_csv(
    OUT_MASK_CSV,
    index=False
)


# 38. GRID-CELL AREA

cell_area = calculate_cell_area(
    steric[
        "lat"
    ].values,
    steric[
        "lon"
    ].values
)


# 39. EXTRACT COMMON-MASK MATRICES

flat_mask = common_mask.ravel()


sla_matrix_raw = (
    sla_values.reshape(
        N_MONTHS,
        -1
    )[
        :,
        flat_mask
    ]
)


steric_matrix_raw = (
    steric_values.reshape(
        N_MONTHS,
        -1
    )[
        :,
        flat_mask
    ]
)


area_vector = (
    cell_area.ravel()[
        flat_mask
    ]
)


# 40. SLA PREPROCESSING

print(
    "\n"
    + "=" * 90
)

print(
    "SLA ANOMALY PREPROCESSING"
)

print(
    "=" * 90
)


sla_anomaly_missing = preprocess_with_missing(
    sla_matrix_raw,
    steric[
        "time"
    ].values
)


sla_missing_before = int(
    np.sum(
        ~np.isfinite(
            sla_anomaly_missing
        )
    )
)


sla_total_entries = int(
    sla_anomaly_missing.size
)


sla_missing_fraction = (
    100.0
    * sla_missing_before
    / sla_total_entries
)


print(
    f"Missing SLA anomaly entries inside retained EOF mask: "
    f"{sla_missing_before:,}"
)

print(
    f"Missing fraction: "
    f"{sla_missing_fraction:.4f}%"
)


# 41. ITERATIVE SLA GAP RECONSTRUCTION

sla_reconstructed, reconstruction_diagnostics = (
    iterative_eof_reconstruction(

        anomaly_matrix=
            sla_anomaly_missing,

        area_vector=
            area_vector,

        n_modes=
            RECONSTRUCTION_MODES,

        max_iter=
            RECON_MAX_ITER,

        tolerance=
            RECON_TOL,
    )
)


# Final numerical cleanup to ensure exact monthly-climatology removal
# and detrending after reconstruction.

sla_anomaly_final = preprocess_complete(
    sla_reconstructed,
    steric[
        "time"
    ].values
)


# 42. TOTAL-STERIC PREPROCESSING

print(
    "\n"
    + "=" * 90
)

print(
    "TOTAL-STERIC ANOMALY PREPROCESSING"
)

print(
    "=" * 90
)


if not np.all(
    np.isfinite(
        steric_matrix_raw
    )
):

    raise RuntimeError(
        "Total-steric matrix contains missing values "
        "despite steric_complete mask."
    )


steric_anomaly_final = preprocess_complete(
    steric_matrix_raw,
    steric[
        "time"
    ].values
)


# 43. CALCULATE SLA EOF

sla_eof = calculate_gridded_eof(

    X=
        sla_anomaly_final,

    cell_area_vector=
        area_vector,

    spatial_mask=
        common_mask,

    nlat=
        steric.sizes[
            "lat"
        ],

    nlon=
        steric.sizes[
            "lon"
        ],

    family=
        "SLA",
)


# 44. CALCULATE TOTAL-STERIC EOF

steric_eof = calculate_gridded_eof(

    X=
        steric_anomaly_final,

    cell_area_vector=
        area_vector,

    spatial_mask=
        common_mask,

    nlat=
        steric.sizes[
            "lat"
        ],

    nlon=
        steric.sizes[
            "lon"
        ],

    family=
        "Total_steric",
)


# 45. NORTH MODE-SEPARATION DIAGNOSTICS

mode_df = pd.DataFrame(
    sla_eof[
        "summary"
    ]
    +
    steric_eof[
        "summary"
    ]
)


mode_df = add_north_separation_diagnostics(
    mode_df
)


mode_df.to_csv(
    OUT_MODE_CSV,
    index=False
)


print(
    "\n"
    + "=" * 90
)

print(
    "NORTH MODE-SEPARATION CHECK"
)

print(
    "=" * 90
)


for family in [
    "SLA",
    "Total_steric"
]:

    sub = mode_df[
        mode_df[
            "family"
        ] == family
    ].sort_values(
        "mode"
    )


    print(
        f"\n{family}:"
    )


    for _, row in sub.iterrows():

        mode = int(
            row[
                "mode"
            ]
        )

        ev = row[
            "explained_variance_percent"
        ]

        err = row[
            "north_explained_variance_error_percent"
        ]


        print(
            f"  EOF{mode}: "
            f"{ev:.3f}% "
            f"± {err:.3f}% "
            f"(North approx.)"
        )


        if (
            mode < N_EOF_MODES
            and pd.notna(
                row[
                    "north_separated_from_next_mode"
                ]
            )
        ):

            separated = bool(
                row[
                    "north_separated_from_next_mode"
                ]
            )


            if separated:

                print(
                    f"       EOF{mode} and EOF{mode+1}: "
                    f"clearly separated by this diagnostic."
                )

            else:

                print(
                    f"       WARNING: EOF{mode} and EOF{mode+1} "
                    f"North uncertainty intervals overlap."
                )

                print(
                    "       Interpret these neighboring modes "
                    "jointly rather than as uniquely ordered modes."
                )


# 46. PC DATAFRAMES


dates = pd.DatetimeIndex(
    steric[
        "time"
    ].values
)


sla_pc = pd.DataFrame(

    sla_eof[
        "pcs"
    ],

    index=
        dates,

    columns=[
        "PC1",
        "PC2",
        "PC3",
    ],
)


steric_pc = pd.DataFrame(

    steric_eof[
        "pcs"
    ],

    index=
        dates,

    columns=[
        "PC1",
        "PC2",
        "PC3",
    ],
)


sla_pc_rm = sla_pc.rolling(
    window=13,
    center=True,
    min_periods=7,
).mean()


steric_pc_rm = steric_pc.rolling(
    window=13,
    center=True,
    min_periods=7,
).mean()


# 47. SAVE PC TIME SERIES

pc_rows = []


for family, raw, running, result in [

    (
        "SLA",
        sla_pc,
        sla_pc_rm,
        sla_eof,
    ),

    (
        "Total_steric",
        steric_pc,
        steric_pc_rm,
        steric_eof,
    ),
]:

    for mode in [
        1,
        2,
        3
    ]:

        for date in dates:

            pc_rows.append({

                "family":
                    family,

                "time":
                    date,

                "mode":
                    mode,

                "explained_variance_percent":
                    result[
                        "explained"
                    ][
                        mode - 1
                    ],

                "pc_standardized_monthly":
                    raw.loc[
                        date,
                        f"PC{mode}"
                    ],

                "pc_13month_running_mean":
                    running.loc[
                        date,
                        f"PC{mode}"
                    ],
            })


pd.DataFrame(
    pc_rows
).to_csv(
    OUT_PC_CSV,
    index=False
)


# 48. SAVE RECONSTRUCTION DIAGNOSTICS

reconstruction_df = pd.DataFrame([{

    "minimum_DUACS_valid_fraction":
        MIN_SLA_VALID_FRACTION,

    "minimum_DUACS_valid_months":
        MIN_SLA_VALID_MONTHS,

    "minimum_per_calendar_month":
        MIN_VALID_PER_CALENDAR_MONTH,

    "reconstruction_modes":
        reconstruction_diagnostics[
            "reconstruction_modes"
        ],

    "iterations":
        reconstruction_diagnostics[
            "iterations"
        ],

    "converged":
        reconstruction_diagnostics[
            "converged"
        ],

    "final_relative_change":
        reconstruction_diagnostics[
            "relative_change"
        ],

    "missing_entries_reconstructed":
        reconstruction_diagnostics[
            "missing_values"
        ],

    "missing_fraction_percent":
        sla_missing_fraction,

    "common_grid_cells":
        N_COMMON,

}])


reconstruction_df.to_csv(
    OUT_RECON_CSV,
    index=False
)


# 49. SAVE EOF NETCDF

reconstruction_cells = np.any(
    ~np.isfinite(
        sla_matrix_raw
    ),
    axis=0
)


reconstruction_grid = np.zeros(
    common_mask.size,
    dtype=np.int8
)


reconstruction_grid[
    flat_mask
] = reconstruction_cells.astype(
    np.int8
)


reconstruction_grid = reconstruction_grid.reshape(
    common_mask.shape
)


output_ds = xr.Dataset(

    data_vars={

        "sla_eof_loading": (
            (
                "mode",
                "lat",
                "lon"
            ),
            sla_eof[
                "patterns"
            ],
        ),

        "total_steric_eof_loading": (
            (
                "mode",
                "lat",
                "lon"
            ),
            steric_eof[
                "patterns"
            ],
        ),

        "common_eof_mask": (
            (
                "lat",
                "lon"
            ),
            common_mask.astype(
                np.int8
            ),
        ),

        "sla_valid_fraction": (
            (
                "lat",
                "lon"
            ),
            sla_valid_fraction,
        ),

        "sla_reconstruction_used": (
            (
                "lat",
                "lon"
            ),
            reconstruction_grid,
        ),

        "grid_cell_area": (
            (
                "lat",
                "lon"
            ),
            cell_area,
        ),

        "bathymetry": (
            (
                "lat",
                "lon"
            ),
            bathy_values,
        ),
    },

    coords={

        "mode":
            [
                1,
                2,
                3
            ],

        "lat":
            steric[
                "lat"
            ].values,

        "lon":
            steric[
                "lon"
            ].values,
    },

    attrs={

        "title":
            "Area-weighted gridded covariance EOF analysis "
            "of Antarctic DUACS SLA and EN4 0–1000 m total steric height",

        "period":
            "January 2008–December 2025",

        "number_of_months":
            N_MONTHS,

        "mask":
            "GEBCO depth >=1000 m; EN4 complete; "
            "DUACS SLA valid >=80% of months; "
            "at least 3 valid observations for every calendar month.",

        "missing_SLA_treatment":
            "Limited missing SLA anomalies inside retained cells were "
            "estimated using iterative area-weighted low-rank EOF "
            "reconstruction while observed anomaly values were held fixed.",

        "preprocessing":
            "Calendar-month climatology and grid-cell linear trend removed.",

        "EOF_method":
            "Grid-cell-area-weighted covariance EOF.",

        "loading_definition":
            "Covariance/regression loading on standardized principal component.",

        "loading_units":
            "cm per 1-standard-deviation PC",
    },
)


output_ds[
    "sla_eof_loading"
].attrs[
    "units"
] = "cm per PC standard deviation"


output_ds[
    "total_steric_eof_loading"
].attrs[
    "units"
] = "cm per PC standard deviation"


output_ds[
    "grid_cell_area"
].attrs[
    "units"
] = "m2"


output_ds[
    "bathymetry"
].attrs[
    "units"
] = "m"


output_ds.to_netcdf(
    OUT_NC
)


# 50. CREATE REGIONAL EOF SUMMARY FOR TABLE S7

lon2d, lat2d = np.meshgrid(
    steric[
        "lon"
    ].values,
    steric[
        "lat"
    ].values
)


sector_rows = []


for family, result in [

    (
        "SLA",
        sla_eof
    ),

    (
        "Total_steric",
        steric_eof
    ),
]:

    for mode_index in range(
        N_EOF_MODES
    ):

        field = result[
            "patterns"
        ][
            mode_index
        ]


        for code, info in SEA_SECTORS.items():


            smask = sector_mask(
                lon2d,
                lat2d,

                info[
                    "lon_min"
                ],

                info[
                    "lon_max"
                ],
            )


            smask &= common_mask


            valid = (
                smask
                & np.isfinite(
                    field
                )
            )


            values = field[
                valid
            ]


            sector_rows.append({

                "family":
                    family,

                "mode":
                    mode_index + 1,

                "explained_variance_percent":
                    result[
                        "explained"
                    ][
                        mode_index
                    ],

                "sea_code":
                    code,

                "sea_name":
                    info[
                        "name"
                    ],

                "n_common_grid_cells":
                    int(
                        smask.sum()
                    ),

                "area_weighted_mean_loading_cm_per_pc_sd":
                    weighted_mean(
                        field,
                        smask,
                        cell_area
                    ),

                "minimum_loading_cm_per_pc_sd":
                    (
                        float(
                            np.nanmin(
                                values
                            )
                        )
                        if values.size
                        else np.nan
                    ),

                "maximum_loading_cm_per_pc_sd":
                    (
                        float(
                            np.nanmax(
                                values
                            )
                        )
                        if values.size
                        else np.nan
                    ),

                "positive_loading_fraction":
                    (
                        float(
                            np.mean(
                                values > 0
                            )
                        )
                        if values.size
                        else np.nan
                    ),

                "negative_loading_fraction":
                    (
                        float(
                            np.mean(
                                values < 0
                            )
                        )
                        if values.size
                        else np.nan
                    ),
            })


sector_df = pd.DataFrame(
    sector_rows
)


sector_df.to_csv(
    OUT_SECTOR_CSV,
    index=False
)




# DATA PRODUCT GROUP 10: FIGURE 13 LOCAL-FORCING / AAO PROCESSED PRODUCTS — WSC MASK FIX
# Source: so_sealevel_paper_fig.py
# File name: Fig13_sector_driver_timeseries_LOW_RAM.csv
# File name: Fig13a_SLA_steric_correlations.csv
# File name: Fig13b_thermosteric_driver_correlations.csv
# File name: Fig13c_halosteric_driver_correlations.csv
# File name: Fig13d_thermosteric_regression_coefficients.csv
# File name: Fig13e_halosteric_regression_coefficients.csv
# File name: Fig13g_AAO_SLA_lag_correlations.csv
# File name: Fig13h_AAO_SLA_composite_LOW_RAM.nc

# FIGURE 13 — LOCAL FORCING AND AAO ATTRIBUTION
# LOW-RAM, CHECKPOINTED, 1080-DPI WORKFLOW — WSC MASK FIX
#
# This version is designed for Google Colab sessions that crashed when:
#   • many large NetCDF files remained open simultaneously;
#   • Qnet* and wind-stress curl were created as complete 3-D arrays;
#   • hundreds of bootstrap composite maps were stored in memory;
#   • a very large physical figure was rendered at 1080 dpi.
#
# LOW-RAM STRATEGY
# ----------------
# 1. Process ONE NetCDF variable at a time.
# 2. Load ONE monthly 2-D field at a time.
# 3. Immediately reduce each field to 13 area-weighted sector means.
# 4. Close every NetCDF before opening the next one.
# 5. Derive Qnet* month-by-month from SSRD, STRD, SLHF and SSHF.
# 6. Derive wind-stress curl month-by-month from U10 and V10.
# 7. Save each small sector series as a checkpoint CSV.
# 8. Calculate the AAO composite on an automatically coarsened SLA grid.
# 9. Use online bootstrap sign counts; do not store all bootstrap maps.
# 10. Save a compact fixed-size figure at 1080 dpi without bbox_inches="tight".
#
# SCIENTIFIC NOTE ON HEAT FLUX
# ----------------------------
# The available radiative fields are:
#   SSRD = downward shortwave radiation
#   STRD = downward longwave radiation
#
# They are not net SSR and net STR. Therefore this workflow uses:
#
#   Qnet* = SSRD + STRD + signed(SLHF) + signed(SSHF)
#
# Qnet* is labelled as a SURFACE HEAT-FLUX PROXY throughout the outputs.
# A complete physical net surface heat flux requires ERA5 SSR and STR.
#
# OUTPUT
# ------
# /content/drive/MyDrive/SAM_Thesis/paper2/



# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "cartopy": "cartopy",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
    print("Package installation complete.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import gc
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.path import Path as MplPath

from scipy.stats import pearsonr, t as student_t, zscore

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

import cartopy.crs as ccrs

warnings.filterwarnings("ignore", category=RuntimeWarning)



# 3. PATHS
BASE_DIR = Path("/content/drive/MyDrive/SAM_Thesis")
DATA_DIR = BASE_DIR / "Data"
INVENTORY_DIR = BASE_DIR / "Processed" / "EN4_NetCDF_inventory"
OUTPUT_DIR = BASE_DIR / "paper2"
CACHE_DIR = OUTPUT_DIR / "Fig13_lowRAM_cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATHS = {
    # Response variables
    "sla": DATA_DIR / "SLA_Antarctic_monthly_2008_2025.nc",
    "total_steric": INVENTORY_DIR / "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc",
    "thermosteric": INVENTORY_DIR / "EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc",
    "halosteric": INVENTORY_DIR / "EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc",

    # Thermosteric drivers
    "ohc": INVENTORY_DIR / "EN4_OHC_0_1000m_monthly_2008_2025_SO.nc",
    "sst": DATA_DIR / "OSTIA_SST_monthly_2008_2025_SO.nc",
    "radiative_flux": DATA_DIR / "SW_LW_era.nc",
    "turbulent_flux": DATA_DIR / "heatflux_latent_sensible_era.nc",
    "wind_uv": DATA_DIR / "Wind_U_V_era.nc",
    "mld": DATA_DIR / "MLD_monthly_2008_2025_SO.nc",
    "sic": DATA_DIR / "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc",

    # Halosteric drivers
    "fwc": INVENTORY_DIR / "EN4_freshwater_content_0_200m_monthly_2008_2025_SO.nc",
    "sss_anomaly": INVENTORY_DIR / "EN4_surface_salinity_anomaly_0_30m_monthly_2008_2025_SO.nc",
    "snowfall": DATA_DIR / "P_Snow_era.nc",
    "stratification": INVENTORY_DIR / "EN4_stratification_20_200m_monthly_2008_2025_SO.nc",

    # Background and climate index
    "gebco": DATA_DIR / "GEBCO_2024_CF.nc",
    "aao": DATA_DIR / "NOAA_CPC_SAM_AAO_monthly_2008_2025.nc",
}

OUT_SECTOR_TS = OUTPUT_DIR / "Fig13_sector_driver_timeseries_LOW_RAM.csv"

OUT_TABLE_A = OUTPUT_DIR / "Fig13a_SLA_steric_correlations.csv"
OUT_TABLE_B = OUTPUT_DIR / "Fig13b_thermosteric_driver_correlations.csv"
OUT_TABLE_C = OUTPUT_DIR / "Fig13c_halosteric_driver_correlations.csv"
OUT_TABLE_D = OUTPUT_DIR / "Fig13d_thermosteric_regression_coefficients.csv"
OUT_TABLE_E = OUTPUT_DIR / "Fig13e_halosteric_regression_coefficients.csv"
OUT_TABLE_G = OUTPUT_DIR / "Fig13g_AAO_SLA_lag_correlations.csv"

OUT_COMPOSITE = OUTPUT_DIR / "Fig13h_AAO_SLA_composite_LOW_RAM.nc"
OUT_LOG = OUTPUT_DIR / "Fig13_LOW_RAM_processing_log.txt"


# 4. USER-CONTROLLED LOW-RAM SETTINGS
REUSE_CACHED_SECTOR_SERIES = True
REUSE_CACHED_COMPOSITE = True

# ERA5 monthly means of accumulated fields in J m-2 are divided by
# 86400 seconds to obtain W m-2.
ERA5_ACCUMULATION_DIVISOR_SECONDS = 86400.0

# Native ERA5 accumulated SLHF and SSHF are normally signed with
# downward positive; upward ocean heat loss is negative.
#
# "ecmwf_native" = add SLHF and SSHF directly.
# "reverse"      = reverse both turbulent-flux signs before addition.
TURBULENT_FLUX_SIGN_MODE = "ecmwf_native"

# Bulk wind-stress calculation constants
AIR_DENSITY_KG_M3 = 1.225
EARTH_RADIUS_M = 6_371_000.0

# The SLA composite and bootstrap are calculated on a coarsened grid.
# Factor 2 substantially reduces RAM and bootstrap runtime while retaining
# the broad circumpolar composite pattern.
COMPOSITE_COARSEN_FACTOR = 2
COMPOSITE_BOOTSTRAP_REPLICATES = 250
COMPOSITE_BOOTSTRAP_SEED = 42

# High-resolution export
SAVE_DPI = 1080

# Compact physical size limits PNG rendering memory:
# about 9.0 × 5.9 inches at 1080 dpi.
FIGSIZE = (9.0, 5.9)

LAT_MIN = -90.0
LAT_MAX = -60.0
LON_MIN = -180.0
LON_MAX = 180.0

AAO_POSITIVE_THRESHOLD = 1.0
AAO_NEGATIVE_THRESHOLD = -1.0
LAGS = np.arange(-6, 7, dtype=int)

HEAT_FLUX_KEY = "Qnet*"


# 6. 13 ANTARCTIC SHELF SECTORS
SEA_SECTORS = OrderedDict([
    ("WED", {"name": "Weddell Sea",         "lon_min":  -60.0, "lon_max":  -20.0}),
    ("KHV", {"name": "King Haakon VII Sea", "lon_min":  -20.0, "lon_max":   10.0}),
    ("RLS", {"name": "Riiser-Larsen Sea",   "lon_min":   10.0, "lon_max":   35.0}),
    ("LAZ", {"name": "Lazarev Sea",         "lon_min":   35.0, "lon_max":   60.0}),
    ("COS", {"name": "Cosmonauts Sea",      "lon_min":   60.0, "lon_max":   90.0}),
    ("COO", {"name": "Cooperation Sea",     "lon_min":   90.0, "lon_max":  115.0}),
    ("DAV", {"name": "Davis Sea",           "lon_min":  115.0, "lon_max":  130.0}),
    ("MAW", {"name": "Mawson Sea",          "lon_min":  130.0, "lon_max":  150.0}),
    ("DUR", {"name": "D'Urville Sea",       "lon_min":  150.0, "lon_max":  170.0}),
    ("SOM", {"name": "Somov Sea",           "lon_min":  170.0, "lon_max": -160.0}),
    ("ROS", {"name": "Ross Sea",            "lon_min": -160.0, "lon_max": -130.0}),
    ("AMU", {"name": "Amundsen Sea",        "lon_min": -130.0, "lon_max": -100.0}),
    ("BEL", {"name": "Bellingshausen Sea",  "lon_min": -100.0, "lon_max":  -60.0}),
])

SEA_ORDER = list(SEA_SECTORS.keys())

THERMO_DRIVER_ORDER = [
    "OHC",
    "SST",
    HEAT_FLUX_KEY,
    "WSC",
    "MLD",
    "SIC",
    "AAO",
]

HALO_DRIVER_ORDER = [
    "FWC",
    "SSS anomaly",
    "SIC",
    "Snowfall",
    "MLD",
    "Stratification",
    "AAO",
]

THERMO_REG_ORDER = [
    HEAT_FLUX_KEY,
    "WSC",
    "MLD",
    "SIC",
    "AAO",
]

HALO_REG_ORDER = [
    "FWC",
    "SIC",
    "Snowfall",
    "MLD",
    "AAO",
]


# 7. GENERIC FILE / COORDINATE HELPERS
def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found:\n{path}")
    return path


def open_dataset_safely(path):
    attempts = []

    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = {
                "decode_times": True,
                "mask_and_scale": True,
                "cache": False,
            }

            if engine is not None:
                kwargs["engine"] = engine

            dataset = xr.open_dataset(path, **kwargs)

            print(
                f"Opened {Path(path).name} with "
                f"engine={engine or 'xarray-default'}"
            )

            return dataset

        except Exception as error:
            attempts.append(f"{engine}: {error}")

    raise RuntimeError(
        f"Could not open:\n{path}\n\n"
        + "\n".join(attempts)
    )


def detect_coordinate(dataset, kind):
    aliases = {
        "lat": ["lat", "latitude", "nav_lat", "y"],
        "lon": ["lon", "longitude", "nav_lon", "x"],
        "time": ["time", "valid_time", "date", "month", "datetime"],
    }

    for name in list(dataset.coords) + list(dataset.variables):
        variable = dataset[name]

        lower_name = name.lower()
        standard_name = str(
            variable.attrs.get("standard_name", "")
        ).lower()
        axis = str(variable.attrs.get("axis", "")).upper()
        units = str(variable.attrs.get("units", "")).lower()

        if kind == "lat":
            if (
                lower_name in aliases["lat"]
                or standard_name == "latitude"
                or axis == "Y"
                or "degrees_north" in units
            ):
                return name

        elif kind == "lon":
            if (
                lower_name in aliases["lon"]
                or standard_name == "longitude"
                or axis == "X"
                or "degrees_east" in units
            ):
                return name

        elif kind == "time":
            if (
                lower_name in aliases["time"]
                or axis == "T"
                or np.issubdtype(variable.dtype, np.datetime64)
            ):
                return name

    raise KeyError(f"Could not detect {kind} coordinate.")


def standardize_dataset(dataset):
    latitude_name = detect_coordinate(dataset, "lat")
    longitude_name = detect_coordinate(dataset, "lon")

    rename_mapping = {}

    if latitude_name != "lat":
        rename_mapping[latitude_name] = "lat"

    if longitude_name != "lon":
        rename_mapping[longitude_name] = "lon"

    try:
        time_name = detect_coordinate(dataset, "time")
        if time_name != "time":
            rename_mapping[time_name] = "time"
    except Exception:
        pass

    if rename_mapping:
        dataset = dataset.rename(rename_mapping)

    normalized_longitude = (
        (dataset["lon"].astype(float) + 180.0) % 360.0
    ) - 180.0

    dataset = dataset.assign_coords(lon=normalized_longitude)

    longitude_values = np.asarray(dataset["lon"].values)
    _, unique_indices = np.unique(longitude_values, return_index=True)

    dataset = dataset.isel(lon=np.sort(unique_indices))
    dataset = dataset.sortby("lon").sortby("lat")

    if "time" in dataset.coords:
        month_start = (
            pd.DatetimeIndex(pd.to_datetime(dataset["time"].values))
            .to_period("M")
            .to_timestamp()
        )

        dataset = dataset.assign_coords(time=month_start)

        if month_start.duplicated().any():
            dataset = dataset.groupby("time").mean(skipna=True)

        dataset = dataset.sortby("time")

    return dataset


def reduce_extra_dimensions(data_array):
    allowed = {"time", "lat", "lon"}

    for dimension in list(data_array.dims):
        if dimension in allowed:
            continue

        size = data_array.sizes[dimension]

        if size == 1:
            data_array = data_array.isel({dimension: 0}, drop=True)

        elif dimension.lower() in {"expver", "number", "ensemble"}:
            data_array = data_array.mean(dimension, skipna=True)

        else:
            print(
                f"WARNING: selecting index 0 from unexpected "
                f"dimension {dimension!r}, size={size}."
            )
            data_array = data_array.isel({dimension: 0}, drop=True)

    return data_array


def prepare_monthly_field(data_array):
    data_array = reduce_extra_dimensions(data_array)

    data_array = data_array.sel(
        time=slice("2008-01-01", "2025-12-31"),
        lat=slice(LAT_MIN, LAT_MAX),
    )

    return data_array.transpose("time", "lat", "lon")


def detect_variable(dataset, exact_names, include_tokens, label):
    lower_to_original = {
        variable_name.lower(): variable_name
        for variable_name in dataset.data_vars
    }

    for candidate in exact_names:
        if candidate.lower() in lower_to_original:
            selected = lower_to_original[candidate.lower()]
            print(f"Detected {label}: {selected}")
            return selected

    scored = []

    for variable_name in dataset.data_vars:
        data_array = dataset[variable_name]

        if not {"time", "lat", "lon"}.issubset(set(data_array.dims)):
            continue

        name_lower = variable_name.lower()
        long_name = str(data_array.attrs.get("long_name", "")).lower()
        standard_name = str(
            data_array.attrs.get("standard_name", "")
        ).lower()

        score = 0

        for token in include_tokens:
            token = token.lower()

            if token in name_lower:
                score += 12

            if token in long_name:
                score += 5

            if token in standard_name:
                score += 5

        if score > 0:
            scored.append((score, variable_name))

    if not scored:
        raise KeyError(
            f"Could not detect {label}.\n"
            f"Available variables: {list(dataset.data_vars)}"
        )

    scored.sort(reverse=True)
    selected = scored[0][1]

    print(f"Detected {label}: {selected}")
    return selected


def height_scale_to_cm(data_array):
    units = str(data_array.attrs.get("units", "")).lower().strip()

    if units in {
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    }:
        return 100.0

    return 1.0


# 8. GEBCO OCEAN MASK
def load_reduced_gebco():
    path = require_file(FILE_PATHS["gebco"], "GEBCO")

    dataset = standardize_dataset(
        open_dataset_safely(path)
    )

    variable_name = detect_variable(
        dataset,
        ["elevation", "z", "depth", "Band1"],
        ["elevation", "bathymetry", "depth"],
        "GEBCO elevation",
    )

    elevation = dataset[variable_name].sel(
        lat=slice(LAT_MIN, LAT_MAX)
    )

    latitude_step = max(
        1,
        int(np.ceil(elevation.sizes["lat"] / 420)),
    )

    longitude_step = max(
        1,
        int(np.ceil(elevation.sizes["lon"] / 840)),
    )

    elevation = elevation.isel(
        lat=slice(None, None, latitude_step),
        lon=slice(None, None, longitude_step),
    ).load()

    dataset.close()

    print(
        "Reduced GEBCO grid:",
        dict(elevation.sizes),
    )

    return elevation


def ocean_mask_on_grid(gebco, latitude, longitude):
    bathymetry = gebco.interp(
        lat=latitude,
        lon=longitude,
        method="nearest",
    )

    return np.asarray(
        bathymetry.values < 0.0,
        dtype=bool,
    )


# 9. SECTOR-MEAN HELPERS
def sector_mask(longitude_2d, latitude_2d, lon_min, lon_max):
    latitude_condition = (
        (latitude_2d >= LAT_MIN)
        & (latitude_2d <= LAT_MAX)
    )

    if lon_max < lon_min:
        longitude_condition = (
            (longitude_2d >= lon_min)
            | (longitude_2d < lon_max)
        )
    else:
        longitude_condition = (
            (longitude_2d >= lon_min)
            & (longitude_2d < lon_max)
        )

    return latitude_condition & longitude_condition


def prepare_sector_geometry(latitude, longitude, ocean_mask):
    longitude_2d, latitude_2d = np.meshgrid(
        np.asarray(longitude, dtype=float),
        np.asarray(latitude, dtype=float),
    )

    area_weights = np.cos(
        np.deg2rad(latitude_2d)
    ).astype(np.float32)

    masks = {}

    for sea_code, information in SEA_SECTORS.items():
        masks[sea_code] = (
            sector_mask(
                longitude_2d,
                latitude_2d,
                information["lon_min"],
                information["lon_max"],
            )
            & ocean_mask
        )

    return area_weights, masks


def area_weighted_sector_values(field, area_weights, masks):
    values = {}

    for sea_code in SEA_ORDER:
        valid = (
            masks[sea_code]
            & np.isfinite(field)
        )

        if valid.sum() == 0:
            values[sea_code] = np.nan
            continue

        values[sea_code] = float(
            np.average(
                field[valid],
                weights=area_weights[valid],
            )
        )

    return values


def cache_path(key):
    safe_name = (
        key.replace(" ", "_")
        .replace("*", "star")
        .replace("/", "_")
    )

    return CACHE_DIR / f"{safe_name}_sector_monthly.csv"


def load_cached_sector_table(key):
    path = cache_path(key)

    if not (
        REUSE_CACHED_SECTOR_SERIES
        and path.exists()
    ):
        return None

    table = pd.read_csv(path, parse_dates=["time"])
    table = table.set_index("time").sort_index()

    required_columns = set(SEA_ORDER)

    if not required_columns.issubset(table.columns):
        return None

    print(f"Loaded cached sector series: {path.name}")
    return table[SEA_ORDER]


def save_sector_cache(key, table):
    path = cache_path(key)

    table.to_csv(path, index_label="time")
    print(f"Saved checkpoint: {path.name}")


# 10. STANDARD FIELD → 13 SECTOR SERIES
def process_standard_field(
    key,
    path,
    exact_names,
    include_tokens,
    gebco,
    apply_height_cm_conversion=False,
):
    cached = load_cached_sector_table(key)

    if cached is not None:
        return cached

    dataset = standardize_dataset(
        open_dataset_safely(path)
    )

    variable_name = detect_variable(
        dataset,
        exact_names,
        include_tokens,
        key,
    )

    data_array = prepare_monthly_field(
        dataset[variable_name]
    )

    scale = (
        height_scale_to_cm(data_array)
        if apply_height_cm_conversion
        else 1.0
    )

    latitude = data_array["lat"].values
    longitude = data_array["lon"].values
    time = pd.DatetimeIndex(data_array["time"].values)

    ocean_mask = ocean_mask_on_grid(
        gebco,
        data_array["lat"],
        data_array["lon"],
    )

    area_weights, masks = prepare_sector_geometry(
        latitude,
        longitude,
        ocean_mask,
    )

    output_rows = []

    print(
        f"Processing {key} month-by-month: "
        f"{data_array.sizes['time']} months"
    )

    for time_index in range(data_array.sizes["time"]):
        field = np.asarray(
            data_array.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze()

        field = field * np.float32(scale)

        values = area_weighted_sector_values(
            field,
            area_weights,
            masks,
        )

        values["time"] = time[time_index]
        output_rows.append(values)

        del field

        if (
            (time_index + 1) % 24 == 0
            or time_index == data_array.sizes["time"] - 1
        ):
            print(
                f"  [{time_index + 1:03d}/"
                f"{data_array.sizes['time']:03d}] "
                f"{time[time_index]:%Y-%m}"
            )

    output = (
        pd.DataFrame(output_rows)
        .set_index("time")
        .sort_index()
    )[SEA_ORDER]

    dataset.close()
    del dataset, data_array, ocean_mask, area_weights, masks
    gc.collect()

    save_sector_cache(key, output)
    return output


# 11. Qnet* PROXY → 13 SECTOR SERIES, MONTH-BY-MONTH
def process_qnet_proxy(gebco):
    cached = load_cached_sector_table(HEAT_FLUX_KEY)

    if cached is not None:
        return cached

    radiative_dataset = standardize_dataset(
        open_dataset_safely(FILE_PATHS["radiative_flux"])
    )

    turbulent_dataset = standardize_dataset(
        open_dataset_safely(FILE_PATHS["turbulent_flux"])
    )

    ssrd_name = detect_variable(
        radiative_dataset,
        ["ssrd"],
        ["ssrd", "short-wave", "shortwave", "solar", "downward"],
        "SSRD",
    )

    strd_name = detect_variable(
        radiative_dataset,
        ["strd"],
        ["strd", "long-wave", "longwave", "thermal", "downward"],
        "STRD",
    )

    slhf_name = detect_variable(
        turbulent_dataset,
        ["slhf"],
        ["slhf", "latent"],
        "SLHF",
    )

    sshf_name = detect_variable(
        turbulent_dataset,
        ["sshf"],
        ["sshf", "sensible"],
        "SSHF",
    )

    ssrd = prepare_monthly_field(radiative_dataset[ssrd_name])
    strd = prepare_monthly_field(radiative_dataset[strd_name])
    slhf = prepare_monthly_field(turbulent_dataset[slhf_name])
    sshf = prepare_monthly_field(turbulent_dataset[sshf_name])

    common_time = (
        pd.Index(pd.DatetimeIndex(ssrd["time"].values))
        .intersection(pd.Index(pd.DatetimeIndex(strd["time"].values)))
        .intersection(pd.Index(pd.DatetimeIndex(slhf["time"].values)))
        .intersection(pd.Index(pd.DatetimeIndex(sshf["time"].values)))
        .sort_values()
    )

    if len(common_time) == 0:
        raise RuntimeError("No common months among Qnet* components.")

    ssrd = ssrd.sel(time=common_time)
    strd = strd.sel(time=common_time)
    slhf = slhf.sel(time=common_time)
    sshf = sshf.sel(time=common_time)

    reference_latitude = ssrd["lat"]
    reference_longitude = ssrd["lon"]

    # Interpolation objects remain lazy; only one monthly field is
    # evaluated inside the loop.
    if not (
        strd.sizes["lat"] == ssrd.sizes["lat"]
        and strd.sizes["lon"] == ssrd.sizes["lon"]
        and np.allclose(strd["lat"], reference_latitude)
        and np.allclose(strd["lon"], reference_longitude)
    ):
        strd = strd.interp(
            lat=reference_latitude,
            lon=reference_longitude,
            method="linear",
        )

    if not (
        slhf.sizes["lat"] == ssrd.sizes["lat"]
        and slhf.sizes["lon"] == ssrd.sizes["lon"]
        and np.allclose(slhf["lat"], reference_latitude)
        and np.allclose(slhf["lon"], reference_longitude)
    ):
        slhf = slhf.interp(
            lat=reference_latitude,
            lon=reference_longitude,
            method="linear",
        )

    if not (
        sshf.sizes["lat"] == ssrd.sizes["lat"]
        and sshf.sizes["lon"] == ssrd.sizes["lon"]
        and np.allclose(sshf["lat"], reference_latitude)
        and np.allclose(sshf["lon"], reference_longitude)
    ):
        sshf = sshf.interp(
            lat=reference_latitude,
            lon=reference_longitude,
            method="linear",
        )

    ocean_mask = ocean_mask_on_grid(
        gebco,
        reference_latitude,
        reference_longitude,
    )

    area_weights, masks = prepare_sector_geometry(
        reference_latitude.values,
        reference_longitude.values,
        ocean_mask,
    )

    turbulent_multiplier = (
        1.0
        if TURBULENT_FLUX_SIGN_MODE == "ecmwf_native"
        else -1.0
    )

    output_rows = []

    print(
        "Processing Qnet* month-by-month:\n"
        "  Qnet* = SSRD + STRD + signed(SLHF) + signed(SSHF)"
    )

    for time_index, timestamp in enumerate(common_time):
        sw = np.asarray(
            ssrd.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze() / np.float32(ERA5_ACCUMULATION_DIVISOR_SECONDS)

        lw = np.asarray(
            strd.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze() / np.float32(ERA5_ACCUMULATION_DIVISOR_SECONDS)

        latent = np.asarray(
            slhf.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze() / np.float32(ERA5_ACCUMULATION_DIVISOR_SECONDS)

        sensible = np.asarray(
            sshf.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze() / np.float32(ERA5_ACCUMULATION_DIVISOR_SECONDS)

        heat_flux_proxy = (
            sw
            + lw
            + np.float32(turbulent_multiplier) * latent
            + np.float32(turbulent_multiplier) * sensible
        )

        heat_flux_proxy[~ocean_mask] = np.nan

        values = area_weighted_sector_values(
            heat_flux_proxy,
            area_weights,
            masks,
        )

        values["time"] = pd.Timestamp(timestamp)
        output_rows.append(values)

        del sw, lw, latent, sensible, heat_flux_proxy

        if (
            (time_index + 1) % 24 == 0
            or time_index == len(common_time) - 1
        ):
            print(
                f"  [{time_index + 1:03d}/"
                f"{len(common_time):03d}] "
                f"{pd.Timestamp(timestamp):%Y-%m}"
            )

    output = (
        pd.DataFrame(output_rows)
        .set_index("time")
        .sort_index()
    )[SEA_ORDER]

    radiative_dataset.close()
    turbulent_dataset.close()

    del (
        radiative_dataset,
        turbulent_dataset,
        ssrd,
        strd,
        slhf,
        sshf,
        ocean_mask,
        area_weights,
        masks,
    )

    gc.collect()

    save_sector_cache(HEAT_FLUX_KEY, output)
    return output


# 12. WIND-STRESS CURL → 13 SECTOR SERIES, MONTH-BY-MONTH
def process_wind_stress_curl(gebco):
    key = "WSC"

    cached = load_cached_sector_table(key)

    if cached is not None:
        return cached

    dataset = standardize_dataset(
        open_dataset_safely(FILE_PATHS["wind_uv"])
    )

    u_name = detect_variable(
        dataset,
        ["u10", "10u", "u_wind", "u"],
        ["u10", "zonal", "u_component"],
        "10 m zonal wind",
    )

    v_name = detect_variable(
        dataset,
        ["v10", "10v", "v_wind", "v"],
        ["v10", "meridional", "v_component"],
        "10 m meridional wind",
    )

    u = prepare_monthly_field(dataset[u_name])
    v = prepare_monthly_field(dataset[v_name])

    common_time = (
        pd.Index(pd.DatetimeIndex(u["time"].values))
        .intersection(pd.Index(pd.DatetimeIndex(v["time"].values)))
        .sort_values()
    )

    u = u.sel(time=common_time)
    v = v.sel(time=common_time)

    if not (
        v.sizes["lat"] == u.sizes["lat"]
        and v.sizes["lon"] == u.sizes["lon"]
        and np.allclose(v["lat"], u["lat"])
        and np.allclose(v["lon"], u["lon"])
    ):
        v = v.interp(
            lat=u["lat"],
            lon=u["lon"],
            method="linear",
        )

    latitude = np.asarray(u["lat"].values, dtype=np.float64)
    longitude = np.asarray(u["lon"].values, dtype=np.float64)

    ocean_mask = ocean_mask_on_grid(
        gebco,
        u["lat"],
        u["lon"],
    )

    area_weights, masks = prepare_sector_geometry(
        latitude,
        longitude,
        ocean_mask,
    )

    latitude_radians = np.deg2rad(latitude)

    # Keep the latitude factor as a broadcastable (lat, 1) array.
    # At/near the South Pole, cos(latitude) approaches zero. Replacing
    # those rows with NaN before division avoids both singular values
    # and the previous boolean-index shape mismatch.
    cosine_latitude_1d = np.cos(latitude_radians)

    safe_cosine_latitude = np.where(
        np.abs(cosine_latitude_1d) < 0.02,
        np.nan,
        cosine_latitude_1d,
    )[:, None].astype(np.float64)

    longitude_spacing = float(
        np.nanmedian(np.diff(np.deg2rad(longitude)))
    )

    output_rows = []

    print("Processing wind-stress curl month-by-month.")

    for time_index, timestamp in enumerate(common_time):
        u_field = np.asarray(
            u.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze()

        v_field = np.asarray(
            v.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze()

        speed = np.hypot(u_field, v_field)

        drag = np.where(
            speed <= 11.0,
            1.2e-3,
            np.where(
                speed <= 20.0,
                (0.49 + 0.065 * speed) * 1.0e-3,
                1.8e-3,
            ),
        ).astype(np.float32)

        tau_x = (
            np.float32(AIR_DENSITY_KG_M3)
            * drag
            * speed
            * u_field
        )

        tau_y = (
            np.float32(AIR_DENSITY_KG_M3)
            * drag
            * speed
            * v_field
        )

        # Periodic central difference in longitude.
        d_tau_y_d_lambda = (
            np.roll(tau_y, -1, axis=1)
            - np.roll(tau_y, 1, axis=1)
        ) / np.float32(2.0 * longitude_spacing)

        # Non-periodic gradient in latitude.
        d_tau_x_d_phi = np.gradient(
            tau_x,
            latitude_radians,
            axis=0,
            edge_order=1,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            curl = (
                d_tau_y_d_lambda
                / (
                    np.float64(EARTH_RADIUS_M)
                    * safe_cosine_latitude
                )
                - d_tau_x_d_phi
                / np.float64(EARTH_RADIUS_M)
            )

        # Ensure a normal writable 2-D array before masking.
        curl = np.asarray(
            curl,
            dtype=np.float32,
        )

        # Rows close to 90°S are already NaN through
        # safe_cosine_latitude broadcasting. Apply only the full
        # two-dimensional ocean mask here.
        curl[~ocean_mask] = np.nan

        values = area_weighted_sector_values(
            curl,
            area_weights,
            masks,
        )

        values["time"] = pd.Timestamp(timestamp)
        output_rows.append(values)

        del (
            u_field,
            v_field,
            speed,
            drag,
            tau_x,
            tau_y,
            d_tau_y_d_lambda,
            d_tau_x_d_phi,
            curl,
        )

        if (
            (time_index + 1) % 24 == 0
            or time_index == len(common_time) - 1
        ):
            print(
                f"  [{time_index + 1:03d}/"
                f"{len(common_time):03d}] "
                f"{pd.Timestamp(timestamp):%Y-%m}"
            )

    output = (
        pd.DataFrame(output_rows)
        .set_index("time")
        .sort_index()
    )[SEA_ORDER]

    dataset.close()

    del (
        dataset,
        u,
        v,
        ocean_mask,
        area_weights,
        masks,
        cosine_latitude_1d,
        safe_cosine_latitude,
    )

    gc.collect()

    save_sector_cache(key, output)
    return output


# 13. AAO READER
def read_aao():
    path = require_file(FILE_PATHS["aao"], "AAO")

    if path.suffix.lower() in {".nc", ".nc4", ".cdf"}:
        dataset = open_dataset_safely(path)

        time_name = None

        for candidate in ["time", "valid_time", "date", "month"]:
            if candidate in dataset.coords:
                time_name = candidate
                break

        if time_name is None:
            for coordinate_name in dataset.coords:
                if np.issubdtype(
                    dataset[coordinate_name].dtype,
                    np.datetime64,
                ):
                    time_name = coordinate_name
                    break

        if time_name is None:
            raise KeyError("Could not detect AAO time coordinate.")

        preferred_variables = [
            "AAO",
            "aao",
            "SAM",
            "sam",
            "AAO_index",
            "SAM_index",
            "index",
        ]

        value_name = None

        for candidate in preferred_variables:
            if candidate in dataset.data_vars:
                value_name = candidate
                break

        if value_name is None:
            candidates = [
                name
                for name in dataset.data_vars
                if time_name in dataset[name].dims
            ]

            if not candidates:
                raise KeyError(
                    "Could not detect AAO variable.\n"
                    f"Available variables: {list(dataset.data_vars)}"
                )

            value_name = candidates[0]

        values = dataset[value_name]

        for dimension in list(values.dims):
            if dimension == time_name:
                continue

            values = values.isel({dimension: 0}, drop=True)

        time = (
            pd.DatetimeIndex(
                pd.to_datetime(values[time_name].values)
            )
            .to_period("M")
            .to_timestamp()
        )

        output = pd.DataFrame(
            {
                "time": time,
                "AAO": np.asarray(
                    values.values,
                    dtype=float,
                ).reshape(-1),
            }
        )

        dataset.close()

    else:
        dataframe = pd.read_csv(path)

        lower_columns = {
            column.lower(): column
            for column in dataframe.columns
        }

        if (
            "year" in lower_columns
            and "month" in lower_columns
        ):
            time = pd.to_datetime(
                {
                    "year": dataframe[lower_columns["year"]],
                    "month": dataframe[lower_columns["month"]],
                    "day": 1,
                }
            )
        else:
            time_column = next(
                (
                    lower_columns[name]
                    for name in ["time", "date", "month"]
                    if name in lower_columns
                ),
                None,
            )

            if time_column is None:
                raise KeyError("Could not detect AAO time column.")

            time = pd.to_datetime(dataframe[time_column])

        value_column = next(
            (
                lower_columns[name]
                for name in [
                    "aao",
                    "sam",
                    "aao_index",
                    "sam_index",
                    "index",
                ]
                if name in lower_columns
            ),
            None,
        )

        if value_column is None:
            numeric_columns = [
                column
                for column in dataframe.columns
                if pd.api.types.is_numeric_dtype(dataframe[column])
                and column.lower() not in {"year", "month"}
            ]

            if not numeric_columns:
                raise KeyError("Could not detect AAO value column.")

            value_column = numeric_columns[0]

        output = pd.DataFrame(
            {
                "time": time,
                "AAO": pd.to_numeric(
                    dataframe[value_column],
                    errors="coerce",
                ),
            }
        )

    output = output.dropna().sort_values("time")

    output["time"] = (
        pd.DatetimeIndex(output["time"])
        .to_period("M")
        .to_timestamp()
    )

    output = (
        output.groupby("time", as_index=False)["AAO"]
        .mean()
        .set_index("time")
    )

    output = output.loc[
        "2008-01-01":"2025-12-01"
    ]

    standard_deviation = output["AAO"].std(ddof=1)

    if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
        raise ValueError("AAO standard deviation is invalid.")

    output["AAO_std"] = (
        output["AAO"] - output["AAO"].mean()
    ) / standard_deviation

    output["AAO_rm3"] = (
        output["AAO_std"]
        .rolling(3, center=True, min_periods=1)
        .mean()
    )

    print(
        f"AAO months: {len(output)}, "
        f"{output.index.min():%Y-%m} to "
        f"{output.index.max():%Y-%m}"
    )

    return output


# 14. DESEASONALIZATION AND STATISTICS
def deseasonalize(dataframe):
    output = dataframe.copy()

    for column in output.columns:
        monthly_climatology = (
            output.groupby(output.index.month)[column]
            .transform("mean")
        )

        output[column] = output[column] - monthly_climatology

    return output


def lag1_autocorrelation(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    values = values[valid]

    if values.size < 4:
        return 0.0

    x = values[:-1]
    y = values[1:]

    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def pearson_ar1_corrected(x, y):
    dataframe = pd.concat([x, y], axis=1).dropna()

    n = len(dataframe)

    if n < 12:
        return np.nan, np.nan, n, np.nan

    x_values = dataframe.iloc[:, 0].values.astype(float)
    y_values = dataframe.iloc[:, 1].values.astype(float)

    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return np.nan, np.nan, n, np.nan

    correlation = float(pearsonr(x_values, y_values)[0])

    rx = lag1_autocorrelation(x_values)
    ry = lag1_autocorrelation(y_values)

    denominator = 1.0 + rx * ry

    if denominator <= 0.0:
        effective_n = float(n)
    else:
        effective_n = n * (1.0 - rx * ry) / denominator

    effective_n = float(np.clip(effective_n, 3.0, n))

    if abs(correlation) >= 1.0:
        p_value = 0.0
    else:
        statistic = correlation * np.sqrt(
            (effective_n - 2.0)
            / max(1.0e-12, 1.0 - correlation ** 2)
        )

        p_value = float(
            2.0
            * student_t.sf(
                abs(statistic),
                df=max(1.0, effective_n - 2.0),
            )
        )

    return correlation, p_value, n, effective_n


def apply_fdr(table, p_column="p_raw"):
    output = table.copy()

    p_values = output[p_column].values.astype(float)
    valid = np.isfinite(p_values)

    q_values = np.full(len(output), np.nan, dtype=float)

    if valid.sum() > 0:
        _, corrected, _, _ = multipletests(
            p_values[valid],
            method="fdr_bh",
        )

        q_values[valid] = corrected

    output["p_fdr"] = q_values

    output["sig"] = [
        "**"
        if np.isfinite(q) and q < 0.01
        else "*"
        if np.isfinite(q) and q < 0.05
        else "•"
        if np.isfinite(q) and q < 0.10
        else ""
        for q in q_values
    ]

    return output


def correlation_panel(target, drivers, driver_order):
    rows = []

    for sea_code in SEA_ORDER:
        for driver_name in driver_order:
            correlation, p_value, n, effective_n = (
                pearson_ar1_corrected(
                    target[sea_code],
                    drivers[driver_name][sea_code],
                )
            )

            rows.append(
                {
                    "sea": sea_code,
                    "driver": driver_name,
                    "r": correlation,
                    "p_raw": p_value,
                    "n": n,
                    "effective_n": effective_n,
                }
            )

    return apply_fdr(pd.DataFrame(rows))


def calculate_vif(predictors):
    complete = predictors.dropna()

    output = pd.Series(np.nan, index=predictors.columns)

    if len(complete) < predictors.shape[1] + 3:
        return output

    usable_columns = [
        column
        for column in complete.columns
        if complete[column].std(ddof=1) > 0.0
    ]

    if not usable_columns:
        return output

    matrix = sm.add_constant(
        complete[usable_columns]
    ).values

    values = [
        variance_inflation_factor(matrix, index)
        for index in range(1, matrix.shape[1])
    ]

    output.loc[usable_columns] = values
    return output


def standardized_hac_regression(target, predictors):
    complete = pd.concat(
        [target.rename("target"), predictors],
        axis=1,
    ).dropna()

    if len(complete) < max(24, predictors.shape[1] + 6):
        return None

    target_standardized = pd.Series(
        zscore(complete["target"].values),
        index=complete.index,
        name="target",
    )

    predictor_standardized = pd.DataFrame(
        index=complete.index
    )

    for column in predictors.columns:
        values = complete[column].values.astype(float)
        standard_deviation = np.std(values, ddof=1)

        if standard_deviation == 0.0:
            predictor_standardized[column] = np.nan
        else:
            predictor_standardized[column] = (
                values - np.mean(values)
            ) / standard_deviation

    regression_data = pd.concat(
        [target_standardized, predictor_standardized],
        axis=1,
    ).dropna()

    if len(regression_data) < max(24, predictors.shape[1] + 6):
        return None

    design_matrix = sm.add_constant(
        regression_data[predictors.columns]
    )

    model = sm.OLS(
        regression_data["target"],
        design_matrix,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 1},
    )

    return model


def regression_panel(target, drivers, predictor_order):
    rows = []

    for sea_code in SEA_ORDER:
        predictors = pd.DataFrame(
            {
                predictor: drivers[predictor][sea_code]
                for predictor in predictor_order
            }
        )

        vif = calculate_vif(predictors)

        model = standardized_hac_regression(
            target[sea_code],
            predictors,
        )

        for predictor in predictor_order:
            if model is None:
                beta = np.nan
                p_value = np.nan
                adjusted_r2 = np.nan
                n = 0
            else:
                beta = float(model.params.get(predictor, np.nan))
                p_value = float(model.pvalues.get(predictor, np.nan))
                adjusted_r2 = float(model.rsquared_adj)
                n = int(model.nobs)

            rows.append(
                {
                    "sea": sea_code,
                    "predictor": predictor,
                    "beta": beta,
                    "p_raw": p_value,
                    "adjusted_r2": adjusted_r2,
                    "vif": float(vif.get(predictor, np.nan)),
                    "n": n,
                }
            )

    return apply_fdr(pd.DataFrame(rows))


def lag_correlation(aao, sla):
    rows = []

    aao_values = aao.values.astype(float)

    for sea_code in SEA_ORDER:
        sla_values = sla[sea_code].values.astype(float)

        for lag in LAGS:
            if lag > 0:
                x = aao_values[:-lag]
                y = sla_values[lag:]

            elif lag < 0:
                x = aao_values[-lag:]
                y = sla_values[:lag]

            else:
                x = aao_values
                y = sla_values

            x_series = pd.Series(x)
            y_series = pd.Series(y)

            correlation, p_value, n, effective_n = (
                pearson_ar1_corrected(
                    x_series,
                    y_series,
                )
            )

            rows.append(
                {
                    "sea": sea_code,
                    "lag": int(lag),
                    "r": correlation,
                    "p_raw": p_value,
                    "n": n,
                    "effective_n": effective_n,
                }
            )

    return apply_fdr(pd.DataFrame(rows))


# 15. LOW-RAM SLA AAO COMPOSITE AND ONLINE YEAR BOOTSTRAP
def coarsened_coordinates(values, factor):
    usable = (len(values) // factor) * factor

    return (
        np.asarray(values[:usable], dtype=float)
        .reshape(-1, factor)
        .mean(axis=1)
    )


def coarsen_nanmean_2d(field, factor):
    if factor <= 1:
        return field.astype(np.float32, copy=False)

    n_latitude = (field.shape[0] // factor) * factor
    n_longitude = (field.shape[1] // factor) * factor

    trimmed = field[:n_latitude, :n_longitude]

    reshaped = trimmed.reshape(
        n_latitude // factor,
        factor,
        n_longitude // factor,
        factor,
    )

    return np.nanmean(
        reshaped,
        axis=(1, 3),
    ).astype(np.float32)


def build_or_load_composite(aao):
    if (
        REUSE_CACHED_COMPOSITE
        and OUT_COMPOSITE.exists()
    ):
        dataset = xr.open_dataset(OUT_COMPOSITE)

        composite = dataset["composite"].load()
        significant = dataset["fdr_significant"].load().astype(bool)

        n_positive = int(dataset.attrs["n_positive_months"])
        n_negative = int(dataset.attrs["n_negative_months"])

        dataset.close()

        print("Loaded cached AAO composite.")
        return composite, significant, n_positive, n_negative

    dataset = standardize_dataset(
        open_dataset_safely(FILE_PATHS["sla"])
    )

    variable_name = detect_variable(
        dataset,
        ["sla", "SLA"],
        ["sla", "sea_level_anomaly"],
        "SLA",
    )

    data_array = prepare_monthly_field(
        dataset[variable_name]
    )

    scale = height_scale_to_cm(data_array)

    common_time = (
        pd.Index(pd.DatetimeIndex(data_array["time"].values))
        .intersection(pd.Index(aao.index))
        .sort_values()
    )

    data_array = data_array.sel(time=common_time)
    aao_common = aao.loc[common_time]

    positive_times = set(
        aao_common.index[
            aao_common["AAO_std"] >= AAO_POSITIVE_THRESHOLD
        ]
    )

    negative_times = set(
        aao_common.index[
            aao_common["AAO_std"] <= AAO_NEGATIVE_THRESHOLD
        ]
    )

    if not positive_times or not negative_times:
        raise RuntimeError(
            "No positive or negative AAO phase months were found."
        )

    factor = COMPOSITE_COARSEN_FACTOR

    latitude = coarsened_coordinates(
        data_array["lat"].values,
        factor,
    )

    longitude = coarsened_coordinates(
        data_array["lon"].values,
        factor,
    )

    shape = (len(latitude), len(longitude))

    positive_years = sorted(
        {timestamp.year for timestamp in positive_times}
    )

    negative_years = sorted(
        {timestamp.year for timestamp in negative_times}
    )

    positive_year_index = {
        year: index
        for index, year in enumerate(positive_years)
    }

    negative_year_index = {
        year: index
        for index, year in enumerate(negative_years)
    }

    positive_sum = np.zeros(
        (len(positive_years),) + shape,
        dtype=np.float32,
    )

    positive_count = np.zeros(
        (len(positive_years),) + shape,
        dtype=np.uint16,
    )

    negative_sum = np.zeros(
        (len(negative_years),) + shape,
        dtype=np.float32,
    )

    negative_count = np.zeros(
        (len(negative_years),) + shape,
        dtype=np.uint16,
    )

    print(
        "Building coarsened AAO SLA composite month-by-month.\n"
        f"  Coarsening factor: {factor}\n"
        f"  Composite grid: {shape}"
    )

    for time_index, timestamp in enumerate(common_time):
        timestamp = pd.Timestamp(timestamp)

        if (
            timestamp not in positive_times
            and timestamp not in negative_times
        ):
            continue

        field = np.asarray(
            data_array.isel(time=time_index).values,
            dtype=np.float32,
        ).squeeze()

        field = field * np.float32(scale)
        field = coarsen_nanmean_2d(field, factor)

        valid = np.isfinite(field)

        if timestamp in positive_times:
            year_index = positive_year_index[timestamp.year]

            positive_sum[year_index][valid] += field[valid]
            positive_count[year_index][valid] += 1

        if timestamp in negative_times:
            year_index = negative_year_index[timestamp.year]

            negative_sum[year_index][valid] += field[valid]
            negative_count[year_index][valid] += 1

        del field, valid

    positive_total_sum = positive_sum.sum(axis=0, dtype=np.float64)
    positive_total_count = positive_count.sum(axis=0, dtype=np.float64)

    negative_total_sum = negative_sum.sum(axis=0, dtype=np.float64)
    negative_total_count = negative_count.sum(axis=0, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_composite = (
            positive_total_sum
            / positive_total_count
            - negative_total_sum
            / negative_total_count
        )

    observed_composite[
        (positive_total_count == 0)
        | (negative_total_count == 0)
    ] = np.nan

    # Online bootstrap counters: no bootstrap-map cube is retained.
    greater_than_zero = np.zeros(shape, dtype=np.uint16)
    less_than_zero = np.zeros(shape, dtype=np.uint16)
    valid_bootstrap_count = np.zeros(shape, dtype=np.uint16)

    random_generator = np.random.default_rng(
        COMPOSITE_BOOTSTRAP_SEED
    )

    print(
        "Running online year-block bootstrap:\n"
        f"  Replicates: {COMPOSITE_BOOTSTRAP_REPLICATES}"
    )

    positive_sum_float = positive_sum.astype(np.float32)
    positive_count_float = positive_count.astype(np.float32)
    negative_sum_float = negative_sum.astype(np.float32)
    negative_count_float = negative_count.astype(np.float32)

    for bootstrap_index in range(
        COMPOSITE_BOOTSTRAP_REPLICATES
    ):
        sampled_positive = random_generator.integers(
            0,
            len(positive_years),
            size=len(positive_years),
        )

        sampled_negative = random_generator.integers(
            0,
            len(negative_years),
            size=len(negative_years),
        )

        positive_frequency = np.bincount(
            sampled_positive,
            minlength=len(positive_years),
        ).astype(np.float32)

        negative_frequency = np.bincount(
            sampled_negative,
            minlength=len(negative_years),
        ).astype(np.float32)

        bootstrap_positive_sum = np.tensordot(
            positive_frequency,
            positive_sum_float,
            axes=(0, 0),
        )

        bootstrap_positive_count = np.tensordot(
            positive_frequency,
            positive_count_float,
            axes=(0, 0),
        )

        bootstrap_negative_sum = np.tensordot(
            negative_frequency,
            negative_sum_float,
            axes=(0, 0),
        )

        bootstrap_negative_count = np.tensordot(
            negative_frequency,
            negative_count_float,
            axes=(0, 0),
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            bootstrap_composite = (
                bootstrap_positive_sum
                / bootstrap_positive_count
                - bootstrap_negative_sum
                / bootstrap_negative_count
            )

        valid = (
            (bootstrap_positive_count > 0)
            & (bootstrap_negative_count > 0)
            & np.isfinite(bootstrap_composite)
        )

        greater_than_zero[valid] += (
            bootstrap_composite[valid] > 0.0
        )

        less_than_zero[valid] += (
            bootstrap_composite[valid] < 0.0
        )

        valid_bootstrap_count[valid] += 1

        del (
            positive_frequency,
            negative_frequency,
            bootstrap_positive_sum,
            bootstrap_positive_count,
            bootstrap_negative_sum,
            bootstrap_negative_count,
            bootstrap_composite,
            valid,
        )

        if (
            (bootstrap_index + 1) % 50 == 0
            or bootstrap_index
            == COMPOSITE_BOOTSTRAP_REPLICATES - 1
        ):
            print(
                f"  Bootstrap "
                f"{bootstrap_index + 1:03d}/"
                f"{COMPOSITE_BOOTSTRAP_REPLICATES:03d}"
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        probability_positive = (
            greater_than_zero.astype(float) + 1.0
        ) / (
            valid_bootstrap_count.astype(float) + 2.0
        )

        probability_negative = (
            less_than_zero.astype(float) + 1.0
        ) / (
            valid_bootstrap_count.astype(float) + 2.0
        )

        bootstrap_p = (
            2.0
            * np.minimum(
                probability_positive,
                probability_negative,
            )
        )

    bootstrap_p = np.clip(bootstrap_p, 0.0, 1.0)
    bootstrap_p[valid_bootstrap_count == 0] = np.nan

    bootstrap_q = np.full(shape, np.nan, dtype=float)

    finite = np.isfinite(bootstrap_p)

    if finite.sum() > 0:
        _, corrected, _, _ = multipletests(
            bootstrap_p[finite],
            method="fdr_bh",
        )

        bootstrap_q[finite] = corrected

    significant = bootstrap_q < 0.05

    composite_data_array = xr.DataArray(
        observed_composite.astype(np.float32),
        coords={
            "lat": latitude,
            "lon": longitude,
        },
        dims=("lat", "lon"),
        name="composite",
        attrs={
            "long_name": (
                "Positive-minus-negative AAO SLA composite"
            ),
            "units": "cm",
        },
    )

    output_dataset = xr.Dataset(
        {
            "composite": composite_data_array,
            "bootstrap_p": (
                ("lat", "lon"),
                bootstrap_p.astype(np.float32),
            ),
            "bootstrap_q": (
                ("lat", "lon"),
                bootstrap_q.astype(np.float32),
            ),
            "fdr_significant": (
                ("lat", "lon"),
                significant.astype(np.int8),
            ),
        },
        attrs={
            "n_positive_months": len(positive_times),
            "n_negative_months": len(negative_times),
            "coarsen_factor": factor,
            "bootstrap_replicates": COMPOSITE_BOOTSTRAP_REPLICATES,
            "description": (
                "Low-RAM year-block bootstrap AAO SLA composite"
            ),
        },
    )

    encoding = {
        variable_name: {
            "zlib": True,
            "complevel": 4,
        }
        for variable_name in output_dataset.data_vars
    }

    output_dataset.to_netcdf(
        OUT_COMPOSITE,
        encoding=encoding,
    )

    dataset.close()

    del (
        dataset,
        data_array,
        positive_sum,
        positive_count,
        negative_sum,
        negative_count,
        positive_sum_float,
        positive_count_float,
        negative_sum_float,
        negative_count_float,
        greater_than_zero,
        less_than_zero,
        valid_bootstrap_count,
        output_dataset,
    )

    gc.collect()

    return (
        composite_data_array,
        xr.DataArray(
            significant,
            coords={
                "lat": latitude,
                "lon": longitude,
            },
            dims=("lat", "lon"),
        ),
        len(positive_times),
        len(negative_times),
    )


# 17. MAIN PROCESSING
print("=" * 78)
print("FIGURE 13 — LOW-RAM PROCESSING")
print("=" * 78)

if REUSE_CACHED_SECTOR_SERIES:
    print(
        "Checkpoint resume is ON. Previously completed sector CSV files "
        "will be loaded and will NOT be recomputed."
    )
else:
    print(
        "Checkpoint resume is OFF. All sector variables will be recomputed."
    )

for key, path in FILE_PATHS.items():
    require_file(path, key)

gebco = load_reduced_gebco()

# Process one file at a time. Each function closes its NetCDF immediately.
sector_tables = {}

sector_tables["SLA"] = process_standard_field(
    "SLA",
    FILE_PATHS["sla"],
    ["sla", "SLA"],
    ["sla", "sea_level_anomaly"],
    gebco,
    apply_height_cm_conversion=True,
)

sector_tables["Total steric"] = process_standard_field(
    "Total steric",
    FILE_PATHS["total_steric"],
    ["total_steric_height"],
    ["total", "steric"],
    gebco,
    apply_height_cm_conversion=True,
)

sector_tables["Thermosteric"] = process_standard_field(
    "Thermosteric",
    FILE_PATHS["thermosteric"],
    ["thermosteric_height"],
    ["thermosteric"],
    gebco,
    apply_height_cm_conversion=True,
)

sector_tables["Halosteric"] = process_standard_field(
    "Halosteric",
    FILE_PATHS["halosteric"],
    ["halosteric_height"],
    ["halosteric"],
    gebco,
    apply_height_cm_conversion=True,
)

sector_tables["OHC"] = process_standard_field(
    "OHC",
    FILE_PATHS["ohc"],
    ["ocean_heat_content"],
    ["ocean_heat_content", "ohc"],
    gebco,
)

sector_tables["SST"] = process_standard_field(
    "SST",
    FILE_PATHS["sst"],
    ["SST", "sst", "analysed_sst"],
    ["sst", "sea_surface_temperature"],
    gebco,
)

sector_tables[HEAT_FLUX_KEY] = process_qnet_proxy(gebco)

sector_tables["WSC"] = process_wind_stress_curl(gebco)

sector_tables["MLD"] = process_standard_field(
    "MLD",
    FILE_PATHS["mld"],
    ["MLD", "mld"],
    ["mixed_layer_depth", "mld"],
    gebco,
)

sector_tables["SIC"] = process_standard_field(
    "SIC",
    FILE_PATHS["sic"],
    ["SIF", "sic", "sea_ice_fraction"],
    ["sea_ice_fraction", "sic", "sif"],
    gebco,
)

sector_tables["FWC"] = process_standard_field(
    "FWC",
    FILE_PATHS["fwc"],
    ["freshwater_content"],
    ["freshwater_content", "fwc"],
    gebco,
)

sector_tables["SSS anomaly"] = process_standard_field(
    "SSS anomaly",
    FILE_PATHS["sss_anomaly"],
    ["surface_salinity_anomaly"],
    ["surface_salinity_anomaly", "sss"],
    gebco,
)

sector_tables["Snowfall"] = process_standard_field(
    "Snowfall",
    FILE_PATHS["snowfall"],
    ["snowfall", "sf", "P_Snow"],
    ["snowfall", "snow"],
    gebco,
)

sector_tables["Stratification"] = process_standard_field(
    "Stratification",
    FILE_PATHS["stratification"],
    ["stratification_20_200m"],
    ["stratification"],
    gebco,
)

aao = read_aao()

# Determine one exact common monthly sequence.
common_time = pd.Index(aao.index)

for table in sector_tables.values():
    common_time = common_time.intersection(
        pd.Index(table.index)
    )

common_time = common_time.sort_values()

if len(common_time) < 120:
    raise RuntimeError(
        f"Only {len(common_time)} common months remain. "
        "Inspect input time coordinates."
    )

print(
    f"Common analysis months: {len(common_time)}, "
    f"{common_time.min():%Y-%m} to "
    f"{common_time.max():%Y-%m}"
)

for key in sector_tables:
    sector_tables[key] = sector_tables[key].loc[
        common_time,
        SEA_ORDER,
    ]

aao = aao.loc[common_time]

# AAO is identical for all seas.
aao_sector = pd.DataFrame(
    {
        sea_code: aao["AAO_std"].values
        for sea_code in SEA_ORDER
    },
    index=common_time,
)

sector_tables["AAO"] = aao_sector

# Remove each variable's calendar-month climatology.
anomaly_tables = {}

for key, table in sector_tables.items():
    if key == "AAO":
        anomaly_tables[key] = table.copy()
    else:
        anomaly_tables[key] = deseasonalize(table)

# Save the compact combined matrix.
combined = pd.DataFrame(index=common_time)

for key, table in anomaly_tables.items():
    for sea_code in SEA_ORDER:
        combined[f"{sea_code}|{key}"] = table[sea_code]

combined.to_csv(OUT_SECTOR_TS, index_label="time")

# Free non-anomaly versions except AAO.
del sector_tables, combined
gc.collect()


# 18. PANELS (a)–(g): STATISTICS
panel_a = correlation_panel(
    anomaly_tables["SLA"],
    {
        "Total steric": anomaly_tables["Total steric"],
        "Thermosteric": anomaly_tables["Thermosteric"],
        "Halosteric": anomaly_tables["Halosteric"],
    },
    ["Total steric", "Thermosteric", "Halosteric"],
)

panel_b = correlation_panel(
    anomaly_tables["Thermosteric"],
    {
        driver: anomaly_tables[driver]
        for driver in THERMO_DRIVER_ORDER
    },
    THERMO_DRIVER_ORDER,
)

panel_c = correlation_panel(
    anomaly_tables["Halosteric"],
    {
        driver: anomaly_tables[driver]
        for driver in HALO_DRIVER_ORDER
    },
    HALO_DRIVER_ORDER,
)

panel_d = regression_panel(
    anomaly_tables["Thermosteric"],
    {
        predictor: anomaly_tables[predictor]
        for predictor in THERMO_REG_ORDER
    },
    THERMO_REG_ORDER,
)

panel_e = regression_panel(
    anomaly_tables["Halosteric"],
    {
        predictor: anomaly_tables[predictor]
        for predictor in HALO_REG_ORDER
    },
    HALO_REG_ORDER,
)

panel_g = lag_correlation(
    aao["AAO_std"],
    anomaly_tables["SLA"],
)

panel_a.to_csv(OUT_TABLE_A, index=False)
panel_b.to_csv(OUT_TABLE_B, index=False)
panel_c.to_csv(OUT_TABLE_C, index=False)
panel_d.to_csv(OUT_TABLE_D, index=False)
panel_e.to_csv(OUT_TABLE_E, index=False)
panel_g.to_csv(OUT_TABLE_G, index=False)


# 19. PANEL (h): LOW-RAM COMPOSITE
(
    composite,
    composite_significant,
    n_positive,
    n_negative,
) = build_or_load_composite(aao)


# DATA PRODUCT GROUP 11: SUPPLEMENTARY FIGURE S1 ARGO-COVERAGE SUPPORT PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: Figure_S1_Argo_coverage_2001_2025_FINAL_LAYOUT_sea_season_counts.csv
# File name: Figure_S1_Argo_coverage_2001_2025_FINAL_LAYOUT_depth_threshold_percentages.csv
# File name: Figure_S1_Argo_coverage_2001_2025_FINAL_LAYOUT_seasonal_profile_count_maps.nc
# File name: Figure_S1_Argo_coverage_2001_2025_FINAL_LAYOUT_profile_summary.csv.gz

# SUPPLEMENTARY FIGURE S1 — ARGO COVERAGE
# Full corrected code for:
#   argo_SO_profiles_2001_2025_cleaned_gridded.nc
#
# Panels:
#   (a) Spring profile-count map
#   (b) Summer profile-count map
#   (c) Autumn profile-count map
#   (d) Winter profile-count map
#   (e) Sea × season profile-count heatmap
#   (f) Percentage reaching 1000 and 2000 dbar
#
# IMPORTANT CORRECTION
# --------------------
# This dataset has:
#   PRES_GRID (N_LEVELS)
#   TEMP      (N_PROF, N_LEVELS)
#   PSAL      (N_PROF, N_LEVELS)
#
# Therefore, maximum pressure is calculated profile-by-profile as the deepest
# PRES_GRID level at which BOTH temperature and salinity are finite.
#
# No overall figure title is added.
# Output is saved at 1080 dpi.



# 0. INSTALL REQUIRED PACKAGES WHEN NEEDED
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
    "netCDF4": "netCDF4",
    "h5netcdf": "h5netcdf",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import gc
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=RuntimeWarning)


# 3. USER SETTINGS
ARGO_FILE = Path(
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "argo_SO_profiles_2001_2025_cleaned_gridded.nc"
)

OUTPUT_DIR = Path("/content/drive/MyDrive/SAM_Thesis/paper2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = "Figure_S1_Argo_coverage_2001_2025_FINAL_LAYOUT"

OUTPUT_HEATMAP_CSV = OUTPUT_DIR / f"{OUTPUT_STEM}_sea_season_counts.csv"
OUTPUT_DEPTH_CSV = OUTPUT_DIR / f"{OUTPUT_STEM}_depth_threshold_percentages.csv"
OUTPUT_SEASONAL_NC = OUTPUT_DIR / f"{OUTPUT_STEM}_seasonal_profile_count_maps.nc"
OUTPUT_PROFILE_SUMMARY = OUTPUT_DIR / f"{OUTPUT_STEM}_profile_summary.csv.gz"
OUTPUT_LOG = OUTPUT_DIR / f"{OUTPUT_STEM}_processing_log.txt"

# Region and gridding
LAT_MIN = -90.0
LAT_MAX = -60.0
LON_MIN = -180.0
LON_MAX = 180.0

MAP_GRID_RESOLUTION_DEG = 1.0
PROFILE_CHUNK_SIZE = 5000

# Set False after the first successful run if you want to avoid writing the
# large profile-level compressed CSV. Aggregated CSV/NetCDF outputs are always
# written.
SAVE_PROFILE_LEVEL_SUMMARY = True

# Southern Hemisphere seasons
SEASONS = OrderedDict([
    ("Spring", [9, 10, 11]),
    ("Summer", [12, 1, 2]),
    ("Autumn", [3, 4, 5]),
    ("Winter", [6, 7, 8]),
])

# 13 Antarctic shelf-sea longitude sectors.
# These are configurable. Keep them identical to the boundaries used in your
# main-paper Figures 4 and 5 if those figures use slightly different limits.
# A wrapped sector is represented by lon_min > lon_max.
SEA_BOUNDS = OrderedDict([
    ("WED", (-60.0,  -20.0)),
    ("KHV", (-20.0,   10.0)),
    ("RLS", ( 10.0,   35.0)),
    ("LAZ", ( 35.0,   60.0)),
    ("COS", ( 60.0,   90.0)),
    ("COO", ( 90.0,  115.0)),
    ("DAV", (115.0,  130.0)),
    ("MAW", (130.0,  150.0)),
    ("DUR", (150.0,  170.0)),
    ("SOM", (170.0, -160.0)),
    ("ROS", (-160.0, -130.0)),
    ("AMU", (-130.0, -100.0)),
    ("BEL", (-100.0,  -60.0)),
])

SEA_ORDER = list(SEA_BOUNDS.keys())

# Figure settings
SAVE_DPI = 1080
FIGSIZE = (11.2, 7.8)

FONT_PANEL = 12.2
FONT_MAP_TICK = 6.5
FONT_AXIS = 9.2
FONT_TICK = 8.0
FONT_CELL = 8.3
FONT_LEGEND = 8.2
FONT_NOTE = 8.3

LAND_COLOR = "#d9d9d9"
COAST_COLOR = "#666666"
GRID_COLOR = "#8c8c8c"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})


# 4. GENERAL HELPERS
def open_dataset_safely(path):
    attempts = []

    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = {
                "decode_times": False,
                "mask_and_scale": True,
                "cache": False,
            }

            if engine is not None:
                kwargs["engine"] = engine

            dataset = xr.open_dataset(path, **kwargs)

            print(
                f"Opened {Path(path).name} with "
                f"engine={engine or 'xarray-default'}"
            )

            return dataset

        except Exception as error:
            attempts.append(f"{engine}: {error}")

    raise RuntimeError(
        f"Could not open file:\n{path}\n\n" + "\n".join(attempts)
    )


def normalize_longitude(longitude):
    longitude = np.asarray(longitude, dtype=np.float64)
    return ((longitude + 180.0) % 360.0) - 180.0


def convert_juld_to_datetime(juld_data_array):
    """
    Convert Argo JULD to monthly-compatible pandas timestamps.

    Standard Argo JULD is days since 1950-01-01 00:00:00 UTC.
    The code first checks the variable's units attribute. If units are absent,
    it uses the standard Argo epoch and prints a warning.
    """
    values = np.asarray(juld_data_array.values)

    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(values))

    units = str(juld_data_array.attrs.get("units", "")).strip()
    calendar = str(juld_data_array.attrs.get("calendar", "standard")).strip()

    if "since" in units.lower():
        temporary = xr.Dataset(
            {
                "JULD": xr.DataArray(
                    values,
                    dims=("N_PROF",),
                    attrs={"units": units, "calendar": calendar},
                )
            }
        )

        try:
            decoded = xr.decode_cf(temporary)["JULD"].values
            return pd.DatetimeIndex(pd.to_datetime(decoded))
        except Exception as error:
            print(
                "WARNING: CF decoding of JULD failed; using standard Argo "
                f"epoch instead. Reason: {error}"
            )

    if not units:
        print(
            "WARNING: JULD has no units attribute. The standard Argo epoch "
            "1950-01-01 is being used."
        )
    else:
        print(
            f"WARNING: JULD units {units!r} were not decoded. The standard "
            "Argo epoch 1950-01-01 is being used."
        )

    numeric = pd.to_numeric(
        pd.Series(values.reshape(-1)),
        errors="coerce",
    ).to_numpy(dtype=float)

    result = pd.DatetimeIndex(
        pd.Timestamp("1950-01-01")
        + pd.to_timedelta(numeric, unit="D")
    )

    return result


def month_to_season(month):
    for season_name, months in SEASONS.items():
        if int(month) in months:
            return season_name
    return np.nan


def longitude_in_sector(longitude, lon_min, lon_max):
    if lon_min <= lon_max:
        return (longitude >= lon_min) & (longitude < lon_max)

    return (longitude >= lon_min) | (longitude < lon_max)


def assign_sea_codes(longitudes, latitudes):
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)

    output = np.full(longitudes.shape, None, dtype=object)

    in_latitude_range = (
        np.isfinite(latitudes)
        & (latitudes >= LAT_MIN)
        & (latitudes <= LAT_MAX)
    )

    for sea_code, (lon_min, lon_max) in SEA_BOUNDS.items():
        mask = (
            in_latitude_range
            & longitude_in_sector(longitudes, lon_min, lon_max)
        )

        output[mask] = sea_code

    return output


def power_of_ten_ceiling(value):
    if not np.isfinite(value) or value <= 1.0:
        return 10.0

    return float(10.0 ** np.ceil(np.log10(value)))


def logarithmic_ticks(vmax):
    maximum_power = int(np.ceil(np.log10(max(vmax, 10.0))))
    return [10.0 ** power for power in range(0, maximum_power + 1)]


def format_log_tick(value, _position=None):
    if value >= 1000:
        return f"{int(round(value)):,}"
    return f"{int(round(value))}"


def circular_map_boundary():
    theta = np.linspace(0.0, 2.0 * np.pi, 361)

    vertices = (
        np.vstack([np.sin(theta), np.cos(theta)]).T
        * 0.5
        + np.array([0.5, 0.5])
    )

    return MplPath(vertices)


def add_manual_polar_labels(axis):
    # Six longitude labels keep four small maps readable and non-overlapping.
    longitude_labels = [
        (0.0, "0°"),
        (60.0, "60°E"),
        (120.0, "120°E"),
        (180.0, "180°"),
        (-120.0, "120°W"),
        (-60.0, "60°W"),
    ]

    for longitude, text in longitude_labels:
        axis.text(
            longitude,
            -58.7,
            text,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=FONT_MAP_TICK,
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )

    latitude_labels = [
        (-60.0, "60°S"),
        (-70.0, "70°S"),
        (-80.0, "80°S"),
    ]

    for latitude, text in latitude_labels:
        axis.text(
            0.0,
            latitude,
            text,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=FONT_MAP_TICK,
            fontweight="bold",
            zorder=20,
        )

    axis.text(
        0.5,
        0.5,
        "90°S",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=FONT_MAP_TICK,
        fontweight="bold",
        zorder=20,
    )


def configure_polar_axis(axis):
    axis.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    axis.set_boundary(
        circular_map_boundary(),
        transform=axis.transAxes,
    )

    axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(-180.0, 181.0, 30.0),
        ylocs=[-60.0, -70.0, -80.0],
        linewidth=0.42,
        color=GRID_COLOR,
        alpha=0.62,
        linestyle=":",
        zorder=2,
    )

    # Cartopy Natural Earth land is used only as a visual background.
    axis.add_feature(
        cfeature.LAND,
        facecolor=LAND_COLOR,
        edgecolor=COAST_COLOR,
        linewidth=0.42,
        zorder=8,
    )

    axis.coastlines(
        resolution="110m",
        color=COAST_COLOR,
        linewidth=0.42,
        zorder=9,
    )

    add_manual_polar_labels(axis)


def add_panel_title(axis, title):
    axis.text(
        0.01,
        1.06,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_PANEL,
        fontweight="bold",
        clip_on=False,
    )


# 5. PROFILE DEPTH CALCULATION USING PRES_GRID
def calculate_profile_depth_support(dataset):
    """
    Return three one-dimensional arrays with length N_PROF:
      maximum_common_ts_pressure
      reaches_1000_dbar
      reaches_2000_dbar

    A pressure level is accepted only when BOTH TEMP and PSAL are finite.
    Processing is chunked along N_PROF to avoid loading both full 132 MB
    variables into memory simultaneously.
    """
    required_variables = ["PRES_GRID", "TEMP", "PSAL"]

    missing = [
        variable_name
        for variable_name in required_variables
        if variable_name not in dataset.variables
    ]

    if missing:
        raise KeyError(
            "The following required variables are missing: "
            + ", ".join(missing)
        )

    pressure_grid = np.asarray(
        dataset["PRES_GRID"].values,
        dtype=np.float32,
    ).reshape(-1)

    if pressure_grid.size != dataset.sizes["N_LEVELS"]:
        raise ValueError(
            "PRES_GRID length does not match the N_LEVELS dimension."
        )

    if not np.isfinite(pressure_grid).any():
        raise ValueError("PRES_GRID contains no finite values.")

    number_of_profiles = int(dataset.sizes["N_PROF"])

    maximum_pressure = np.full(
        number_of_profiles,
        np.nan,
        dtype=np.float32,
    )

    reaches_1000 = np.zeros(number_of_profiles, dtype=bool)
    reaches_2000 = np.zeros(number_of_profiles, dtype=bool)

    pressure_matrix = pressure_grid[None, :]
    threshold_1000 = pressure_grid >= 1000.0
    threshold_2000 = pressure_grid >= 2000.0

    print("\nCalculating profile depth support from PRES_GRID + valid TEMP/PSAL...")
    print(f"Profiles       : {number_of_profiles:,}")
    print(f"Pressure levels: {pressure_grid.size}")
    print(
        f"Pressure range : {np.nanmin(pressure_grid):.2f} to "
        f"{np.nanmax(pressure_grid):.2f} dbar"
    )
    print(f"Chunk size     : {PROFILE_CHUNK_SIZE:,} profiles")

    for start in range(0, number_of_profiles, PROFILE_CHUNK_SIZE):
        stop = min(start + PROFILE_CHUNK_SIZE, number_of_profiles)

        temperature = np.asarray(
            dataset["TEMP"].isel(N_PROF=slice(start, stop)).values,
            dtype=np.float32,
        )

        salinity = np.asarray(
            dataset["PSAL"].isel(N_PROF=slice(start, stop)).values,
            dtype=np.float32,
        )

        if temperature.shape != salinity.shape:
            raise ValueError(
                "TEMP and PSAL chunk shapes do not match: "
                f"{temperature.shape} versus {salinity.shape}"
            )

        valid_common_ts = (
            np.isfinite(temperature)
            & np.isfinite(salinity)
        )

        deepest = np.max(
            np.where(valid_common_ts, pressure_matrix, -np.inf),
            axis=1,
        )

        deepest[~np.isfinite(deepest)] = np.nan

        maximum_pressure[start:stop] = deepest.astype(np.float32)

        if threshold_1000.any():
            reaches_1000[start:stop] = np.any(
                valid_common_ts[:, threshold_1000],
                axis=1,
            )

        if threshold_2000.any():
            reaches_2000[start:stop] = np.any(
                valid_common_ts[:, threshold_2000],
                axis=1,
            )

        del temperature, salinity, valid_common_ts, deepest
        gc.collect()

        print(
            f"  [{stop:>7,}/{number_of_profiles:,}] "
            f"profiles processed"
        )

    return maximum_pressure, reaches_1000, reaches_2000


# 6. BUILD PROFILE-LEVEL SUMMARY
def build_profile_summary(dataset):
    required_variables = ["JULD", "LATITUDE", "LONGITUDE"]

    missing = [
        variable_name
        for variable_name in required_variables
        if variable_name not in dataset.variables
    ]

    if missing:
        raise KeyError(
            "Required profile coordinates are missing: "
            + ", ".join(missing)
        )

    number_of_profiles = int(dataset.sizes["N_PROF"])

    latitude = np.asarray(
        dataset["LATITUDE"].values,
        dtype=np.float32,
    ).reshape(-1)

    longitude = normalize_longitude(
        dataset["LONGITUDE"].values
    ).astype(np.float32)

    dates = convert_juld_to_datetime(dataset["JULD"])

    if not (
        len(latitude)
        == len(longitude)
        == len(dates)
        == number_of_profiles
    ):
        raise ValueError(
            "Profile coordinate lengths do not match N_PROF."
        )

    maximum_pressure, reaches_1000, reaches_2000 = (
        calculate_profile_depth_support(dataset)
    )

    summary = pd.DataFrame({
        "profile_index": np.arange(number_of_profiles, dtype=np.int64),
        "time": dates,
        "latitude": latitude,
        "longitude": longitude,
        "maximum_common_ts_pressure_dbar": maximum_pressure,
        "reaches_1000_dbar": reaches_1000,
        "reaches_2000_dbar": reaches_2000,
    })

    summary = summary.replace([np.inf, -np.inf], np.nan)

    summary = summary.dropna(
        subset=["time", "latitude", "longitude"]
    ).copy()

    summary = summary[
        (summary["latitude"] >= LAT_MIN)
        & (summary["latitude"] <= LAT_MAX)
    ].copy()

    summary["month"] = summary["time"].dt.month.astype(int)
    summary["season"] = summary["month"].map(month_to_season)

    summary["sea"] = assign_sea_codes(
        summary["longitude"].to_numpy(),
        summary["latitude"].to_numpy(),
    )

    summary = summary.dropna(
        subset=["season", "sea"]
    ).copy()

    summary["sea"] = pd.Categorical(
        summary["sea"],
        categories=SEA_ORDER,
        ordered=True,
    )

    summary["season"] = pd.Categorical(
        summary["season"],
        categories=list(SEASONS.keys()),
        ordered=True,
    )

    summary = summary.sort_values(
        ["time", "profile_index"]
    ).reset_index(drop=True)

    print("\nProfile summary created:")
    print(f"Retained profiles : {len(summary):,}")
    print(f"First date        : {summary['time'].min()}")
    print(f"Last date         : {summary['time'].max()}")
    print(
        "Profiles with valid common T–S depth: "
        f"{summary['maximum_common_ts_pressure_dbar'].notna().sum():,}"
    )

    return summary


# 7. AGGREGATED STATISTICS
def create_seasonal_profile_count_maps(profile_summary):
    longitude_edges = np.arange(
        LON_MIN,
        LON_MAX + MAP_GRID_RESOLUTION_DEG,
        MAP_GRID_RESOLUTION_DEG,
    )

    latitude_edges = np.arange(
        LAT_MIN,
        LAT_MAX + MAP_GRID_RESOLUTION_DEG,
        MAP_GRID_RESOLUTION_DEG,
    )

    longitude_centres = (
        longitude_edges[:-1]
        + 0.5 * MAP_GRID_RESOLUTION_DEG
    )

    latitude_centres = (
        latitude_edges[:-1]
        + 0.5 * MAP_GRID_RESOLUTION_DEG
    )

    data_arrays = []

    for season_name in SEASONS.keys():
        subset = profile_summary[
            profile_summary["season"] == season_name
        ]

        counts, _, _ = np.histogram2d(
            subset["latitude"].to_numpy(dtype=float),
            subset["longitude"].to_numpy(dtype=float),
            bins=[latitude_edges, longitude_edges],
        )

        data_arrays.append(
            xr.DataArray(
                counts.astype(np.int32),
                coords={
                    "lat": latitude_centres,
                    "lon": longitude_centres,
                },
                dims=("lat", "lon"),
                name=season_name,
            )
        )

    seasonal = xr.concat(
        data_arrays,
        dim=pd.Index(list(SEASONS.keys()), name="season"),
    )

    seasonal.name = "profile_count"

    seasonal.attrs.update({
        "long_name": "Seasonal Argo profile count per map grid cell",
        "grid_resolution_degrees": MAP_GRID_RESOLUTION_DEG,
        "profile_definition": "One count per Argo profile",
        "period": "2001-2025",
    })

    return seasonal


def create_sea_season_count_table(profile_summary):
    table = pd.crosstab(
        profile_summary["sea"],
        profile_summary["season"],
        dropna=False,
    )

    table = table.reindex(
        index=SEA_ORDER,
        columns=list(SEASONS.keys()),
        fill_value=0,
    )

    return table.astype(int)


def create_depth_threshold_table(profile_summary):
    rows = []

    for sea_code in SEA_ORDER:
        subset = profile_summary[
            profile_summary["sea"] == sea_code
        ]

        number_of_profiles = len(subset)

        if number_of_profiles == 0:
            percentage_1000 = np.nan
            percentage_2000 = np.nan
            count_1000 = 0
            count_2000 = 0
        else:
            count_1000 = int(subset["reaches_1000_dbar"].sum())
            count_2000 = int(subset["reaches_2000_dbar"].sum())

            percentage_1000 = (
                100.0 * count_1000 / number_of_profiles
            )

            percentage_2000 = (
                100.0 * count_2000 / number_of_profiles
            )

        rows.append({
            "sea": sea_code,
            "total_profiles": number_of_profiles,
            "profiles_reaching_1000_dbar": count_1000,
            "percentage_reaching_1000_dbar": percentage_1000,
            "profiles_reaching_2000_dbar": count_2000,
            "percentage_reaching_2000_dbar": percentage_2000,
        })

    return pd.DataFrame(rows).set_index("sea")


# 9. MAIN PROCESSING
print("=" * 78)
print("LOADING ARGO DATA")
print("=" * 78)

if not ARGO_FILE.exists():
    raise FileNotFoundError(f"Argo file not found:\n{ARGO_FILE}")

argo_dataset = open_dataset_safely(ARGO_FILE)
print(argo_dataset)

required_dimensions = ["N_PROF", "N_LEVELS"]
missing_dimensions = [
    dimension
    for dimension in required_dimensions
    if dimension not in argo_dataset.dims
]

if missing_dimensions:
    raise ValueError(
        "The input file does not have the required dimensions: "
        + ", ".join(missing_dimensions)
    )

profile_summary = build_profile_summary(argo_dataset)

# Close the large NetCDF before plotting and aggregation.
argo_dataset.close()
del argo_dataset
gc.collect()

seasonal_counts = create_seasonal_profile_count_maps(profile_summary)
sea_season_counts = create_sea_season_count_table(profile_summary)
depth_thresholds = create_depth_threshold_table(profile_summary)

print("\nSea × season profile counts:")
print(sea_season_counts)

print("\nDepth-threshold percentages:")
print(depth_thresholds)

# Save supporting outputs before plotting.
sea_season_counts.to_csv(OUTPUT_HEATMAP_CSV)
depth_thresholds.to_csv(OUTPUT_DEPTH_CSV)

seasonal_counts.to_dataset().to_netcdf(
    OUTPUT_SEASONAL_NC,
    encoding={
        "profile_count": {
            "zlib": True,
            "complevel": 4,
            "dtype": "int32",
        }
    },
)

if SAVE_PROFILE_LEVEL_SUMMARY:
    profile_summary.to_csv(
        OUTPUT_PROFILE_SUMMARY,
        index=False,
        compression="gzip",
    )




# DATA PRODUCT GROUP 12: SUPPLEMENTARY FIGURE S3 ARGO–EN4 VALIDATION DATA PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: Figure_S3_Argo_EN4_0_1000_validation_matched_profiles.csv
# File name: Figure_S3_Argo_EN4_0_1000_validation_sea_bias.csv
# File name: Figure_S3_Argo_EN4_0_1000_validation_seasonal_correlations.csv
# File name: Figure_S3_Argo_EN4_0_1000_validation_matched_counts.csv

# FIGURE S3 — ARGO–EN4 0–1000 m VALIDATION
# HDF-ERROR FIXED • LOW-RAM • CHECKPOINTED • RESUMABLE • 1080-DPI
#
# Panels
# ------
# (a) Argo versus EN4 total steric height
# (b) Argo versus EN4 thermosteric height
# (c) Argo versus EN4 halosteric height
# (d) Sea-wise mean bias: Argo − EN4
# (e) Seasonal Pearson correlation
# (f) Number of matched Argo profiles
#
# Main corrections compared with the failed workflow
# ---------------------------------------------------
# 1. Large NetCDF inputs are copied from Google Drive to /content first.
#    This avoids intermittent Google-Drive HDF5 read errors.
# 2. h5netcdf is tried before netCDF4.
# 3. The Argo TEMP/PSAL arrays are never loaded in full. Profiles are read
#    in contiguous chunks.
# 4. The EN4 all-month reference profile is calculated once and cached.
# 5. Completed Argo chunks are cached as CSV files and reused after a crash.
# 6. The final matched-profile CSV is reused unless FORCE_RECOMPUTE=True.
# 7. The scientific calculations use the same fixed 2008–2025 grid-cell
#    reference profile and TEOS-10 decomposition used for the EN4 products.
#
# Outputs are written to:
# /content/drive/MyDrive/SAM_Thesis/paper2


# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "gsw": "gsw",
    "h5netcdf": "h5netcdf",
    "netCDF4": "netCDF4",
    "dask": "dask[array]",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )

print("All required packages are available.")


# 1. IMPORTS
import os
import gc
import time
import shutil
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import dask
import gsw

from scipy import stats
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)



# 3. USER SETTINGS
FORCE_RECOMPUTE = False
REUSE_REFERENCE_CACHE = True
REUSE_CHUNK_CACHE = True
STAGE_INPUTS_TO_LOCAL = True

# Chunk size controls RAM usage. 1000–2500 is safe for standard Colab RAM.
ARGO_PROFILE_CHUNK = 1500

# Plot settings
SAVE_DPI = 1080
FIGSIZE = (9.6, 6.9)
ADD_OVERALL_TITLE = False
MAX_SCATTER_POINTS_DISPLAY = 18000
RANDOM_SEED = 42

# Validation period and layer
START_DATE = pd.Timestamp("2008-01-01")
END_DATE = pd.Timestamp("2025-12-31 23:59:59")
MAX_DEPTH_M = 1000.0
MAX_VERTICAL_GAP_M = 100.0
MAX_NEAREST_CELL_DISTANCE_KM = 120.0

# A profile must have common T/S coverage at the top and down to 1000 m.
MAX_ALLOWED_TOP_DEPTH_M = 15.0
MIN_ALLOWED_BOTTOM_DEPTH_M = 1000.0

# Scientific range guard. Values outside these broad limits are rejected.
TEMP_RANGE_C = (-3.5, 15.0)
SAL_RANGE = (20.0, 40.0)

VERBOSE = True


# 4. PATHS — OUTPUT NAMES REMAIN FIXED
BASE_DIR = Path("/content/drive/MyDrive/SAM_Thesis")
DATA_DIR = BASE_DIR / "Data"
EN4_PRODUCT_DIR = BASE_DIR / "Processed" / "EN4_NetCDF_inventory"
OUTPUT_DIR = BASE_DIR / "paper2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARGO_CANDIDATES = [
    DATA_DIR / "argo_SO_profiles_2001_2025_cleaned_gridded.nc",
    DATA_DIR / "argo_SO_profiles_2001_2025_cleaned.nc",
]

EN4_RAW_CANDIDATES = [
    DATA_DIR / "EN4" / "final_monthly" / "EN4.2.2_g10_Antarctic_monthly_2008_2025.nc",
    DATA_DIR / "EN4.2.2_g10_Antarctic_monthly_2008_2025.nc",
]

EN4_TOTAL_FILE = EN4_PRODUCT_DIR / "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
EN4_THERMO_FILE = EN4_PRODUCT_DIR / "EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc"
EN4_HALO_FILE = EN4_PRODUCT_DIR / "EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc"

MATCHED_CSV = OUTPUT_DIR / "Figure_S3_Argo_EN4_0_1000_validation_matched_profiles.csv"
SEA_BIAS_CSV = OUTPUT_DIR / "Figure_S3_Argo_EN4_0_1000_validation_sea_bias.csv"
SEASONAL_CORR_CSV = OUTPUT_DIR / "Figure_S3_Argo_EN4_0_1000_validation_seasonal_correlations.csv"
SEA_COUNT_CSV = OUTPUT_DIR / "Figure_S3_Argo_EN4_0_1000_validation_matched_counts.csv"
PROCESSING_LOG = OUTPUT_DIR / "Figure_S3_Argo_EN4_0_1000_validation_processing_log.txt"

REFERENCE_CACHE = OUTPUT_DIR / "Figure_S3_EN4_fixed_reference_0_1000m.npz"
CHUNK_CACHE_DIR = OUTPUT_DIR / "Figure_S3_matching_chunks_v2"
CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_STAGE_DIR = Path("/content/Figure_S3_input_cache")
LOCAL_STAGE_DIR.mkdir(parents=True, exist_ok=True)


# 5. ANTARCTIC SEA DEFINITIONS
SEA_SECTORS = OrderedDict([
    ("WED", {"name": "Weddell Sea",         "lon_min": -60.0,  "lon_max": -20.0}),
    ("KHV", {"name": "King Haakon VII Sea", "lon_min": -20.0,  "lon_max":  10.0}),
    ("RLS", {"name": "Riiser-Larsen Sea",   "lon_min":  10.0,  "lon_max":  35.0}),
    ("LAZ", {"name": "Lazarev Sea",         "lon_min":  35.0,  "lon_max":  60.0}),
    ("COS", {"name": "Cosmonauts Sea",      "lon_min":  60.0,  "lon_max":  90.0}),
    ("COO", {"name": "Cooperation Sea",     "lon_min":  90.0,  "lon_max": 115.0}),
    ("DAV", {"name": "Davis Sea",           "lon_min": 115.0,  "lon_max": 130.0}),
    ("MAW", {"name": "Mawson Sea",          "lon_min": 130.0,  "lon_max": 150.0}),
    ("DUR", {"name": "D'Urville Sea",       "lon_min": 150.0,  "lon_max": 170.0}),
    ("SOM", {"name": "Somov Sea",           "lon_min": 170.0,  "lon_max": -160.0}),
    ("ROS", {"name": "Ross Sea",            "lon_min": -160.0, "lon_max": -130.0}),
    ("AMU", {"name": "Amundsen Sea",        "lon_min": -130.0, "lon_max": -100.0}),
    ("BEL", {"name": "Bellingshausen Sea",  "lon_min": -100.0, "lon_max": -60.0}),
])

SEA_ORDER = list(SEA_SECTORS.keys())
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


# ============================================================================
# 6. GENERAL HELPERS
# ============================================================================
def log(message=""):
    if VERBOSE:
        print(message, flush=True)


def first_existing(candidates, label):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {label} file was found. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found:\n{path}")
    return path


def normalize_longitude(values):
    values = np.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def assign_sea(longitude):
    longitude = float(normalize_longitude([longitude])[0])
    for code, information in SEA_SECTORS.items():
        west = information["lon_min"]
        east = information["lon_max"]
        if west <= east:
            inside = west <= longitude < east
        else:
            inside = longitude >= west or longitude < east
        if inside:
            return code
    return None


def month_to_season(month):
    month = int(month)
    if month in (9, 10, 11):
        return "Spring"
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    return "Winter"


def open_dataset_robust(
    path,
    *,
    decode_times=True,
    chunks=None,
    drop_variables=None,
):
    """Open a NetCDF with multiple engines; h5netcdf is attempted first."""
    path = Path(path)
    attempts = []

    for engine in ("h5netcdf", "netcdf4", None, "scipy"):
        try:
            kwargs = {
                "decode_times": decode_times,
                "mask_and_scale": True,
                "cache": False,
            }
            if engine is not None:
                kwargs["engine"] = engine
            if chunks is not None:
                kwargs["chunks"] = chunks
            if drop_variables is not None:
                kwargs["drop_variables"] = drop_variables

            dataset = xr.open_dataset(path, **kwargs)
            log(f"Opened {path.name} with engine={engine or 'xarray-default'}")
            return dataset
        except Exception as error:
            attempts.append(f"{engine or 'default'}: {type(error).__name__}: {error}")
            gc.collect()

    raise RuntimeError(
        f"Could not open NetCDF file:\n{path}\n\nAttempts:\n"
        + "\n".join(attempts)
    )


def stage_file_to_local(source_path, label, retries=3):
    """Copy a Drive file to /content and verify byte size."""
    source_path = Path(source_path)

    if not STAGE_INPUTS_TO_LOCAL:
        return source_path

    destination = LOCAL_STAGE_DIR / source_path.name
    source_size = source_path.stat().st_size

    if destination.exists() and destination.stat().st_size == source_size:
        log(f"Using staged local {label}: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".partial")

    for attempt in range(1, retries + 1):
        try:
            log(
                f"Staging {label} to local runtime "
                f"({source_size / 1024**2:.1f} MB), attempt {attempt}/{retries} ..."
            )
            if temporary.exists():
                temporary.unlink()

            with source_path.open("rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())

            if temporary.stat().st_size != source_size:
                raise IOError(
                    f"Staged file size mismatch: "
                    f"{temporary.stat().st_size} != {source_size}"
                )

            temporary.replace(destination)
            log(f"Staged local file: {destination}")
            return destination

        except Exception as error:
            log(f"  staging attempt failed: {error}")
            time.sleep(3 * attempt)

    raise RuntimeError(
        f"Could not stage {label} after {retries} attempts:\n{source_path}"
    )


def standardize_3d_product(dataset, variable_name):
    rename = {}
    for source, target in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("valid_time", "time"),
    ):
        if source in dataset.coords or source in dataset.dims:
            if target not in dataset.coords and target not in dataset.dims:
                rename[source] = target
    if rename:
        dataset = dataset.rename(rename)

    data = dataset[variable_name]
    data = data.assign_coords(lon=normalize_longitude(data["lon"].values))
    _, unique_indices = np.unique(data["lon"].values, return_index=True)
    data = data.isel(lon=np.sort(unique_indices)).sortby("lon").sortby("lat")
    return data.transpose("time", "lat", "lon")


def decode_argo_juld(juld):
    values = np.asarray(juld.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)

    units = str(juld.attrs.get("units", "")).lower()
    candidate_origins = []

    if "since" in units:
        try:
            origin_text = units.split("since", 1)[1].strip()
            candidate_origins.append(pd.Timestamp(origin_text))
        except Exception:
            pass

    candidate_origins.extend([
        pd.Timestamp("1950-01-01"),
        pd.Timestamp("1970-01-01"),
    ])

    best_dates = None
    best_score = -1

    for origin in candidate_origins:
        dates = origin + pd.to_timedelta(values.astype(float), unit="D")
        valid = pd.DatetimeIndex(dates)
        score = int(((valid.year >= 1990) & (valid.year <= 2035)).sum())
        if score > best_score:
            best_score = score
            best_dates = valid

    if best_dates is None:
        raise ValueError("Could not decode Argo JULD values.")

    return pd.DatetimeIndex(best_dates)


def nearest_regular_index(grid, values):
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    positions = np.searchsorted(grid, values)
    positions = np.clip(positions, 1, len(grid) - 1)
    left = positions - 1
    right = positions
    choose_right = np.abs(values - grid[right]) < np.abs(values - grid[left])
    return np.where(choose_right, right, left).astype(int)


def nearest_circular_longitude_index(grid, values):
    grid = normalize_longitude(grid)
    values = normalize_longitude(values)

    positions = np.searchsorted(grid, values)
    candidate_a = np.mod(positions - 1, len(grid))
    candidate_b = np.mod(positions, len(grid))

    distance_a = np.abs(
        ((values - grid[candidate_a] + 180.0) % 360.0) - 180.0
    )
    distance_b = np.abs(
        ((values - grid[candidate_b] + 180.0) % 360.0) - 180.0
    )

    return np.where(distance_b < distance_a, candidate_b, candidate_a).astype(int)


def haversine_km(lon1, lat1, lon2, lat2):
    radius_km = 6371.0
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def target_depth_grid(native_depth):
    native_depth = np.asarray(native_depth, dtype=float)
    inside = native_depth[(native_depth > 0.0) & (native_depth < MAX_DEPTH_M)]
    mandatory = np.array([0.0, MAX_DEPTH_M], dtype=float)
    depth = np.unique(np.round(np.concatenate([inside, mandatory]), 6))
    depth.sort()
    return depth


def interpolate_depth_axis(values, native_depth, target_depth):
    function = interp1d(
        native_depth,
        values,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    output = function(target_depth)
    # Match the EN4 product construction: represent 0 m with first EN4 level.
    zero_index = int(np.argmin(np.abs(target_depth - 0.0)))
    output[zero_index] = values[0]
    return output


def trapezoid(values, x):
    function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return function(values, x=x)


def integrate_steric(delta_specific_volume, pressure_dbar, gravity):
    return trapezoid(
        delta_specific_volume * 1.0e4 / gravity,
        pressure_dbar,
    )


def scatter_statistics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return {
            "r": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "n": len(x),
        }

    regression = stats.linregress(x, y)
    return {
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(regression.slope),
        "intercept": float(regression.intercept),
        "bias": float(np.mean(y - x)),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "n": int(len(x)),
    }


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def nice_symmetric_limit(values, minimum=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return minimum
    maximum = max(float(np.nanpercentile(np.abs(values), 99.5)), minimum)
    steps = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50], dtype=float)
    larger = steps[steps >= maximum]
    return float(larger[0] if larger.size else np.ceil(maximum / 10.0) * 10.0)


# 7. INPUT DISCOVERY AND LOCAL STAGING
ARGO_SOURCE = first_existing(ARGO_CANDIDATES, "Argo profile")
EN4_RAW_SOURCE = first_existing(EN4_RAW_CANDIDATES, "raw EN4")
require_file(EN4_TOTAL_FILE, "EN4 total steric")
require_file(EN4_THERMO_FILE, "EN4 thermosteric")
require_file(EN4_HALO_FILE, "EN4 halosteric")

log("=" * 88)
log("FIGURE S3 — ARGO–EN4 0–1000 m VALIDATION")
log("=" * 88)
log(f"Argo source      : {ARGO_SOURCE}")
log(f"EN4 raw source   : {EN4_RAW_SOURCE}")
log(f"EN4 total product: {EN4_TOTAL_FILE}")
log(f"EN4 thermo       : {EN4_THERMO_FILE}")
log(f"EN4 halo         : {EN4_HALO_FILE}")

ARGO_FILE = stage_file_to_local(ARGO_SOURCE, "Argo file")
EN4_RAW_FILE = stage_file_to_local(EN4_RAW_SOURCE, "raw EN4 file")
EN4_TOTAL_LOCAL = stage_file_to_local(EN4_TOTAL_FILE, "EN4 total product")
EN4_THERMO_LOCAL = stage_file_to_local(EN4_THERMO_FILE, "EN4 thermosteric product")
EN4_HALO_LOCAL = stage_file_to_local(EN4_HALO_FILE, "EN4 halosteric product")


# 8. READ EN4 MONTHLY PRODUCTS
def load_en4_product(path, variable_name):
    dataset = open_dataset_robust(path, decode_times=True)
    try:
        data = standardize_3d_product(dataset, variable_name).load()
        units = str(data.attrs.get("units", "")).lower()
        if units in {"m", "metre", "meter", "metres", "meters"}:
            data = data * 100.0
        elif "cm" not in units:
            # The verified products are metres; this fallback preserves that behavior.
            data = data * 100.0
        data.attrs["units"] = "cm"
        return data
    finally:
        dataset.close()
        gc.collect()


log("\nLoading compact EN4 monthly steric products ...")
en4_total = load_en4_product(EN4_TOTAL_LOCAL, "total_steric_height")
en4_thermo = load_en4_product(EN4_THERMO_LOCAL, "thermosteric_height")
en4_halo = load_en4_product(EN4_HALO_LOCAL, "halosteric_height")

# Ensure exact common coordinates.
en4_thermo = en4_thermo.sel(time=en4_total.time, lat=en4_total.lat, lon=en4_total.lon)
en4_halo = en4_halo.sel(time=en4_total.time, lat=en4_total.lat, lon=en4_total.lon)

en4_times = pd.DatetimeIndex(pd.to_datetime(en4_total.time.values))
en4_latitudes = en4_total.lat.values.astype(float)
en4_longitudes = en4_total.lon.values.astype(float)
month_to_time_index = {
    pd.Timestamp(date).to_period("M"): index
    for index, date in enumerate(en4_times)
}

log(
    f"EN4 product grid: time={len(en4_times)}, "
    f"lat={len(en4_latitudes)}, lon={len(en4_longitudes)}"
)


# 9. BUILD OR LOAD FIXED EN4 REFERENCE PROFILES
def build_reference_cache():
    source_size = EN4_RAW_SOURCE.stat().st_size

    if REFERENCE_CACHE.exists() and REUSE_REFERENCE_CACHE and not FORCE_RECOMPUTE:
        cached = np.load(REFERENCE_CACHE, allow_pickle=False)
        cached_size = int(cached["source_size_bytes"])
        if cached_size == source_size:
            log(f"\nLoading cached EN4 reference: {REFERENCE_CACHE.name}")
            return {
                "depth": cached["depth"].astype(float),
                "temperature_c": cached["temperature_c"].astype(np.float32),
                "salinity": cached["salinity"].astype(np.float32),
                "lat": cached["lat"].astype(float),
                "lon": cached["lon"].astype(float),
            }
        log("Reference cache source-size mismatch; recalculating.")

    log("\nCalculating fixed 2008–2025 EN4 reference profiles with low RAM ...")

    dataset = open_dataset_robust(
        EN4_RAW_FILE,
        decode_times=False,
        chunks={"time": 12},
    )

    try:
        rename = {}
        for source, target in (
            ("latitude", "lat"),
            ("longitude", "lon"),
        ):
            if source in dataset.coords or source in dataset.dims:
                if target not in dataset.coords and target not in dataset.dims:
                    rename[source] = target
        if rename:
            dataset = dataset.rename(rename)

        native_depth = dataset["depth"].values.astype(float)
        analysis_depth = target_depth_grid(native_depth)

        with dask.config.set(scheduler="single-threaded"):
            temperature_native = (
                dataset["temperature"]
                .mean("time", skipna=True)
                .compute()
                .values
                .astype(np.float32)
            )
            salinity_native = (
                dataset["salinity"]
                .mean("time", skipna=True)
                .compute()
                .values
                .astype(np.float32)
            )

        # EN4 temperature is potential temperature in Kelvin.
        if np.nanmedian(temperature_native) > 100.0:
            temperature_native = temperature_native - 273.15

        temperature_reference = interpolate_depth_axis(
            temperature_native,
            native_depth,
            analysis_depth,
        ).astype(np.float32)

        salinity_reference = interpolate_depth_axis(
            salinity_native,
            native_depth,
            analysis_depth,
        ).astype(np.float32)

        latitudes = dataset["lat"].values.astype(float)
        longitudes = normalize_longitude(dataset["lon"].values.astype(float))

        # Sort longitude and remove duplicates if necessary.
        sort_indices = np.argsort(longitudes)
        longitudes = longitudes[sort_indices]
        temperature_reference = temperature_reference[:, :, sort_indices]
        salinity_reference = salinity_reference[:, :, sort_indices]
        unique_longitudes, unique_indices = np.unique(longitudes, return_index=True)
        longitudes = unique_longitudes
        temperature_reference = temperature_reference[:, :, unique_indices]
        salinity_reference = salinity_reference[:, :, unique_indices]

        np.savez_compressed(
            REFERENCE_CACHE,
            depth=analysis_depth.astype(np.float32),
            temperature_c=temperature_reference,
            salinity=salinity_reference,
            lat=latitudes.astype(np.float32),
            lon=longitudes.astype(np.float32),
            source_size_bytes=np.array(source_size, dtype=np.int64),
        )

        log(f"Saved EN4 reference cache: {REFERENCE_CACHE}")

        return {
            "depth": analysis_depth,
            "temperature_c": temperature_reference,
            "salinity": salinity_reference,
            "lat": latitudes,
            "lon": longitudes,
        }

    finally:
        dataset.close()
        gc.collect()


reference = build_reference_cache()
analysis_depth = reference["depth"]
reference_temperature = reference["temperature_c"]
reference_salinity = reference["salinity"]

# Verify reference and product grids agree.
if not np.allclose(reference["lat"], en4_latitudes, atol=1e-5):
    raise ValueError("Raw EN4 and derived EN4 latitude grids do not agree.")
if not np.allclose(reference["lon"], en4_longitudes, atol=1e-5):
    raise ValueError("Raw EN4 and derived EN4 longitude grids do not agree.")

log(f"Reference depth levels: {len(analysis_depth)}; {analysis_depth[0]:.1f}–{analysis_depth[-1]:.1f} m")


# ============================================================================
# 10. ARGO CHUNK PROCESSING
# ============================================================================
def process_argo_chunks():
    if MATCHED_CSV.exists() and not FORCE_RECOMPUTE:
        log(f"\nLoading completed matched table: {MATCHED_CSV.name}")
        return pd.read_csv(MATCHED_CSV, parse_dates=["time"])

    dataset = open_dataset_robust(ARGO_FILE, decode_times=False)

    try:
        required_variables = [
            "PRES_GRID", "JULD", "LATITUDE", "LONGITUDE", "TEMP", "PSAL"
        ]
        missing = [name for name in required_variables if name not in dataset.variables]
        if missing:
            raise KeyError(
                f"Argo file is missing variables {missing}. "
                f"Available variables: {list(dataset.variables)}"
            )

        pressure_grid = dataset["PRES_GRID"].values.astype(float)
        if np.any(np.diff(pressure_grid) <= 0):
            raise ValueError("PRES_GRID must be strictly increasing.")

        argo_times = decode_argo_juld(dataset["JULD"])
        latitudes_all = dataset["LATITUDE"].values.astype(float)
        longitudes_all = normalize_longitude(dataset["LONGITUDE"].values.astype(float))

        n_profiles = int(dataset.sizes["N_PROF"])
        log(f"\nArgo profiles in source: {n_profiles:,}")
        log(f"Processing chunk size: {ARGO_PROFILE_CHUNK:,}")

        all_chunk_files = []

        for start in range(0, n_profiles, ARGO_PROFILE_CHUNK):
            stop = min(start + ARGO_PROFILE_CHUNK, n_profiles)
            chunk_file = CHUNK_CACHE_DIR / f"matched_{start:06d}_{stop:06d}.csv"
            all_chunk_files.append(chunk_file)

            if chunk_file.exists() and REUSE_CHUNK_CACHE and not FORCE_RECOMPUTE:
                log(f"Reusing chunk {start:06d}:{stop:06d}")
                continue

            chunk_times = argo_times[start:stop]
            chunk_latitudes = latitudes_all[start:stop]
            chunk_longitudes = longitudes_all[start:stop]

            preliminary = (
                (chunk_times >= START_DATE)
                & (chunk_times <= END_DATE)
                & np.isfinite(chunk_latitudes)
                & np.isfinite(chunk_longitudes)
                & (chunk_latitudes >= -90.0)
                & (chunk_latitudes <= -50.0)
            )

            if preliminary.sum() == 0:
                pd.DataFrame().to_csv(chunk_file, index=False)
                log(f"[{stop:>6}/{n_profiles}] no profiles in analysis period/domain")
                continue

            # Read only this contiguous block from the large HDF5 arrays.
            temperature_chunk = (
                dataset["TEMP"]
                .isel(N_PROF=slice(start, stop))
                .load()
                .values
                .astype(np.float32)
            )
            salinity_chunk = (
                dataset["PSAL"]
                .isel(N_PROF=slice(start, stop))
                .load()
                .values
                .astype(np.float32)
            )

            if np.nanmedian(temperature_chunk) > 100.0:
                temperature_chunk = temperature_chunk - 273.15

            rows = []
            local_indices = np.where(preliminary)[0]

            for local_index in local_indices:
                profile_time = pd.Timestamp(chunk_times[local_index])
                profile_latitude = float(chunk_latitudes[local_index])
                profile_longitude = float(chunk_longitudes[local_index])
                sea = assign_sea(profile_longitude)
                if sea is None:
                    continue

                month_period = profile_time.to_period("M")
                if month_period not in month_to_time_index:
                    continue
                time_index = month_to_time_index[month_period]

                lat_index = int(nearest_regular_index(en4_latitudes, [profile_latitude])[0])
                lon_index = int(nearest_circular_longitude_index(en4_longitudes, [profile_longitude])[0])

                en4_latitude = float(en4_latitudes[lat_index])
                en4_longitude = float(en4_longitudes[lon_index])
                match_distance = float(
                    haversine_km(
                        profile_longitude,
                        profile_latitude,
                        en4_longitude,
                        en4_latitude,
                    )
                )
                if match_distance > MAX_NEAREST_CELL_DISTANCE_KM:
                    continue

                en4_total_value = float(en4_total.values[time_index, lat_index, lon_index])
                en4_thermo_value = float(en4_thermo.values[time_index, lat_index, lon_index])
                en4_halo_value = float(en4_halo.values[time_index, lat_index, lon_index])

                if not np.all(np.isfinite([
                    en4_total_value,
                    en4_thermo_value,
                    en4_halo_value,
                ])):
                    continue

                temperature_native = temperature_chunk[local_index].astype(float)
                salinity_native = salinity_chunk[local_index].astype(float)

                common = (
                    np.isfinite(pressure_grid)
                    & np.isfinite(temperature_native)
                    & np.isfinite(salinity_native)
                    & (temperature_native >= TEMP_RANGE_C[0])
                    & (temperature_native <= TEMP_RANGE_C[1])
                    & (salinity_native >= SAL_RANGE[0])
                    & (salinity_native <= SAL_RANGE[1])
                )

                if common.sum() < 4:
                    continue

                pressure_valid = pressure_grid[common]
                temperature_valid = temperature_native[common]
                salinity_valid = salinity_native[common]

                # Convert pressure to positive depth at the profile latitude.
                depth_valid = -gsw.z_from_p(pressure_valid, profile_latitude)
                ordering = np.argsort(depth_valid)
                depth_valid = depth_valid[ordering]
                temperature_valid = temperature_valid[ordering]
                salinity_valid = salinity_valid[ordering]

                unique_depth, unique_indices = np.unique(depth_valid, return_index=True)
                depth_valid = unique_depth
                temperature_valid = temperature_valid[unique_indices]
                salinity_valid = salinity_valid[unique_indices]

                if depth_valid[0] > MAX_ALLOWED_TOP_DEPTH_M:
                    continue
                if depth_valid[-1] < MIN_ALLOWED_BOTTOM_DEPTH_M:
                    continue
                if np.nanmax(np.diff(depth_valid)) > MAX_VERTICAL_GAP_M:
                    continue

                temperature_function = interp1d(
                    depth_valid,
                    temperature_valid,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                    assume_sorted=True,
                )
                salinity_function = interp1d(
                    depth_valid,
                    salinity_valid,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                    assume_sorted=True,
                )

                profile_temperature = temperature_function(analysis_depth)
                profile_salinity = salinity_function(analysis_depth)

                # Match the EN4 construction at 0 m: use the shallowest valid value.
                profile_temperature[0] = temperature_valid[0]
                profile_salinity[0] = salinity_valid[0]

                if not (
                    np.isfinite(profile_temperature).all()
                    and np.isfinite(profile_salinity).all()
                ):
                    continue

                reference_temperature_profile = reference_temperature[:, lat_index, lon_index].astype(float)
                reference_salinity_profile = reference_salinity[:, lat_index, lon_index].astype(float)
                if not (
                    np.isfinite(reference_temperature_profile).all()
                    and np.isfinite(reference_salinity_profile).all()
                ):
                    continue

                # Use the matched EN4 grid-cell pressure/gravity to reproduce
                # the EN4 product integration as directly as possible.
                pressure_target = gsw.p_from_z(-analysis_depth, en4_latitude)
                gravity_target = gsw.grav(en4_latitude, pressure_target)

                absolute_salinity_reference = gsw.SA_from_SP(
                    reference_salinity_profile,
                    pressure_target,
                    en4_longitude,
                    en4_latitude,
                )
                conservative_temperature_reference = gsw.CT_from_pt(
                    absolute_salinity_reference,
                    reference_temperature_profile,
                )
                specific_volume_reference = gsw.specvol(
                    absolute_salinity_reference,
                    conservative_temperature_reference,
                    pressure_target,
                )

                absolute_salinity_profile = gsw.SA_from_SP(
                    profile_salinity,
                    pressure_target,
                    en4_longitude,
                    en4_latitude,
                )
                conservative_temperature_profile = gsw.CT_from_t(
                    absolute_salinity_profile,
                    profile_temperature,
                    pressure_target,
                )
                potential_temperature_profile = gsw.pt0_from_t(
                    absolute_salinity_profile,
                    profile_temperature,
                    pressure_target,
                )
                specific_volume_profile = gsw.specvol(
                    absolute_salinity_profile,
                    conservative_temperature_profile,
                    pressure_target,
                )

                argo_total_cm = 100.0 * integrate_steric(
                    specific_volume_profile - specific_volume_reference,
                    pressure_target,
                    gravity_target,
                )

                conservative_temperature_thermo = gsw.CT_from_pt(
                    absolute_salinity_reference,
                    potential_temperature_profile,
                )
                specific_volume_thermo = gsw.specvol(
                    absolute_salinity_reference,
                    conservative_temperature_thermo,
                    pressure_target,
                )

                argo_thermo_cm = 100.0 * integrate_steric(
                    specific_volume_thermo - specific_volume_reference,
                    pressure_target,
                    gravity_target,
                )
                argo_halo_cm = argo_total_cm - argo_thermo_cm

                if not np.all(np.isfinite([
                    argo_total_cm,
                    argo_thermo_cm,
                    argo_halo_cm,
                ])):
                    continue

                rows.append({
                    "profile_index": int(start + local_index),
                    "time": profile_time,
                    "season": month_to_season(profile_time.month),
                    "sea": sea,
                    "latitude": profile_latitude,
                    "longitude": profile_longitude,
                    "en4_latitude": en4_latitude,
                    "en4_longitude": en4_longitude,
                    "match_distance_km": match_distance,
                    "argo_total_cm": float(argo_total_cm),
                    "en4_total_cm": en4_total_value,
                    "argo_thermo_cm": float(argo_thermo_cm),
                    "en4_thermo_cm": en4_thermo_value,
                    "argo_halo_cm": float(argo_halo_cm),
                    "en4_halo_cm": en4_halo_value,
                })

            pd.DataFrame(rows).to_csv(chunk_file, index=False)
            log(
                f"[{stop:>6}/{n_profiles}] saved {len(rows):>4} matches: "
                f"{chunk_file.name}"
            )

            del temperature_chunk, salinity_chunk
            gc.collect()

        log("\nCombining cached matching chunks ...")
        tables = []
        for chunk_file in all_chunk_files:
            if not chunk_file.exists() or chunk_file.stat().st_size == 0:
                continue
            try:
                table = pd.read_csv(chunk_file, parse_dates=["time"])
            except pd.errors.EmptyDataError:
                continue
            if not table.empty:
                tables.append(table)

        if not tables:
            raise RuntimeError(
                "No valid Argo–EN4 profile matches were produced. "
                "Review the pressure coverage and variable ranges."
            )

        matched_table = pd.concat(tables, ignore_index=True)
        matched_table = matched_table.drop_duplicates("profile_index").sort_values("time")
        matched_table.to_csv(MATCHED_CSV, index=False)
        log(f"Saved completed matched table: {MATCHED_CSV}")
        log(f"Matched profiles: {len(matched_table):,}")
        return matched_table

    finally:
        dataset.close()
        gc.collect()


matched = process_argo_chunks()

# Free the reference and monthly product arrays before high-DPI plotting.
del reference_temperature, reference_salinity, reference
try:
    del en4_total, en4_thermo, en4_halo
except Exception:
    pass
gc.collect()


# 11. SUMMARY TABLES
component_definitions = OrderedDict([
    ("total", {
        "argo": "argo_total_cm",
        "en4": "en4_total_cm",
        "label": "Total steric",
        "color": "#1665C1",
    }),
    ("thermo", {
        "argo": "argo_thermo_cm",
        "en4": "en4_thermo_cm",
        "label": "Thermosteric",
        "color": "#229A22",
    }),
    ("halo", {
        "argo": "argo_halo_cm",
        "en4": "en4_halo_cm",
        "label": "Halosteric",
        "color": "#9227A8",
    }),
])

scatter_summary = {
    key: scatter_statistics(matched[info["en4"]], matched[info["argo"]])
    for key, info in component_definitions.items()
}

sea_bias_rows = []
for sea in SEA_ORDER:
    subset = matched[matched["sea"] == sea]
    row = {"sea": sea, "matched_profiles": int(len(subset))}
    for key, info in component_definitions.items():
        row[f"bias_{key}_cm"] = (
            float(np.mean(subset[info["argo"]] - subset[info["en4"]]))
            if len(subset) else np.nan
        )
    sea_bias_rows.append(row)

sea_bias = pd.DataFrame(sea_bias_rows)
sea_bias.to_csv(SEA_BIAS_CSV, index=False)

season_rows = []
for season in SEASON_ORDER:
    subset = matched[matched["season"] == season]
    row = {"season": season, "matched_profiles": int(len(subset))}
    for key, info in component_definitions.items():
        row[f"r_{key}"] = correlation(subset[info["en4"]], subset[info["argo"]])
    season_rows.append(row)

seasonal_correlation = pd.DataFrame(season_rows)
seasonal_correlation.to_csv(SEASONAL_CORR_CSV, index=False)

sea_counts = (
    matched.groupby("sea")
    .size()
    .reindex(SEA_ORDER, fill_value=0)
    .astype(int)
    .rename("matched_profiles")
    .reset_index()
)
sea_counts.to_csv(SEA_COUNT_CSV, index=False)




# DATA PRODUCT GROUP 13: SUPPLEMENTARY FIGURE S4 EN4 DEPTH-SENSITIVITY DATA PRODUCTS
# Source: so_sealevel_paper_fig.py
# File name: EN4_total_steric_0_2000m_monthly_2008_2025_SO.nc
# File name: EN4_thermosteric_0_2000m_monthly_2008_2025_SO.nc
# File name: EN4_halosteric_0_2000m_monthly_2008_2025_SO.nc
# File name: Figure_S4_EN4_depth_sensitivity_fields.nc
# File name: Figure_S4_EN4_depth_sensitivity_summary.csv

# FIGURE S4 — EN4 INTEGRATION-DEPTH SENSITIVITY
# Southern Ocean, 2008–2025
#
# Structure
# ---------
#                     0–1000 m       0–2000 m       Difference
# Total steric            (a)             (b)             (c)
# Thermosteric            (d)             (e)             (f)
# Halosteric              (g)             (h)             (i)
#
# Spatial rule
# ------------
# Every map uses the SAME fixed spatial mask:
#   • GEBCO water depth >= 2000 m
#   • finite 0–1000 m total / thermo / halo values
#   • finite 0–2000 m total / thermo / halo values
#   • required valid-data fraction through the study period
#
# The 0–1000 m maps are therefore NOT allowed to show cells that are
# unavailable in the 0–2000 m products.
#
# Difference
# ----------
# Difference = 0–2000 m field − 0–1000 m field
#
# Default displayed diagnostic
# ----------------------------
# WINTER_MINUS_SUMMER:
#   Winter seasonal climatology − Summer seasonal climatology
#
# This signed seasonal contrast is used because Figure S4 contains only
# one map for each depth/component combination. It retains direction and
# is compatible with a diverging red–blue colour scale.
#
# Other supported diagnostics can be selected below:
#   "winter_minus_summer"
#   "spring_anomaly"
#   "summer_anomaly"
#   "autumn_anomaly"
#   "winter_anomaly"
#   "seasonal_peak_to_peak"
#   "seasonal_rms"
#
# 0–2000 m files
# --------------
# If the three 0–2000 m monthly products already exist, the script reuses
# them. Otherwise, it creates them month-by-month from the raw EN4 T–S
# file using the same fixed 2008–2025 grid-cell reference-state method
# used for the corrected 0–1000 m products.
#
# No overall figure title is added.


# 0. INSTALL REQUIRED PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "h5netcdf": "h5netcdf",
    "netCDF4": "netCDF4",
    "cftime": "cftime",
    "dask": "dask[array]",
    "scipy": "scipy",
    "gsw": "gsw",
    "matplotlib": "matplotlib",
    "cartopy": "cartopy",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
    print("Package installation completed.")
else:
    print("All required packages are already installed.")


# 1. IMPORTS
import os
import gc
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gsw

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# 3. INPUT AND OUTPUT PATHS
BASE_DIR = Path("/content/drive/MyDrive/SAM_Thesis")

RAW_EN4_FILE = (
    BASE_DIR
    / "Data/EN4/final_monthly/"
    / "EN4.2.2_g10_Antarctic_monthly_2008_2025.nc"
)

DERIVED_DIR = (
    BASE_DIR
    / "Processed/EN4_NetCDF_inventory"
)

OUTPUT_DIR = (
    BASE_DIR
    / "paper2"
)

DERIVED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing corrected 0–1000 m products
FILES_1000 = {
    "total": (
        DERIVED_DIR
        / "EN4_total_steric_0_1000m_monthly_2008_2025_SO.nc"
    ),
    "thermo": (
        DERIVED_DIR
        / "EN4_thermosteric_0_1000m_monthly_2008_2025_SO.nc"
    ),
    "halo": (
        DERIVED_DIR
        / "EN4_halosteric_0_1000m_monthly_2008_2025_SO.nc"
    ),
}

# 0–2000 m products created/reused by this script
FILES_2000 = {
    "total": (
        DERIVED_DIR
        / "EN4_total_steric_0_2000m_monthly_2008_2025_SO.nc"
    ),
    "thermo": (
        DERIVED_DIR
        / "EN4_thermosteric_0_2000m_monthly_2008_2025_SO.nc"
    ),
    "halo": (
        DERIVED_DIR
        / "EN4_halosteric_0_2000m_monthly_2008_2025_SO.nc"
    ),
}



OUT_FIELDS_NC = (
    OUTPUT_DIR
    / "Figure_S4_EN4_depth_sensitivity_fields.nc"
)

OUT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "Figure_S4_EN4_depth_sensitivity_summary.csv"
)

OUT_SUMMARY_TXT = (
    OUTPUT_DIR
    / "Figure_S4_EN4_depth_sensitivity_summary.txt"
)


# 4. ANALYSIS SETTINGS
YEAR_START = 2008
YEAR_END = 2025

LAT_MIN = -90.0
LAT_MAX = -60.0

DEPTH_1000 = 1000.0
DEPTH_2000 = 2000.0

# Reuse existing 0–2000 products when all three exist.
REUSE_EXISTING_2000_PRODUCTS = True

# Copy the approximately 2 GB raw EN4 file from Drive to local Colab
# storage before computing 0–2000 m products. This avoids many HDF5
# read failures caused by repeatedly reading a large file from Drive.
STAGE_RAW_EN4_TO_LOCAL = True
LOCAL_CACHE_DIR = Path("/content/Figure_S4_EN4_cache")

# Monthly support required for the fixed comparison mask.
# 1.00 = all 216 months must be valid in all six products.
COMMON_VALID_FRACTION = 1.00

# Default signed diagnostic displayed in the nine maps.
COMPARISON_MODE = "winter_minus_summer"

SUPPORTED_COMPARISON_MODES = {
    "winter_minus_summer",
    "spring_anomaly",
    "summer_anomaly",
    "autumn_anomaly",
    "winter_anomaly",
    "seasonal_peak_to_peak",
    "seasonal_rms",
}

if COMPARISON_MODE not in SUPPORTED_COMPARISON_MODES:
    raise ValueError(
        f"Unsupported COMPARISON_MODE={COMPARISON_MODE!r}. "
        f"Choose one of {sorted(SUPPORTED_COMPARISON_MODES)}"
    )

SEASON_ORDER = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

SEASON_MONTHS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Autumn": [3, 4, 5],
    "Winter": [6, 7, 8],
}

COMPONENT_ORDER = [
    "total",
    "thermo",
    "halo",
]

COMPONENT_LABELS = {
    "total": "Total steric",
    "thermo": "Thermosteric",
    "halo": "Halosteric",
}

COLUMN_LABELS = [
    "0–1000 m",
    "0–2000 m",
    "Difference",
]

PANEL_LETTERS = [
    ["(a)", "(b)", "(c)"],
    ["(d)", "(e)", "(f)"],
    ["(g)", "(h)", "(i)"],
]

DIFFERENCE_DEFINITION = "0–2000 m minus 0–1000 m"

MERIDIANS = np.arange(-180, 181, 30)
PARALLELS = [-60, -70, -80]

DRAW_2000_M_CONTOUR = True
BATHYMETRY_CONTOUR_LEVEL = 2000.0

SAVE_DPI = 1080


# 6. GENERIC HELPERS
def print_header(text):
    print("\n" + "=" * 92)
    print(text)
    print("=" * 92)


def safe_is_datetime(dtype):
    try:
        return np.issubdtype(dtype, np.datetime64)
    except TypeError:
        return False


def open_dataset_safely(
    file_path,
    chunks=None,
    decode_times=True,
):
    """
    Open a NetCDF file using several engines.
    h5netcdf is attempted first because it is generally reliable for
    large HDF5-backed EN4 files.
    """
    attempts = []

    for engine in [
        "h5netcdf",
        "netcdf4",
        None,
        "scipy",
    ]:
        for decode in (
            [decode_times]
            if decode_times is False
            else [True, False]
        ):
            try:
                kwargs = {
                    "decode_times": decode,
                    "mask_and_scale": True,
                }

                if engine is not None:
                    kwargs["engine"] = engine

                if chunks is not None:
                    kwargs["chunks"] = chunks

                dataset = xr.open_dataset(
                    file_path,
                    **kwargs,
                )

                engine_name = (
                    "xarray-default"
                    if engine is None
                    else engine
                )

                print(
                    f"Opened {Path(file_path).name} "
                    f"with engine={engine_name}, "
                    f"decode_times={decode}"
                )

                return dataset

            except Exception as error:
                attempts.append(
                    f"engine={engine}, decode_times={decode}: {error}"
                )

    raise RuntimeError(
        f"Could not open:\n{file_path}\n\n"
        + "\n".join(attempts)
    )


def stage_file_to_local(source_path, cache_directory):
    """
    Copy a large Drive file to local Colab storage when required.
    """
    source_path = Path(source_path)
    cache_directory.mkdir(parents=True, exist_ok=True)

    destination = cache_directory / source_path.name

    if destination.exists():
        source_size = source_path.stat().st_size
        destination_size = destination.stat().st_size

        if source_size == destination_size:
            print("Using existing local staged file:")
            print(destination)
            return destination

        destination.unlink()

    print("Copying large EN4 input from Drive to local runtime...")
    print("Source     :", source_path)
    print("Destination:", destination)

    shutil.copy2(source_path, destination)

    print("Local staging completed.")
    return destination


def detect_coordinate(dataset, coordinate_type):
    names = list(dataset.coords) + [
        name
        for name in dataset.variables
        if name not in dataset.coords
    ]

    aliases = {
        "time": ["time", "valid_time", "date", "datetime", "month", "t"],
        "depth": ["depth", "lev", "level", "pressure", "pres", "z"],
        "lat": ["lat", "latitude", "nav_lat", "y"],
        "lon": ["lon", "longitude", "nav_lon", "x"],
    }

    for name in names:
        variable = dataset[name]

        lower_name = name.lower()
        standard_name = str(
            variable.attrs.get("standard_name", "")
        ).lower()
        axis = str(
            variable.attrs.get("axis", "")
        ).upper()
        units = str(
            variable.attrs.get("units", "")
        ).lower()

        if coordinate_type == "time":
            if (
                lower_name in aliases["time"]
                or standard_name == "time"
                or axis == "T"
                or " since " in units
                or safe_is_datetime(variable.dtype)
            ):
                return name

        elif coordinate_type == "depth":
            if (
                lower_name in aliases["depth"]
                or standard_name in {
                    "depth",
                    "sea_water_pressure",
                }
                or axis == "Z"
                or units in {
                    "m",
                    "meter",
                    "metre",
                    "dbar",
                }
            ):
                return name

        elif coordinate_type == "lat":
            if (
                lower_name in aliases["lat"]
                or standard_name == "latitude"
                or axis == "Y"
                or "degree_north" in units
            ):
                return name

        elif coordinate_type == "lon":
            if (
                lower_name in aliases["lon"]
                or standard_name == "longitude"
                or axis == "X"
                or "degree_east" in units
            ):
                return name

    raise KeyError(
        f"Could not identify the {coordinate_type} coordinate. "
        f"Available coordinates/variables: {names}"
    )


def standardize_coordinates(
    dataset,
    time_name=None,
    depth_name=None,
    lat_name=None,
    lon_name=None,
):
    rename_mapping = {}

    if time_name is not None and time_name != "time":
        rename_mapping[time_name] = "time"

    if depth_name is not None and depth_name != "depth":
        rename_mapping[depth_name] = "depth"

    if lat_name is not None and lat_name != "lat":
        rename_mapping[lat_name] = "lat"

    if lon_name is not None and lon_name != "lon":
        rename_mapping[lon_name] = "lon"

    if rename_mapping:
        dataset = dataset.rename(rename_mapping)

    return dataset


def normalize_and_sort_longitude(dataset):
    if "lon" not in dataset.coords:
        return dataset

    if dataset["lon"].ndim != 1:
        raise ValueError("Only one-dimensional longitude is supported.")

    longitude = (
        (dataset["lon"].astype(float) + 180.0) % 360.0
    ) - 180.0

    dataset = dataset.assign_coords(lon=longitude)

    _, unique_indices = np.unique(
        dataset["lon"].values,
        return_index=True,
    )

    dataset = dataset.isel(
        lon=np.sort(unique_indices)
    )

    return dataset.sortby("lon")


def subset_antarctic_latitudes(
    dataset,
    southern_limit=LAT_MIN,
    northern_limit=LAT_MAX,
):
    latitude = dataset["lat"].values

    if latitude[0] <= latitude[-1]:
        output = dataset.sel(
            lat=slice(
                southern_limit,
                northern_limit,
            )
        )
    else:
        output = dataset.sel(
            lat=slice(
                northern_limit,
                southern_limit,
            )
        )

    return output.sortby("lat")


def choose_data_variable(
    dataset,
    preferred_names,
    excluded_names=None,
):
    if excluded_names is None:
        excluded_names = []

    variables = [
        name
        for name in dataset.data_vars
        if name not in excluded_names
    ]

    for preferred_name in preferred_names:
        for variable_name in variables:
            if preferred_name.lower() in variable_name.lower():
                return variable_name

    if not variables:
        raise ValueError(
            "No suitable science variable was found."
        )

    return variables[0]


def find_exact_or_partial_variable(dataset, names):
    for name in names:
        if name in dataset.data_vars:
            return name

    for name in names:
        for candidate in dataset.data_vars:
            if name.lower() in candidate.lower():
                return candidate

    raise KeyError(
        f"Could not find any of {names}. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def target_depth_grid(native_depth, maximum_depth):
    """
    Construct a depth grid containing:
      • every native EN4 level inside the target layer
      • an explicit 0 m upper boundary
      • an explicit maximum-depth lower boundary
    """
    native_depth = np.asarray(
        native_depth,
        dtype=float,
    )

    native_inside = native_depth[
        (native_depth > 0.0)
        & (native_depth < maximum_depth)
    ]

    mandatory = np.array(
        [0.0, maximum_depth],
        dtype=float,
    )

    target = np.unique(
        np.round(
            np.concatenate(
                [native_inside, mandatory]
            ),
            6,
        )
    )

    target.sort()
    return target


def interpolate_depth(
    values,
    native_depth,
    target_depth,
):
    """
    Linear interpolation along the first (depth) axis.
    The approximately 5 m shallowest EN4 value represents the 0 m
    boundary, matching the existing corrected 0–1000 m workflow.
    """
    interpolator = interp1d(
        np.asarray(native_depth, dtype=float),
        np.asarray(values, dtype=float),
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )

    output = interpolator(
        np.asarray(target_depth, dtype=float)
    )

    zero_index = int(
        np.argmin(
            np.abs(
                np.asarray(target_depth, dtype=float)
                - 0.0
            )
        )
    )

    output[zero_index] = np.asarray(
        values,
        dtype=float,
    )[0]

    return output


def trapezoid(values, coordinate, axis=0):
    function = (
        np.trapezoid
        if hasattr(np, "trapezoid")
        else np.trapz
    )

    return function(
        values,
        x=coordinate,
        axis=axis,
    )


def integrate_steric(
    delta_specific_volume,
    pressure_dbar,
    gravity,
):
    """
    1 dbar = 1e4 Pa. The result is metres.
    """
    return trapezoid(
        delta_specific_volume
        * 1.0e4
        / gravity,
        pressure_dbar,
        axis=0,
    )


def convert_height_to_cm(data_array):
    units = str(
        data_array.attrs.get("units", "")
    ).strip().lower()

    if units in {
        "cm",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
    }:
        output = data_array.copy()
        output.attrs["units"] = "cm"
        return output

    if units in {
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    }:
        output = data_array * 100.0
        output.attrs.update(data_array.attrs)
        output.attrs["units"] = "cm"
        return output

    sample = np.asarray(
        data_array.isel(
            time=slice(
                0,
                min(
                    12,
                    data_array.sizes.get("time", 1),
                ),
            )
        ).values,
        dtype=float,
    )

    sample = sample[np.isfinite(sample)]

    if (
        sample.size > 0
        and np.nanpercentile(
            np.abs(sample),
            99,
        ) < 1.0
    ):
        output = data_array * 100.0
        output.attrs.update(data_array.attrs)
        output.attrs["units"] = "cm"
        return output

    output = data_array.copy()
    output.attrs["units"] = units if units else "unknown"
    return output


def robust_symmetric_limit(
    fields,
    percentile=99.0,
):
    values = []

    for field in fields:
        array = np.asarray(
            field.values,
            dtype=float,
        )

        array = array[np.isfinite(array)]

        if array.size:
            values.append(array)

    if not values:
        return 1.0

    combined = np.concatenate(values)

    maximum = float(
        np.nanpercentile(
            np.abs(combined),
            percentile,
        )
    )

    if maximum <= 1.0:
        interval = 0.25
    elif maximum <= 2.0:
        interval = 0.5
    elif maximum <= 5.0:
        interval = 1.0
    elif maximum <= 10.0:
        interval = 2.0
    elif maximum <= 20.0:
        interval = 5.0
    else:
        interval = 10.0

    return max(
        interval,
        float(
            interval
            * np.ceil(
                maximum / interval
            )
        ),
    )


def robust_positive_limit(
    fields,
    percentile=99.0,
):
    values = []

    for field in fields:
        array = np.asarray(
            field.values,
            dtype=float,
        )

        array = array[np.isfinite(array)]

        if array.size:
            values.append(array)

    if not values:
        return 1.0

    combined = np.concatenate(values)

    maximum = float(
        np.nanpercentile(
            combined,
            percentile,
        )
    )

    if maximum <= 1.0:
        interval = 0.25
    elif maximum <= 2.0:
        interval = 0.5
    elif maximum <= 5.0:
        interval = 1.0
    elif maximum <= 10.0:
        interval = 2.0
    elif maximum <= 20.0:
        interval = 5.0
    else:
        interval = 10.0

    return max(
        interval,
        float(
            interval
            * np.ceil(
                maximum / interval
            )
        ),
    )


def coordinate_mesh(data_array):
    return np.meshgrid(
        data_array["lon"].values,
        data_array["lat"].values,
    )


def format_longitude(longitude):
    value = int(longitude)

    if value == 0:
        return "0°"

    if abs(value) == 180:
        return "180°"

    if value < 0:
        return f"{abs(value)}°W"

    return f"{value}°E"


# 7. VERIFY REQUIRED INPUTS
for component, file_path in FILES_1000.items():
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing corrected 0–1000 m {component} product:\n"
            f"{file_path}"
        )

have_all_2000_files = all(
    file_path.exists()
    for file_path in FILES_2000.values()
)

if not have_all_2000_files:
    if not RAW_EN4_FILE.exists():
        raise FileNotFoundError(
            "The 0–2000 m products do not exist and the raw EN4 file "
            f"was not found:\n{RAW_EN4_FILE}"
        )


# 8. CREATE 0–2000 M MONTHLY PRODUCTS WHEN NEEDED
def create_2000m_products():
    print_header(
        "CREATING CORRECTED EN4 0–2000 M MONTHLY STERIC PRODUCTS"
    )

    raw_input = RAW_EN4_FILE

    if STAGE_RAW_EN4_TO_LOCAL:
        raw_input = stage_file_to_local(
            RAW_EN4_FILE,
            LOCAL_CACHE_DIR,
        )

    raw = open_dataset_safely(
        raw_input,
        chunks={"time": 1},
    )

    time_name = detect_coordinate(raw, "time")
    depth_name = detect_coordinate(raw, "depth")
    lat_name = detect_coordinate(raw, "lat")
    lon_name = detect_coordinate(raw, "lon")

    raw = standardize_coordinates(
        raw,
        time_name=time_name,
        depth_name=depth_name,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    raw = subset_antarctic_latitudes(
        normalize_and_sort_longitude(raw)
    )

    temperature_name = find_exact_or_partial_variable(
        raw,
        [
            "temperature",
            "potential_temperature",
            "temp",
        ],
    )

    salinity_name = find_exact_or_partial_variable(
        raw,
        [
            "salinity",
            "practical_salinity",
            "psal",
        ],
    )

    temperature = raw[temperature_name].sel(
        time=slice(
            f"{YEAR_START}-01-01",
            f"{YEAR_END}-12-31",
        )
    )

    salinity = raw[salinity_name].sel(
        time=temperature["time"]
    )

    dates = pd.to_datetime(
        temperature["time"].values
    )

    native_depth = np.asarray(
        raw["depth"].values,
        dtype=float,
    )

    latitude = np.asarray(
        raw["lat"].values,
        dtype=float,
    )

    longitude = np.asarray(
        raw["lon"].values,
        dtype=float,
    )

    number_of_times = len(dates)
    number_of_latitudes = len(latitude)
    number_of_longitudes = len(longitude)

    print(
        f"Raw EN4 dimensions: time={number_of_times}, "
        f"depth={len(native_depth)}, "
        f"lat={number_of_latitudes}, "
        f"lon={number_of_longitudes}"
    )

    analysis_depth = target_depth_grid(
        native_depth,
        DEPTH_2000,
    )

    print(
        "0–2000 m analysis levels:",
        len(analysis_depth),
    )

    # The existing corrected 0–1000 m file contains GEBCO water depth
    # already interpolated onto the EN4 grid.
    mask_source = open_dataset_safely(
        FILES_1000["total"],
        chunks=None,
    )

    mask_source = standardize_coordinates(
        mask_source,
        time_name=detect_coordinate(mask_source, "time"),
        lat_name=detect_coordinate(mask_source, "lat"),
        lon_name=detect_coordinate(mask_source, "lon"),
    )

    mask_source = subset_antarctic_latitudes(
        normalize_and_sort_longitude(mask_source)
    )

    if "gebco_water_depth" not in mask_source.data_vars:
        raise KeyError(
            "The corrected 0–1000 m file does not contain "
            "'gebco_water_depth'."
        )

    water_depth = (
        mask_source["gebco_water_depth"]
        .sel(
            lat=latitude,
            lon=longitude,
        )
        .load()
        .values
        .astype(float)
    )

    bathymetry_mask = (
        np.isfinite(water_depth)
        & (water_depth >= DEPTH_2000)
    )

    print(
        "Cells with GEBCO depth >=2000 m:",
        f"{int(bathymetry_mask.sum()):,}",
    )

    # Fixed grid-cell all-month reference profiles.
    print(
        "Calculating fixed 2008–2025 all-month reference profiles..."
    )

    reference_temperature_native = (
        temperature.mean(
            dim="time",
            skipna=True,
        )
        .load()
        .values
        .astype(float)
    )

    reference_salinity_native = (
        salinity.mean(
            dim="time",
            skipna=True,
        )
        .load()
        .values
        .astype(float)
    )

    reference_temperature_k = interpolate_depth(
        reference_temperature_native,
        native_depth,
        analysis_depth,
    )

    reference_salinity = interpolate_depth(
        reference_salinity_native,
        native_depth,
        analysis_depth,
    )

    # EN4 temperature is treated as potential temperature in kelvin.
    reference_potential_temperature_c = (
        reference_temperature_k
        - 273.15
    )

    reference_complete = (
        np.isfinite(
            reference_potential_temperature_c
        ).all(axis=0)
        & np.isfinite(
            reference_salinity
        ).all(axis=0)
    )

    pressure_depth_lat = gsw.p_from_z(
        -analysis_depth[:, None],
        latitude[None, :],
    )

    pressure_3d = np.broadcast_to(
        pressure_depth_lat[:, :, None],
        (
            len(analysis_depth),
            number_of_latitudes,
            number_of_longitudes,
        ),
    )

    latitude_3d = latitude[None, :, None]
    longitude_3d = longitude[None, None, :]

    gravity_3d = gsw.grav(
        latitude_3d,
        pressure_3d,
    )

    reference_absolute_salinity = gsw.SA_from_SP(
        reference_salinity,
        pressure_3d,
        longitude_3d,
        latitude_3d,
    )

    reference_conservative_temperature = gsw.CT_from_pt(
        reference_absolute_salinity,
        reference_potential_temperature_c,
    )

    reference_specific_volume = gsw.specvol(
        reference_absolute_salinity,
        reference_conservative_temperature,
        pressure_3d,
    )

    total_output = np.full(
        (
            number_of_times,
            number_of_latitudes,
            number_of_longitudes,
        ),
        np.nan,
        dtype=np.float32,
    )

    thermo_output = np.full_like(
        total_output,
        np.nan,
    )

    halo_output = np.full_like(
        total_output,
        np.nan,
    )

    valid_output = np.zeros(
        total_output.shape,
        dtype=np.int8,
    )

    for time_index in range(number_of_times):
        date = pd.Timestamp(dates[time_index])

        print(
            f"[{time_index + 1:03d}/{number_of_times:03d}] "
            f"{date:%Y-%m}"
        )

        monthly_temperature_native = (
            temperature
            .isel(time=time_index)
            .load()
            .values
            .astype(float)
        )

        monthly_salinity_native = (
            salinity
            .isel(time=time_index)
            .load()
            .values
            .astype(float)
        )

        monthly_temperature_k = interpolate_depth(
            monthly_temperature_native,
            native_depth,
            analysis_depth,
        )

        monthly_salinity = interpolate_depth(
            monthly_salinity_native,
            native_depth,
            analysis_depth,
        )

        monthly_potential_temperature_c = (
            monthly_temperature_k
            - 273.15
        )

        monthly_complete = (
            np.isfinite(
                monthly_potential_temperature_c
            ).all(axis=0)
            & np.isfinite(
                monthly_salinity
            ).all(axis=0)
        )

        valid = (
            bathymetry_mask
            & reference_complete
            & monthly_complete
        )

        monthly_absolute_salinity = gsw.SA_from_SP(
            monthly_salinity,
            pressure_3d,
            longitude_3d,
            latitude_3d,
        )

        monthly_conservative_temperature = gsw.CT_from_pt(
            monthly_absolute_salinity,
            monthly_potential_temperature_c,
        )

        monthly_specific_volume = gsw.specvol(
            monthly_absolute_salinity,
            monthly_conservative_temperature,
            pressure_3d,
        )

        total_height = integrate_steric(
            monthly_specific_volume
            - reference_specific_volume,
            pressure_3d,
            gravity_3d,
        )

        thermosteric_conservative_temperature = gsw.CT_from_pt(
            reference_absolute_salinity,
            monthly_potential_temperature_c,
        )

        thermosteric_specific_volume = gsw.specvol(
            reference_absolute_salinity,
            thermosteric_conservative_temperature,
            pressure_3d,
        )

        thermosteric_height = integrate_steric(
            thermosteric_specific_volume
            - reference_specific_volume,
            pressure_3d,
            gravity_3d,
        )

        halosteric_height = (
            total_height
            - thermosteric_height
        )

        total_output[time_index] = np.where(
            valid,
            total_height,
            np.nan,
        ).astype(np.float32)

        thermo_output[time_index] = np.where(
            valid,
            thermosteric_height,
            np.nan,
        ).astype(np.float32)

        halo_output[time_index] = np.where(
            valid,
            halosteric_height,
            np.nan,
        ).astype(np.float32)

        valid_output[time_index] = valid.astype(
            np.int8
        )

        del (
            monthly_temperature_native,
            monthly_salinity_native,
            monthly_temperature_k,
            monthly_salinity,
            monthly_potential_temperature_c,
            monthly_absolute_salinity,
            monthly_conservative_temperature,
            monthly_specific_volume,
            thermosteric_conservative_temperature,
            thermosteric_specific_volume,
            total_height,
            thermosteric_height,
            halosteric_height,
        )

        gc.collect()

    coordinates = {
        "time": dates,
        "latitude": latitude.astype(np.float32),
        "longitude": longitude.astype(np.float32),
    }

    common_dataset_attributes = {
        "Conventions": "CF-1.8",
        "source_en4_file": str(RAW_EN4_FILE),
        "period": f"{YEAR_START}-01 through {YEAR_END}-12",
        "region": "Southern Ocean EN4 Antarctic domain",
        "integration_layer": "0–2000 m",
        "reference_state": (
            "One fixed grid-cell, depth-dependent all-month mean "
            "potential-temperature and practical-salinity profile "
            "for 2008–2025."
        ),
        "depth_mask_rule": (
            "GEBCO water depth >=2000 m and complete monthly/reference "
            "temperature and salinity through the complete 0–2000 m layer."
        ),
        "seasonal_cycle_preserved": (
            "Yes; no calendar-month climatology was removed."
        ),
        "temperature_interpretation": (
            "EN4 potential temperature in kelvin."
        ),
        "salinity_interpretation": (
            "EN4 Practical Salinity."
        ),
        "processing_software": (
            "Python, NumPy, Xarray, SciPy and GSW-Python"
        ),
        "processing_time_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    output_specs = {
        "total": {
            "variable": "total_steric_height",
            "long_name": (
                "EN4 total steric height anomaly "
                "integrated over 0–2000 m"
            ),
            "values": total_output,
            "calculation": (
                "Integral of monthly minus fixed-reference "
                "specific volume over pressure divided by local gravity."
            ),
        },
        "thermo": {
            "variable": "thermosteric_height",
            "long_name": (
                "EN4 thermosteric height anomaly "
                "integrated over 0–2000 m"
            ),
            "values": thermo_output,
            "calculation": (
                "Monthly potential-temperature effect with Absolute "
                "Salinity held at the fixed reference profile."
            ),
        },
        "halo": {
            "variable": "halosteric_height",
            "long_name": (
                "EN4 halosteric height anomaly "
                "integrated over 0–2000 m"
            ),
            "values": halo_output,
            "calculation": (
                "Total steric height minus corrected thermosteric height."
            ),
        },
    }

    for component, specification in output_specs.items():
        output_dataset = xr.Dataset(
            {
                specification["variable"]: (
                    (
                        "time",
                        "latitude",
                        "longitude",
                    ),
                    specification["values"],
                ),
                "valid_layer_mask": (
                    (
                        "time",
                        "latitude",
                        "longitude",
                    ),
                    valid_output,
                ),
                "gebco_water_depth": (
                    (
                        "latitude",
                        "longitude",
                    ),
                    water_depth.astype(np.float32),
                ),
            },
            coords=coordinates,
        )

        output_dataset[
            specification["variable"]
        ].attrs.update(
            {
                "long_name": specification["long_name"],
                "units": "m",
                "calculation": specification["calculation"],
            }
        )

        output_dataset["valid_layer_mask"].attrs.update(
            {
                "long_name": "monthly valid 0–2000 m layer mask",
                "flag_values": np.array(
                    [0, 1],
                    dtype=np.int8,
                ),
                "flag_meanings": "invalid valid",
            }
        )

        output_dataset["gebco_water_depth"].attrs.update(
            {
                "long_name": "GEBCO ocean water depth",
                "units": "m",
                "positive": "down",
            }
        )

        output_dataset.attrs.update(
            common_dataset_attributes
        )

        encoding = {
            specification["variable"]: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(9.96921e36),
            },
            "valid_layer_mask": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "int8",
                "_FillValue": np.int8(-127),
            },
            "gebco_water_depth": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(9.96921e36),
            },
        }

        if FILES_2000[component].exists():
            FILES_2000[component].unlink()

        output_dataset.to_netcdf(
            FILES_2000[component],
            engine="h5netcdf",
            encoding=encoding,
        )

        print(
            f"Saved {component} 0–2000 m product:\n"
            f"{FILES_2000[component]}"
        )

        output_dataset.close()

    mask_source.close()
    raw.close()

    del (
        total_output,
        thermo_output,
        halo_output,
        valid_output,
        reference_temperature_native,
        reference_salinity_native,
        reference_temperature_k,
        reference_salinity,
        reference_potential_temperature_c,
        reference_absolute_salinity,
        reference_conservative_temperature,
        reference_specific_volume,
    )

    gc.collect()


if (
    REUSE_EXISTING_2000_PRODUCTS
    and have_all_2000_files
):
    print_header(
        "USING EXISTING CORRECTED EN4 0–2000 M PRODUCTS"
    )

    for component in COMPONENT_ORDER:
        print(
            f"{component:>7}: {FILES_2000[component]}"
        )

else:
    create_2000m_products()


# 9. OPEN THE SIX MONTHLY PRODUCTS
print_header(
    "OPENING 0–1000 M AND 0–2000 M MONTHLY PRODUCTS"
)

PREFERRED_VARIABLES = {
    "total": [
        "total_steric_height",
        "total_steric",
    ],
    "thermo": [
        "thermosteric_height",
        "thermosteric",
    ],
    "halo": [
        "halosteric_height",
        "halosteric",
    ],
}


def load_steric_product(file_path, component):
    dataset = open_dataset_safely(
        file_path,
        chunks={"time": 12},
    )

    dataset = standardize_coordinates(
        dataset,
        time_name=detect_coordinate(dataset, "time"),
        lat_name=detect_coordinate(dataset, "lat"),
        lon_name=detect_coordinate(dataset, "lon"),
    )

    dataset = subset_antarctic_latitudes(
        normalize_and_sort_longitude(dataset)
    )

    variable_name = choose_data_variable(
        dataset,
        preferred_names=PREFERRED_VARIABLES[component],
        excluded_names=[
            "valid_layer_mask",
            "gebco_water_depth",
        ],
    )

    data = convert_height_to_cm(
        dataset[variable_name].sel(
            time=slice(
                f"{YEAR_START}-01-01",
                f"{YEAR_END}-12-31",
            )
        )
    )

    return dataset, data


datasets_1000 = {}
datasets_2000 = {}
data_1000 = {}
data_2000 = {}

for component in COMPONENT_ORDER:
    datasets_1000[component], data_1000[component] = (
        load_steric_product(
            FILES_1000[component],
            component,
        )
    )

    datasets_2000[component], data_2000[component] = (
        load_steric_product(
            FILES_2000[component],
            component,
        )
    )


# 10. ALIGN TIME AND GRID
alignment_list = []

for component in COMPONENT_ORDER:
    alignment_list.append(data_1000[component])
    alignment_list.append(data_2000[component])

aligned = xr.align(
    *alignment_list,
    join="inner",
)

for component_index, component in enumerate(COMPONENT_ORDER):
    data_1000[component] = aligned[2 * component_index]
    data_2000[component] = aligned[2 * component_index + 1]

common_time = data_1000["total"]["time"]
common_latitude = data_1000["total"]["lat"]
common_longitude = data_1000["total"]["lon"]

number_of_months = len(common_time)

print(
    "Common period:",
    pd.Timestamp(common_time.values[0]),
    "to",
    pd.Timestamp(common_time.values[-1]),
)

print("Common months:", number_of_months)
print(
    "Common grid:",
    len(common_latitude),
    "latitudes ×",
    len(common_longitude),
    "longitudes",
)


# 11. FIXED COMMON MASK: ONLY CELLS DEEPER THAN 2000 M
if "gebco_water_depth" not in datasets_1000["total"].data_vars:
    raise KeyError(
        "The 0–1000 m total product does not contain "
        "'gebco_water_depth'."
    )

water_depth = (
    datasets_1000["total"]["gebco_water_depth"]
    .rename(
        {
            name: replacement
            for name, replacement in {
                "latitude": "lat",
                "longitude": "lon",
            }.items()
            if (
                name in datasets_1000["total"][
                    "gebco_water_depth"
                ].dims
                and replacement not in datasets_1000[
                    "total"
                ]["gebco_water_depth"].dims
            )
        }
    )
    .sel(
        lat=common_latitude,
        lon=common_longitude,
    )
    .load()
)

deep_water_mask = (
    np.isfinite(water_depth)
    & (water_depth >= DEPTH_2000)
)

common_mask = deep_water_mask.copy()

for component in COMPONENT_ORDER:
    valid_fraction_1000 = (
        data_1000[component]
        .notnull()
        .mean(dim="time")
    )

    valid_fraction_2000 = (
        data_2000[component]
        .notnull()
        .mean(dim="time")
    )

    common_mask = (
        common_mask
        & (
            valid_fraction_1000
            >= COMMON_VALID_FRACTION - 1.0e-10
        )
        & (
            valid_fraction_2000
            >= COMMON_VALID_FRACTION - 1.0e-10
        )
    )

common_mask = common_mask.compute()

common_cell_count = int(
    common_mask.sum().values
)

if common_cell_count == 0:
    raise RuntimeError(
        "The fixed common >2000 m mask contains no cells. "
        "Reduce COMMON_VALID_FRACTION only if scientifically justified."
    )

print_header(
    "FIXED COMMON MASK"
)

print(
    "GEBCO depth threshold     : >=2000 m"
)

print(
    "Monthly valid fraction    :",
    f"{COMMON_VALID_FRACTION:.2f}",
)

print(
    "Common spatial cells      :",
    f"{common_cell_count:,}",
)


# 12. CALCULATE THE DEPTH-SENSITIVITY FIELDS
def seasonal_anomalies(data_array):
    all_month_mean = data_array.mean(
        dim="time",
        skipna=True,
    )

    output = {}

    for season in SEASON_ORDER:
        seasonal_mean = (
            data_array
            .where(
                data_array["time"].dt.month.isin(
                    SEASON_MONTHS[season]
                ),
                drop=True,
            )
            .mean(
                dim="time",
                skipna=True,
            )
        )

        output[season] = (
            seasonal_mean
            - all_month_mean
        )

    return output


def select_comparison_field(data_array, mode):
    anomalies = seasonal_anomalies(data_array)

    if mode == "winter_minus_summer":
        field = (
            anomalies["Winter"]
            - anomalies["Summer"]
        )

    elif mode == "spring_anomaly":
        field = anomalies["Spring"]

    elif mode == "summer_anomaly":
        field = anomalies["Summer"]

    elif mode == "autumn_anomaly":
        field = anomalies["Autumn"]

    elif mode == "winter_anomaly":
        field = anomalies["Winter"]

    elif mode == "seasonal_peak_to_peak":
        stack = xr.concat(
            [
                anomalies[season]
                for season in SEASON_ORDER
            ],
            dim="season",
        )

        field = (
            stack.max(
                dim="season",
                skipna=True,
            )
            - stack.min(
                dim="season",
                skipna=True,
            )
        )

    elif mode == "seasonal_rms":
        stack = xr.concat(
            [
                anomalies[season]
                for season in SEASON_ORDER
            ],
            dim="season",
        )

        field = np.sqrt(
            (
                stack ** 2
            ).mean(
                dim="season",
                skipna=True,
            )
        )

    else:
        raise ValueError(mode)

    return field


fields_1000 = {}
fields_2000 = {}
difference_fields = {}

for component in COMPONENT_ORDER:
    fields_1000[component] = (
        select_comparison_field(
            data_1000[component],
            COMPARISON_MODE,
        )
        .where(common_mask)
        .compute()
    )

    fields_2000[component] = (
        select_comparison_field(
            data_2000[component],
            COMPARISON_MODE,
        )
        .where(common_mask)
        .compute()
    )

    difference_fields[component] = (
        fields_2000[component]
        - fields_1000[component]
    ).where(common_mask)


# 13. SAVE PROCESSED FIGURE S4 FIELDS
component_coordinate = xr.DataArray(
    COMPONENT_ORDER,
    dims="component",
    name="component",
)

fields_1000_stack = xr.concat(
    [
        fields_1000[component]
        for component in COMPONENT_ORDER
    ],
    dim=component_coordinate,
)

fields_2000_stack = xr.concat(
    [
        fields_2000[component]
        for component in COMPONENT_ORDER
    ],
    dim=component_coordinate,
)

difference_stack = xr.concat(
    [
        difference_fields[component]
        for component in COMPONENT_ORDER
    ],
    dim=component_coordinate,
)

output_fields = xr.Dataset(
    {
        "field_0_1000m": fields_1000_stack,
        "field_0_2000m": fields_2000_stack,
        "difference_0_2000m_minus_0_1000m": difference_stack,
        "common_deeper_than_2000m_mask": common_mask.astype(np.int8),
        "gebco_water_depth": water_depth,
    }
)

for variable_name in [
    "field_0_1000m",
    "field_0_2000m",
    "difference_0_2000m_minus_0_1000m",
]:
    output_fields[variable_name].attrs["units"] = "cm"

output_fields[
    "field_0_1000m"
].attrs["long_name"] = (
    f"0–1000 m EN4 {COMPARISON_MODE.replace('_', ' ')} field"
)

output_fields[
    "field_0_2000m"
].attrs["long_name"] = (
    f"0–2000 m EN4 {COMPARISON_MODE.replace('_', ' ')} field"
)

output_fields[
    "difference_0_2000m_minus_0_1000m"
].attrs["long_name"] = (
    f"0–2000 m minus 0–1000 m difference in "
    f"EN4 {COMPARISON_MODE.replace('_', ' ')}"
)

output_fields[
    "common_deeper_than_2000m_mask"
].attrs.update(
    {
        "long_name": (
            "fixed common cells deeper than 2000 m"
        ),
        "flag_values": np.array(
            [0, 1],
            dtype=np.int8,
        ),
        "flag_meanings": "excluded included",
    }
)

output_fields.attrs.update(
    {
        "title": "Figure S4 EN4 integration-depth sensitivity",
        "comparison_mode": COMPARISON_MODE,
        "difference_definition": DIFFERENCE_DEFINITION,
        "common_valid_fraction": COMMON_VALID_FRACTION,
        "common_cell_count": common_cell_count,
        "period": f"{YEAR_START}-01 through {YEAR_END}-12",
    }
)

output_fields.rename(
    {
        "lat": "latitude",
        "lon": "longitude",
    }
).to_netcdf(
    OUT_FIELDS_NC,
    engine="h5netcdf",
    encoding={
        "field_0_1000m": {
            "zlib": True,
            "complevel": 4,
        },
        "field_0_2000m": {
            "zlib": True,
            "complevel": 4,
        },
        "difference_0_2000m_minus_0_1000m": {
            "zlib": True,
            "complevel": 4,
        },
        "common_deeper_than_2000m_mask": {
            "zlib": True,
            "complevel": 4,
        },
        "gebco_water_depth": {
            "zlib": True,
            "complevel": 4,
        },
    },
)


# 14. SUMMARY TABLE
summary_rows = []

for component in COMPONENT_ORDER:
    for case_name, field in [
        ("0–1000 m", fields_1000[component]),
        ("0–2000 m", fields_2000[component]),
        ("Difference", difference_fields[component]),
    ]:
        values = np.asarray(
            field.values,
            dtype=float,
        )

        values = values[np.isfinite(values)]

        summary_rows.append(
            {
                "component": COMPONENT_LABELS[component],
                "depth_case": case_name,
                "comparison_mode": COMPARISON_MODE,
                "n_common_cells": int(values.size),
                "mean_cm": (
                    float(np.mean(values))
                    if values.size
                    else np.nan
                ),
                "median_cm": (
                    float(np.median(values))
                    if values.size
                    else np.nan
                ),
                "standard_deviation_cm": (
                    float(np.std(values))
                    if values.size
                    else np.nan
                ),
                "minimum_cm": (
                    float(np.min(values))
                    if values.size
                    else np.nan
                ),
                "maximum_cm": (
                    float(np.max(values))
                    if values.size
                    else np.nan
                ),
                "RMSE_cm": (
                    float(
                        np.sqrt(
                            np.mean(
                                values ** 2
                            )
                        )
                    )
                    if values.size
                    else np.nan
                ),
            }
        )

summary_table = pd.DataFrame(
    summary_rows
)

summary_table.to_csv(
    OUT_SUMMARY_CSV,
    index=False,
)




# DATA PRODUCT GROUP 14: TABLE S4 COMPLETE RECOMPUTED TREND STATISTICS
# Source: so_sealevel_paper_fig.py
# File name: Table_S4_complete_trend_statistics_RECOMPUTED_detailed.csv
# File name: Table_S4_complete_trend_statistics_RECOMPUTED_publication.csv
# File name: Table_S4_complete_trend_statistics_RECOMPUTED.xlsx
# File name: Table_S4_sector_annual_timeseries_RECOMPUTED.csv
# File name: Table_S4_recomputation_audit_vs_original.csv


# TABLE S4 — COMPLETE SEA-WISE TREND STATISTICS, 2008–2025
# FULL RECOMPUTATION FROM THE PRECOMPUTED FIGURE 11 ANNUAL FIELDS
#
# Purpose
# -------
# Recompute, audit, and save genuine sea-wise trend statistics for:
#   1. SLA
#   2. Total steric height
#   3. Thermosteric height
#   4. Halosteric height
#
# Input
# -----
# Fig11_deseasonalized_annual_anomalies_common_grid.nc
#
# Main correction relative to the earlier Figure 11 calculation
# The Hamed–Rao variance correction is calculated as:
#
#   correction_factor = 1 + [2 / n(n-1)(n-2)] * ABS(weighted_ACF_sum)
#
# Using ABS prevents a negative weighted autocorrelation sum from being
# clipped to an artificial value such as 1e-6. The earlier clipping could
# create unrealistically large Z statistics. No 1e-6 variance-factor clip
# is used in this corrected computation.
#
# Statistical workflow
# --------------------
# • Area-weighted sea-wise annual mean from the fixed common grid.
# • Sen slope, converted to cm decade^-1.
# • Hamed–Rao modified Mann–Kendall test on the annual series.
# • Benjamini–Hochberg FDR correction separately across 13 seas for each
#   variable, matching the original Figure 11 FDR scope.
# • 3-year circular moving-block residual-bootstrap 95% CI, 2,000 runs.
# • Explicit saving of S, uncorrected variance, correction factor,
#   corrected variance, Z, raw p, FDR q, and significant lags.
#
# Outputs
# -------
# Table_S4_complete_trend_statistics_RECOMPUTED_detailed.csv
# Table_S4_complete_trend_statistics_RECOMPUTED_publication.csv
# Table_S4_complete_trend_statistics_RECOMPUTED.xlsx
# Table_S4_sector_annual_timeseries_RECOMPUTED.csv
# Table_S4_recomputation_audit_vs_original.csv
# Table_S4_recomputation_summary.txt




# 0. INSTALL MISSING PACKAGES

import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "h5py": "h5py",
    "openpyxl": "openpyxl",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )
    print("Package installation completed.")
else:
    print("All required packages are already installed.")



# 1. IMPORTS
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore", category=RuntimeWarning)



# 3. INPUT AND OUTPUT PATHS
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/SAM_Thesis/paper2")
LOCAL_TEST_DIR = Path("/mnt/data")

ANNUAL_FILE_CANDIDATES = [
    DRIVE_OUTPUT_DIR / "Fig11_deseasonalized_annual_anomalies_common_grid.nc",
    LOCAL_TEST_DIR / "Fig11_deseasonalized_annual_anomalies_common_grid.nc",
]

ORIGINAL_SECTOR_CSV_CANDIDATES = [
    DRIVE_OUTPUT_DIR / "Fig11_13sea_Sen_slopes_bootstrap_CI.csv",
    LOCAL_TEST_DIR / "Fig11_13sea_Sen_slopes_bootstrap_CI.csv",
]


def first_existing(candidates, required=True):
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        raise FileNotFoundError(
            "None of the required files was found:\n"
            + "\n".join(str(path) for path in candidates)
        )

    return None


ANNUAL_NC = first_existing(ANNUAL_FILE_CANDIDATES, required=True)
ORIGINAL_SECTOR_CSV = first_existing(
    ORIGINAL_SECTOR_CSV_CANDIDATES,
    required=False,
)

# In Colab, always write to the thesis paper2 directory.
# Outside Colab, write next to the uploaded test file.
if str(ANNUAL_NC).startswith("/content/drive"):
    OUTPUT_DIR = DRIVE_OUTPUT_DIR
else:
    OUTPUT_DIR = LOCAL_TEST_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DETAILED_CSV = (
    OUTPUT_DIR
    / "Table_S4_complete_trend_statistics_RECOMPUTED_detailed.csv"
)

OUT_PUBLICATION_CSV = (
    OUTPUT_DIR
    / "Table_S4_complete_trend_statistics_RECOMPUTED_publication.csv"
)

OUT_EXCEL = (
    OUTPUT_DIR
    / "Table_S4_complete_trend_statistics_RECOMPUTED.xlsx"
)

OUT_ANNUAL_SERIES_CSV = (
    OUTPUT_DIR
    / "Table_S4_sector_annual_timeseries_RECOMPUTED.csv"
)

OUT_AUDIT_CSV = (
    OUTPUT_DIR
    / "Table_S4_recomputation_audit_vs_original.csv"
)

OUT_SUMMARY_TXT = (
    OUTPUT_DIR
    / "Table_S4_recomputation_summary.txt"
)


# 4. ANALYSIS SETTINGS
YEAR_START = 2008
YEAR_END = 2025

MIN_VALID_YEARS = 12

MK_ALPHA = 0.05
FDR_ALPHA = 0.05

# None means use all available lags that exceed the approximate
# white-noise confidence limits. Set to 3 for a first-three-lag
# sensitivity analysis without changing any other code.
HAMED_RAO_MAX_LAG = None

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH_YEARS = 3
BOOTSTRAP_SEED = 4217

# To match Figure 11, FDR is applied independently for each variable
# across its 13 sea-wise tests.
FDR_SCOPE = "within_variable"

VARIABLE_ORDER = [
    "SLA",
    "Total steric",
    "Thermosteric",
    "Halosteric",
]

VARIABLE_TO_NETCDF = OrderedDict([
    ("SLA", "sla_annual_anomaly"),
    ("Total steric", "total_steric_annual_anomaly"),
    ("Thermosteric", "thermosteric_annual_anomaly"),
    ("Halosteric", "halosteric_annual_anomaly"),
])


# 5. ANTARCTIC MARGINAL-SEA DEFINITIONS
SEA_SECTORS = OrderedDict([
    (
        "WED",
        {
            "name": "Weddell Sea",
            "lon_min": -60.0,
            "lon_max": -20.0,
        },
    ),
    (
        "KHV",
        {
            "name": "King Haakon VII Sea",
            "lon_min": -20.0,
            "lon_max": 10.0,
        },
    ),
    (
        "RLS",
        {
            "name": "Riiser-Larsen Sea",
            "lon_min": 10.0,
            "lon_max": 35.0,
        },
    ),
    (
        "LAZ",
        {
            "name": "Lazarev Sea",
            "lon_min": 35.0,
            "lon_max": 60.0,
        },
    ),
    (
        "COS",
        {
            "name": "Cosmonauts Sea",
            "lon_min": 60.0,
            "lon_max": 90.0,
        },
    ),
    (
        "COO",
        {
            "name": "Cooperation Sea",
            "lon_min": 90.0,
            "lon_max": 115.0,
        },
    ),
    (
        "DAV",
        {
            "name": "Davis Sea",
            "lon_min": 115.0,
            "lon_max": 130.0,
        },
    ),
    (
        "MAW",
        {
            "name": "Mawson Sea",
            "lon_min": 130.0,
            "lon_max": 150.0,
        },
    ),
    (
        "DUR",
        {
            "name": "D'Urville Sea",
            "lon_min": 150.0,
            "lon_max": 170.0,
        },
    ),
    (
        "SOM",
        {
            "name": "Somov Sea",
            "lon_min": 170.0,
            "lon_max": -160.0,
        },
    ),
    (
        "ROS",
        {
            "name": "Ross Sea",
            "lon_min": -160.0,
            "lon_max": -130.0,
        },
    ),
    (
        "AMU",
        {
            "name": "Amundsen Sea",
            "lon_min": -130.0,
            "lon_max": -100.0,
        },
    ),
    (
        "BEL",
        {
            "name": "Bellingshausen Sea",
            "lon_min": -100.0,
            "lon_max": -60.0,
        },
    ),
])

SEA_ORDER = list(SEA_SECTORS.keys())


# 6. GENERAL HELPERS
def read_hdf5_array(handle, variable_name, dtype=float):
    """Read one HDF5/NetCDF4 variable and apply its fill value."""
    if variable_name not in handle:
        raise KeyError(
            f"Missing variable '{variable_name}'. "
            f"Available variables: {list(handle.keys())}"
        )

    dataset = handle[variable_name]
    values = np.asarray(dataset[:], dtype=dtype)

    fill_value = dataset.attrs.get("_FillValue", None)

    if fill_value is not None:
        fill_array = np.asarray(fill_value).reshape(-1)

        if fill_array.size:
            fill_scalar = float(fill_array[0])

            if np.isfinite(fill_scalar):
                values = np.where(
                    values == fill_scalar,
                    np.nan,
                    values,
                )

    return values


def longitude_sector_mask(
    longitude_values,
    minimum_longitude,
    maximum_longitude,
):
    """One-dimensional sector mask, including dateline crossing."""
    longitude_values = np.asarray(longitude_values, dtype=float)

    if minimum_longitude <= maximum_longitude:
        return (
            (longitude_values >= minimum_longitude)
            & (longitude_values < maximum_longitude)
        )

    return (
        (longitude_values >= minimum_longitude)
        | (longitude_values < maximum_longitude)
    )


def sen_slope(values, times):
    """Median pairwise Sen slope in value units per time unit."""
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(times)
    values = values[valid]
    times = times[valid]

    number_of_values = values.size

    if number_of_values < 2:
        return np.nan

    first_indices, second_indices = np.triu_indices(
        number_of_values,
        k=1,
    )

    time_difference = (
        times[second_indices]
        - times[first_indices]
    )

    value_difference = (
        values[second_indices]
        - values[first_indices]
    )

    valid_pairs = time_difference != 0.0

    if not np.any(valid_pairs):
        return np.nan

    pairwise_slopes = (
        value_difference[valid_pairs]
        / time_difference[valid_pairs]
    )

    return float(np.nanmedian(pairwise_slopes))


def mann_kendall_score_and_variance(values):
    """Mann–Kendall S and tie-corrected unmodified variance."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    number_of_values = values.size

    if number_of_values < 2:
        return np.nan, np.nan

    score = 0.0

    for index in range(number_of_values - 1):
        score += np.sign(
            values[index + 1:] - values[index]
        ).sum()

    _, tie_counts = np.unique(
        values,
        return_counts=True,
    )

    tie_term = np.sum(
        tie_counts
        * (tie_counts - 1)
        * (2 * tie_counts + 5)
    )

    variance = (
        number_of_values
        * (number_of_values - 1)
        * (2 * number_of_values + 5)
        - tie_term
    ) / 18.0

    return float(score), float(variance)


def biased_autocorrelation(values, maximum_lag):
    """
    Biased ACF used by the reference pyMannKendall implementation.

    The covariance denominator remains n at every lag, after which the
    sequence is normalized by the lag-zero covariance.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Autocorrelation input must be one-dimensional.")

    number_of_values = values.size

    if number_of_values == 0:
        return np.asarray([], dtype=float)

    maximum_lag = int(
        min(
            max(maximum_lag, 0),
            number_of_values - 1,
        )
    )

    centered = values - np.mean(values)

    auto_covariance = (
        np.correlate(
            centered,
            centered,
            mode="full",
        )[number_of_values - 1:]
        / number_of_values
    )

    selected = auto_covariance[:maximum_lag + 1]

    if selected.size == 0:
        return selected

    if auto_covariance[0] == 0.0:
        return selected

    return selected / auto_covariance[0]


def hamed_rao_modified_mk(
    values,
    times,
    alpha=MK_ALPHA,
    maximum_lag=HAMED_RAO_MAX_LAG,
):
    """
    Corrected Hamed–Rao modified Mann–Kendall test.

    The key reliability correction is ABS(weighted_acf_sum), consistent
    with the established pyMannKendall implementation. This guarantees a
    non-deflated correction factor >= 1 and removes the need for an
    artificial 1e-6 lower clipping value.
    """
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(times)
    values = values[valid]
    times = times[valid]

    number_of_values = values.size

    empty_result = {
        "n": int(number_of_values),
        "sen_slope_per_year": np.nan,
        "mk_s": np.nan,
        "mk_tau": np.nan,
        "mk_variance_uncorrected": np.nan,
        "standard_mk_z": np.nan,
        "standard_mk_p": np.nan,
        "hamed_rao_max_lag": np.nan,
        "acf_confidence_limit": np.nan,
        "significant_acf_lags": "",
        "significant_acf_values": "",
        "weighted_acf_sum": np.nan,
        "hamed_rao_correction_factor": np.nan,
        "mk_variance_corrected": np.nan,
        "mann_kendall_z": np.nan,
        "raw_p": np.nan,
    }

    if number_of_values < 4:
        return empty_result

    score, uncorrected_variance = (
        mann_kendall_score_and_variance(values)
    )

    if not np.isfinite(uncorrected_variance) or uncorrected_variance <= 0.0:
        result = empty_result.copy()
        result.update(
            {
                "mk_s": score,
                "mk_tau": 0.0,
                "mk_variance_uncorrected": uncorrected_variance,
                "mann_kendall_z": 0.0,
                "raw_p": 1.0,
            }
        )
        return result

    slope_per_year = sen_slope(values, times)

    # An intercept does not affect ranks, but it improves interpretability
    # of the detrended residual sequence.
    intercept = np.nanmedian(
        values - slope_per_year * times
    )

    detrended = (
        values
        - (intercept + slope_per_year * times)
    )

    ranked_detrended = rankdata(
        detrended,
        method="average",
    )

    if maximum_lag is None:
        selected_maximum_lag = number_of_values - 1
    else:
        selected_maximum_lag = min(
            int(maximum_lag),
            number_of_values - 1,
        )

    autocorrelation_values = biased_autocorrelation(
        ranked_detrended,
        selected_maximum_lag,
    )

    confidence_limit = (
        norm.ppf(1.0 - alpha / 2.0)
        / np.sqrt(number_of_values)
    )

    weighted_autocorrelation_sum = 0.0
    significant_lags = []
    significant_values = []

    for lag in range(1, selected_maximum_lag + 1):
        coefficient = float(autocorrelation_values[lag])

        if (
            np.isfinite(coefficient)
            and abs(coefficient) > confidence_limit
        ):
            significant_lags.append(lag)
            significant_values.append(coefficient)

            weighted_autocorrelation_sum += (
                (number_of_values - lag)
                * (number_of_values - lag - 1)
                * (number_of_values - lag - 2)
                * coefficient
            )

    denominator = (
        number_of_values
        * (number_of_values - 1)
        * (number_of_values - 2)
    )

    correction_factor = (
        1.0
        + (
            2.0
            * abs(weighted_autocorrelation_sum)
            / denominator
        )
    )

    corrected_variance = (
        uncorrected_variance
        * correction_factor
    )

    if score > 0.0:
        corrected_z = (
            score - 1.0
        ) / np.sqrt(corrected_variance)

        standard_z = (
            score - 1.0
        ) / np.sqrt(uncorrected_variance)

    elif score < 0.0:
        corrected_z = (
            score + 1.0
        ) / np.sqrt(corrected_variance)

        standard_z = (
            score + 1.0
        ) / np.sqrt(uncorrected_variance)

    else:
        corrected_z = 0.0
        standard_z = 0.0

    raw_p = float(
        2.0 * norm.sf(abs(corrected_z))
    )

    standard_p = float(
        2.0 * norm.sf(abs(standard_z))
    )

    tau = float(
        score
        / (0.5 * number_of_values * (number_of_values - 1))
    )

    return {
        "n": int(number_of_values),
        "sen_slope_per_year": float(slope_per_year),
        "mk_s": float(score),
        "mk_tau": tau,
        "mk_variance_uncorrected": float(uncorrected_variance),
        "standard_mk_z": float(standard_z),
        "standard_mk_p": standard_p,
        "hamed_rao_max_lag": int(selected_maximum_lag),
        "acf_confidence_limit": float(confidence_limit),
        "significant_acf_lags": ";".join(
            str(lag) for lag in significant_lags
        ),
        "significant_acf_values": ";".join(
            f"{coefficient:.8f}"
            for coefficient in significant_values
        ),
        "weighted_acf_sum": float(weighted_autocorrelation_sum),
        "hamed_rao_correction_factor": float(correction_factor),
        "mk_variance_corrected": float(corrected_variance),
        "mann_kendall_z": float(corrected_z),
        "raw_p": raw_p,
    }


def benjamini_hochberg_qvalues(p_values):
    """Benjamini–Hochberg FDR-adjusted q values."""
    p_values = np.asarray(p_values, dtype=np.float64)

    q_values = np.full_like(
        p_values,
        np.nan,
        dtype=np.float64,
    )

    finite_mask = np.isfinite(p_values)
    finite_p_values = p_values[finite_mask]

    number_of_tests = finite_p_values.size

    if number_of_tests == 0:
        return q_values

    order = np.argsort(finite_p_values)
    sorted_p = finite_p_values[order]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=np.float64,
    )

    sorted_q = (
        sorted_p
        * number_of_tests
        / ranks
    )

    sorted_q = np.minimum.accumulate(
        sorted_q[::-1]
    )[::-1]

    sorted_q = np.clip(
        sorted_q,
        0.0,
        1.0,
    )

    finite_q = np.empty_like(sorted_q)
    finite_q[order] = sorted_q
    q_values[finite_mask] = finite_q

    return q_values


def moving_block_bootstrap_ci(
    annual_values,
    years,
    block_length=BOOTSTRAP_BLOCK_LENGTH_YEARS,
    number_of_bootstraps=BOOTSTRAP_REPLICATES,
    random_seed=BOOTSTRAP_SEED,
):
    """
    Residual circular moving-block bootstrap confidence interval.

    A Sen trend is fitted first. Residuals are sampled in contiguous
    circular blocks and added to the fitted values at the original time
    positions. Slopes are returned in cm decade^-1.
    """
    annual_values = np.asarray(annual_values, dtype=np.float64)
    years = np.asarray(years, dtype=np.float64)

    valid = np.isfinite(annual_values) & np.isfinite(years)
    annual_values = annual_values[valid]
    years = years[valid]

    number_of_years = annual_values.size

    if number_of_years < MIN_VALID_YEARS:
        return np.nan, np.nan

    point_slope = sen_slope(
        annual_values,
        years,
    )

    intercept = np.nanmedian(
        annual_values - point_slope * years
    )

    fitted = intercept + point_slope * years
    residuals = annual_values - fitted

    random_generator = np.random.default_rng(
        random_seed
    )

    bootstrap_slopes = np.full(
        number_of_bootstraps,
        np.nan,
        dtype=np.float64,
    )

    relative_time = np.arange(
        number_of_years,
        dtype=np.float64,
    )

    for bootstrap_index in range(number_of_bootstraps):
        sampled_indices = []

        while len(sampled_indices) < number_of_years:
            block_start = int(
                random_generator.integers(
                    0,
                    number_of_years,
                )
            )

            sampled_indices.extend(
                [
                    (block_start + offset) % number_of_years
                    for offset in range(block_length)
                ]
            )

        sampled_indices = np.asarray(
            sampled_indices[:number_of_years],
            dtype=np.int64,
        )

        bootstrap_values = (
            fitted
            + residuals[sampled_indices]
        )

        bootstrap_slopes[bootstrap_index] = (
            sen_slope(
                bootstrap_values,
                relative_time,
            )
            * 10.0
        )

    finite_slopes = bootstrap_slopes[
        np.isfinite(bootstrap_slopes)
    ]

    if finite_slopes.size == 0:
        return np.nan, np.nan

    return (
        float(np.nanpercentile(finite_slopes, 2.5)),
        float(np.nanpercentile(finite_slopes, 97.5)),
    )


def p_for_publication(value):
    """Publication-style p-value text."""
    if not np.isfinite(value):
        return "N/A"

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def yes_no(value):
    if pd.isna(value):
        return "N/A"

    return "Yes" if bool(value) else "No"


# 7. LOAD THE PRECOMPUTED ANNUAL FIELDS
print("\nUsing annual NetCDF:")
print(ANNUAL_NC)

with h5py.File(ANNUAL_NC, "r") as handle:
    years = read_hdf5_array(
        handle,
        "year",
        dtype=float,
    )

    latitude = read_hdf5_array(
        handle,
        "latitude",
        dtype=float,
    )

    longitude = read_hdf5_array(
        handle,
        "longitude",
        dtype=float,
    )

    fixed_common_mask = read_hdf5_array(
        handle,
        "fixed_common_mask",
        dtype=float,
    ).astype(bool)

    annual_fields = OrderedDict()

    for variable_label, netcdf_name in VARIABLE_TO_NETCDF.items():
        annual_fields[variable_label] = read_hdf5_array(
            handle,
            netcdf_name,
            dtype=float,
        )

longitude = ((longitude + 180.0) % 360.0) - 180.0

if years.size == 0:
    raise RuntimeError("The annual NetCDF contains no years.")

period_mask = (
    (years >= YEAR_START)
    & (years <= YEAR_END)
)

if not np.any(period_mask):
    raise RuntimeError(
        f"No data were found between {YEAR_START} and {YEAR_END}."
    )

years = years[period_mask]

for variable_label in VARIABLE_ORDER:
    annual_fields[variable_label] = (
        annual_fields[variable_label][period_mask]
    )

expected_shape = (
    years.size,
    latitude.size,
    longitude.size,
)

for variable_label, values in annual_fields.items():
    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {variable_label}: {values.shape}; "
            f"expected {expected_shape}."
        )

if fixed_common_mask.shape != (
    latitude.size,
    longitude.size,
):
    raise ValueError(
        "The fixed common mask does not match the annual grid."
    )

print(
    f"Years: {int(years[0])}–{int(years[-1])} "
    f"({years.size} annual values)"
)
print(
    "Grid:",
    f"{latitude.size} latitude × {longitude.size} longitude"
)
print(
    "Fixed common cells:",
    f"{int(fixed_common_mask.sum()):,}"
)


# 8. BUILD SEA MASKS AND AREA WEIGHTS
# For a regular latitude–longitude grid, cell area is proportional to
# cos(latitude). The longitude-width factor is common and cancels in the
# weighted mean.
latitude_weights = np.cos(
    np.deg2rad(latitude)
)[:, None]

area_weights = np.broadcast_to(
    latitude_weights,
    fixed_common_mask.shape,
).astype(np.float64)

sea_masks = OrderedDict()

for sea_code, information in SEA_SECTORS.items():
    one_dimensional_longitude_mask = longitude_sector_mask(
        longitude,
        information["lon_min"],
        information["lon_max"],
    )

    sea_masks[sea_code] = (
        np.broadcast_to(
            one_dimensional_longitude_mask[None, :],
            fixed_common_mask.shape,
        )
        & fixed_common_mask
    )

    print(
        f"{sea_code}: {information['name']:<22} "
        f"common cells = {int(sea_masks[sea_code].sum()):4d}"
    )


def area_weighted_sector_series(annual_values, mask):
    """Area-weighted annual sea mean using the fixed spatial mask."""
    annual_values = np.asarray(annual_values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    output = np.full(
        annual_values.shape[0],
        np.nan,
        dtype=np.float64,
    )

    for year_index in range(annual_values.shape[0]):
        field = annual_values[year_index]

        valid = (
            mask
            & np.isfinite(field)
            & np.isfinite(area_weights)
        )

        if not np.any(valid):
            continue

        denominator = np.sum(area_weights[valid])

        if denominator <= 0.0:
            continue

        output[year_index] = (
            np.sum(
                field[valid]
                * area_weights[valid]
            )
            / denominator
        )

    return output


# 9. CALCULATE THE 52 SEA–VARIABLE TREND TESTS
trend_rows = []
annual_series_rows = []

print("\nRecomputing sea-wise Sen slopes, Hamed–Rao statistics and CIs...")

for variable_index, variable_label in enumerate(VARIABLE_ORDER):
    annual_values = annual_fields[variable_label]

    variable_rows = []

    for sea_index, sea_code in enumerate(SEA_ORDER):
        information = SEA_SECTORS[sea_code]
        mask = sea_masks[sea_code]

        annual_series = area_weighted_sector_series(
            annual_values,
            mask,
        )

        for year, value in zip(years, annual_series):
            annual_series_rows.append(
                {
                    "sea_code": sea_code,
                    "sea_name": information["name"],
                    "variable": variable_label,
                    "year": int(year),
                    "annual_anomaly_cm": value,
                }
            )

        valid_year_mask = (
            np.isfinite(annual_series)
            & np.isfinite(years)
        )

        selected_values = annual_series[valid_year_mask]
        selected_years = years[valid_year_mask]

        if selected_values.size < MIN_VALID_YEARS:
            test_result = hamed_rao_modified_mk(
                selected_values,
                selected_years,
            )

            slope_decade = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        else:
            test_result = hamed_rao_modified_mk(
                selected_values,
                selected_years,
                alpha=MK_ALPHA,
                maximum_lag=HAMED_RAO_MAX_LAG,
            )

            slope_decade = (
                sen_slope(
                    selected_values,
                    selected_years,
                )
                * 10.0
            )

            ci_lower, ci_upper = moving_block_bootstrap_ci(
                selected_values,
                selected_years,
                block_length=BOOTSTRAP_BLOCK_LENGTH_YEARS,
                number_of_bootstraps=BOOTSTRAP_REPLICATES,
                random_seed=(
                    BOOTSTRAP_SEED
                    + variable_index * 100
                    + sea_index
                ),
            )

        row = {
            "sea_code": sea_code,
            "sea_name": information["name"],
            "variable": variable_label,
            "sen_slope_cm_decade": slope_decade,
            "ci_lower_cm_decade": ci_lower,
            "ci_upper_cm_decade": ci_upper,
            **test_result,
            "valid_years": int(selected_values.size),
            "common_grid_cells": int(mask.sum()),
        }

        row["bootstrap_ci_excludes_zero"] = bool(
            np.isfinite(ci_lower)
            and np.isfinite(ci_upper)
            and (
                (ci_lower > 0.0)
                or (ci_upper < 0.0)
            )
        )

        variable_rows.append(row)

    variable_p_values = np.asarray(
        [row["raw_p"] for row in variable_rows],
        dtype=float,
    )

    variable_q_values = benjamini_hochberg_qvalues(
        variable_p_values
    )

    for row, q_value in zip(
        variable_rows,
        variable_q_values,
    ):
        row["fdr_q"] = q_value

        row["fdr_significant"] = bool(
            np.isfinite(q_value)
            and q_value < FDR_ALPHA
        )

        row["significance_direction"] = (
            "Increasing"
            if row["fdr_significant"]
            and row["sen_slope_cm_decade"] > 0.0
            else (
                "Decreasing"
                if row["fdr_significant"]
                and row["sen_slope_cm_decade"] < 0.0
                else "Not significant"
            )
        )

        trend_rows.append(row)


# 10. BUILD AND ORDER OUTPUT TABLES
trend_table = pd.DataFrame(trend_rows)
annual_series_table = pd.DataFrame(annual_series_rows)

sea_rank = {code: index for index, code in enumerate(SEA_ORDER)}
variable_rank = {
    variable: index
    for index, variable in enumerate(VARIABLE_ORDER)
}

trend_table["_sea_rank"] = trend_table["sea_code"].map(sea_rank)
trend_table["_variable_rank"] = trend_table["variable"].map(variable_rank)

trend_table = (
    trend_table
    .sort_values(
        ["_sea_rank", "_variable_rank"]
    )
    .drop(
        columns=["_sea_rank", "_variable_rank"]
    )
    .reset_index(drop=True)
)

annual_series_table["_sea_rank"] = (
    annual_series_table["sea_code"].map(sea_rank)
)
annual_series_table["_variable_rank"] = (
    annual_series_table["variable"].map(variable_rank)
)

annual_series_table = (
    annual_series_table
    .sort_values(
        ["_sea_rank", "_variable_rank", "year"]
    )
    .drop(
        columns=["_sea_rank", "_variable_rank"]
    )
    .reset_index(drop=True)
)

# Detailed numeric table — never replace numeric values with formatted text.
detailed_column_order = [
    "sea_code",
    "sea_name",
    "variable",
    "sen_slope_cm_decade",
    "ci_lower_cm_decade",
    "ci_upper_cm_decade",
    "mk_s",
    "mk_tau",
    "mk_variance_uncorrected",
    "standard_mk_z",
    "standard_mk_p",
    "hamed_rao_max_lag",
    "acf_confidence_limit",
    "significant_acf_lags",
    "significant_acf_values",
    "weighted_acf_sum",
    "hamed_rao_correction_factor",
    "mk_variance_corrected",
    "mann_kendall_z",
    "raw_p",
    "fdr_q",
    "fdr_significant",
    "significance_direction",
    "bootstrap_ci_excludes_zero",
    "valid_years",
    "common_grid_cells",
]

trend_table = trend_table[detailed_column_order]

publication_table = pd.DataFrame(
    {
        "Sea": (
            trend_table["sea_name"]
            + " ("
            + trend_table["sea_code"]
            + ")"
        ),
        "Variable": trend_table["variable"],
        "Sen slope (cm decade^-1)": trend_table[
            "sen_slope_cm_decade"
        ].map(
            lambda value: (
                "N/A"
                if not np.isfinite(value)
                else f"{value:.2f}"
            )
        ),
        "95% CI (cm decade^-1)": [
            (
                "N/A"
                if not (
                    np.isfinite(lower)
                    and np.isfinite(upper)
                )
                else f"[{lower:.2f}, {upper:.2f}]"
            )
            for lower, upper in zip(
                trend_table["ci_lower_cm_decade"],
                trend_table["ci_upper_cm_decade"],
            )
        ],
        "Mann-Kendall Z": trend_table[
            "mann_kendall_z"
        ].map(
            lambda value: (
                "N/A"
                if not np.isfinite(value)
                else f"{value:.2f}"
            )
        ),
        "Raw p": trend_table["raw_p"].map(
            p_for_publication
        ),
        "FDR p": trend_table["fdr_q"].map(
            p_for_publication
        ),
        "Significant": trend_table[
            "fdr_significant"
        ].map(yes_no),
    }
)


# 11. OPTIONAL AUDIT AGAINST THE ORIGINAL FIGURE 11 SECTOR CSV
audit_table = pd.DataFrame()

if ORIGINAL_SECTOR_CSV is not None:
    print("\nOriginal Figure 11 sector CSV found:")
    print(ORIGINAL_SECTOR_CSV)

    original_table = pd.read_csv(ORIGINAL_SECTOR_CSV)

    required_original_columns = {
        "variable",
        "sea_code",
        "sen_slope_cm_decade",
        "ci_lower_cm_decade",
        "ci_upper_cm_decade",
        "hamed_rao_p",
        "hamed_rao_factor",
        "fdr_q",
        "fdr_significant",
    }

    missing_original_columns = (
        required_original_columns
        - set(original_table.columns)
    )

    if missing_original_columns:
        print(
            "Original-audit comparison skipped because columns are missing:",
            sorted(missing_original_columns),
        )
    else:
        original_selected = original_table[
            [
                "variable",
                "sea_code",
                "sen_slope_cm_decade",
                "ci_lower_cm_decade",
                "ci_upper_cm_decade",
                "hamed_rao_p",
                "hamed_rao_factor",
                "fdr_q",
                "fdr_significant",
            ]
        ].rename(
            columns={
                "sen_slope_cm_decade": "original_slope_cm_decade",
                "ci_lower_cm_decade": "original_ci_lower_cm_decade",
                "ci_upper_cm_decade": "original_ci_upper_cm_decade",
                "hamed_rao_p": "original_raw_p",
                "hamed_rao_factor": "original_correction_factor",
                "fdr_q": "original_fdr_q",
                "fdr_significant": "original_significant",
            }
        )

        new_selected = trend_table[
            [
                "variable",
                "sea_code",
                "sen_slope_cm_decade",
                "ci_lower_cm_decade",
                "ci_upper_cm_decade",
                "raw_p",
                "hamed_rao_correction_factor",
                "fdr_q",
                "fdr_significant",
                "mann_kendall_z",
            ]
        ].rename(
            columns={
                "sen_slope_cm_decade": "recomputed_slope_cm_decade",
                "ci_lower_cm_decade": "recomputed_ci_lower_cm_decade",
                "ci_upper_cm_decade": "recomputed_ci_upper_cm_decade",
                "raw_p": "recomputed_raw_p",
                "hamed_rao_correction_factor": (
                    "recomputed_correction_factor"
                ),
                "fdr_q": "recomputed_fdr_q",
                "fdr_significant": "recomputed_significant",
            }
        )

        audit_table = original_selected.merge(
            new_selected,
            on=["variable", "sea_code"],
            how="outer",
            validate="one_to_one",
        )

        audit_table["slope_difference"] = (
            audit_table["recomputed_slope_cm_decade"]
            - audit_table["original_slope_cm_decade"]
        )

        audit_table["ci_lower_difference"] = (
            audit_table["recomputed_ci_lower_cm_decade"]
            - audit_table["original_ci_lower_cm_decade"]
        )

        audit_table["ci_upper_difference"] = (
            audit_table["recomputed_ci_upper_cm_decade"]
            - audit_table["original_ci_upper_cm_decade"]
        )

        audit_table["significance_changed"] = (
            audit_table["recomputed_significant"].astype("boolean")
            != audit_table["original_significant"].astype("boolean")
        )

        audit_table["_sea_rank"] = audit_table[
            "sea_code"
        ].map(sea_rank)
        audit_table["_variable_rank"] = audit_table[
            "variable"
        ].map(variable_rank)

        audit_table = (
            audit_table
            .sort_values(
                ["_sea_rank", "_variable_rank"]
            )
            .drop(
                columns=["_sea_rank", "_variable_rank"]
            )
            .reset_index(drop=True)
        )


# 12. SAVE CSV OUTPUTS
trend_table.to_csv(
    OUT_DETAILED_CSV,
    index=False,
)

publication_table.to_csv(
    OUT_PUBLICATION_CSV,
    index=False,
    encoding="utf-8-sig",
)

annual_series_table.to_csv(
    OUT_ANNUAL_SERIES_CSV,
    index=False,
)

if not audit_table.empty:
    audit_table.to_csv(
        OUT_AUDIT_CSV,
        index=False,
    )


# 13. SAVE A FORMATTED EXCEL WORKBOOK
with pd.ExcelWriter(
    OUT_EXCEL,
    engine="openpyxl",
) as writer:
    publication_table.to_excel(
        writer,
        sheet_name="Table S4",
        index=False,
        startrow=2,
    )

    trend_table.to_excel(
        writer,
        sheet_name="Detailed statistics",
        index=False,
    )

    annual_series_table.to_excel(
        writer,
        sheet_name="Annual sea series",
        index=False,
    )

    if not audit_table.empty:
        audit_table.to_excel(
            writer,
            sheet_name="Audit vs original",
            index=False,
        )

    workbook = writer.book

    table_sheet = workbook["Table S4"]
    table_sheet["A1"] = (
        "Table S4. Complete sea-wise trend statistics during 2008–2025"
    )
    table_sheet.merge_cells("A1:H1")

    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

    title_fill = PatternFill(
        fill_type="solid",
        fgColor="17365D",
    )
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="2F75B5",
    )
    significant_fill = PatternFill(
        fill_type="solid",
        fgColor="E2F0D9",
    )
    thin_border = Border(
        left=Side(style="thin", color="BFBFBF"),
        right=Side(style="thin", color="BFBFBF"),
        top=Side(style="thin", color="BFBFBF"),
        bottom=Side(style="thin", color="BFBFBF"),
    )

    table_sheet["A1"].fill = title_fill
    table_sheet["A1"].font = Font(
        bold=True,
        color="FFFFFF",
        size=14,
    )
    table_sheet["A1"].alignment = Alignment(
        horizontal="center",
        vertical="center",
        wrap_text=True,
    )
    table_sheet.row_dimensions[1].height = 28

    for cell in table_sheet[3]:
        cell.fill = header_fill
        cell.font = Font(
            bold=True,
            color="FFFFFF",
        )
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )
        cell.border = thin_border

    table_sheet.freeze_panes = "A4"
    table_sheet.auto_filter.ref = (
        f"A3:H{table_sheet.max_row}"
    )

    column_widths = {
        "A": 30,
        "B": 18,
        "C": 23,
        "D": 25,
        "E": 18,
        "F": 13,
        "G": 13,
        "H": 13,
    }

    for column_letter, width in column_widths.items():
        table_sheet.column_dimensions[column_letter].width = width

    for row_index in range(4, table_sheet.max_row + 1):
        for column_index in range(1, 9):
            cell = table_sheet.cell(
                row=row_index,
                column=column_index,
            )
            cell.border = thin_border
            cell.alignment = Alignment(
                horizontal=(
                    "left"
                    if column_index in {1, 2}
                    else "center"
                ),
                vertical="center",
            )

        if table_sheet.cell(row=row_index, column=8).value == "Yes":
            for column_index in range(1, 9):
                table_sheet.cell(
                    row=row_index,
                    column=column_index,
                ).fill = significant_fill

    # Add a compact methods note below the table.
    note_row = table_sheet.max_row + 2
    table_sheet.cell(row=note_row, column=1).value = (
        "Notes: Sen slopes use area-weighted deseasonalized annual anomalies. "
        "The Hamed–Rao correction uses the absolute weighted rank-ACF sum; "
        "FDR is applied separately across 13 seas for each variable. "
        "Significant = FDR q < 0.05."
    )
    table_sheet.merge_cells(
        start_row=note_row,
        start_column=1,
        end_row=note_row,
        end_column=8,
    )
    table_sheet.cell(
        row=note_row,
        column=1,
    ).alignment = Alignment(
        wrap_text=True,
        vertical="top",
    )
    table_sheet.cell(
        row=note_row,
        column=1,
    ).font = Font(
        italic=True,
        size=9,
    )
    table_sheet.row_dimensions[note_row].height = 42

    # Improve widths on supporting sheets.
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]

        if sheet_name == "Table S4":
            continue

        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            maximum_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells[:100]:
                value = "" if cell.value is None else str(cell.value)
                maximum_length = max(maximum_length, len(value))

            worksheet.column_dimensions[column_letter].width = min(
                max(maximum_length + 2, 12),
                34,
            )


# 14. QUALITY-CONTROL CHECKS
if len(trend_table) != 52:
    raise RuntimeError(
        f"Expected 52 sea–variable rows, obtained {len(trend_table)}."
    )

if trend_table[[
    "sen_slope_cm_decade",
    "mann_kendall_z",
    "raw_p",
    "fdr_q",
]].isna().any().any():
    missing_rows = trend_table[
        trend_table[[
            "sen_slope_cm_decade",
            "mann_kendall_z",
            "raw_p",
            "fdr_q",
        ]].isna().any(axis=1)
    ]

    print("\nWARNING: Some rows contain N/A statistics:")
    print(
        missing_rows[
            ["sea_code", "variable", "valid_years"]
        ].to_string(index=False)
    )

if (
    trend_table["hamed_rao_correction_factor"]
    .dropna()
    .lt(1.0 - 1e-12)
    .any()
):
    raise RuntimeError(
        "A corrected Hamed–Rao factor below 1 was produced, "
        "which should not occur with ABS(weighted_ACF_sum)."
    )

if (
    trend_table["raw_p"].dropna().lt(0.0).any()
    or trend_table["raw_p"].dropna().gt(1.0).any()
    or trend_table["fdr_q"].dropna().lt(0.0).any()
    or trend_table["fdr_q"].dropna().gt(1.0).any()
):
    raise RuntimeError("A p or q value lies outside [0, 1].")

# The slope and bootstrap CI should remain numerically identical to the
# original output because only the Hamed–Rao variance correction changed.
maximum_slope_difference = np.nan
maximum_ci_difference = np.nan
changed_significance_count = 0

if not audit_table.empty:
    maximum_slope_difference = float(
        np.nanmax(np.abs(audit_table["slope_difference"]))
    )

    maximum_ci_difference = float(
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        audit_table["ci_lower_difference"].to_numpy(dtype=float),
                        audit_table["ci_upper_difference"].to_numpy(dtype=float),
                    ]
                )
            )
        )
    )

    changed_significance_count = int(
        audit_table["significance_changed"]
        .fillna(False)
        .sum()
    )


# 15. PRINT AND SAVE SUMMARY
changed_rows_text = "None"

if not audit_table.empty:
    changed_rows = audit_table[
        audit_table["significance_changed"].fillna(False)
    ]

    if not changed_rows.empty:
        changed_rows_text = "\n".join(
            f"• {row.sea_code} — {row.variable}: "
            f"original={row.original_significant}, "
            f"recomputed={row.recomputed_significant}"
            for row in changed_rows.itertuples()
        )

summary_lines = [
    "=" * 94,
    "TABLE S4 — COMPLETE SEA-WISE TREND STATISTICS, RECOMPUTED",
    "=" * 94,
    "",
    f"Annual input       : {ANNUAL_NC}",
    f"Period             : {YEAR_START}–{YEAR_END}",
    f"Annual values      : {len(years)}",
    f"Fixed common cells : {int(fixed_common_mask.sum()):,}",
    f"Sea sectors        : {len(SEA_SECTORS)}",
    f"Variables          : {len(VARIABLE_ORDER)}",
    f"Trend tests        : {len(trend_table)}",
    "",
    "Corrected Hamed–Rao method:",
    "• Sen-slope detrending of each sea-wise annual series.",
    "• Rank autocorrelation and approximate 95% white-noise bounds.",
    "• Only autocorrelation coefficients outside the bounds contribute.",
    "• Variance factor uses ABS(weighted autocorrelation sum).",
    "• No artificial 1e-6 lower clipping value is used.",
    "",
    f"Hamed–Rao maximum lag : {HAMED_RAO_MAX_LAG}",
    f"FDR scope              : {FDR_SCOPE}",
    f"FDR alpha              : {FDR_ALPHA}",
    f"Bootstrap replicates   : {BOOTSTRAP_REPLICATES}",
    f"Bootstrap block length : {BOOTSTRAP_BLOCK_LENGTH_YEARS} years",
    f"Bootstrap base seed    : {BOOTSTRAP_SEED}",
    "",
    f"Detailed CSV      : {OUT_DETAILED_CSV}",
    f"Publication CSV   : {OUT_PUBLICATION_CSV}",
    f"Excel workbook    : {OUT_EXCEL}",
    f"Annual-series CSV : {OUT_ANNUAL_SERIES_CSV}",
    f"Audit CSV         : {OUT_AUDIT_CSV if not audit_table.empty else 'Not created'}",
    "",
    f"Maximum slope difference versus original : {maximum_slope_difference}",
    f"Maximum CI difference versus original    : {maximum_ci_difference}",
    f"Significance classifications changed     : {changed_significance_count}",
    "",
    "Changed significance rows:",
    changed_rows_text,
]

OUT_SUMMARY_TXT.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\n" + "\n".join(summary_lines))

print("\nPublication table preview:\n")
print(publication_table.to_string(index=False))

print("\nCompleted successfully.")



# DATA PRODUCT GROUP 15: TABLE S5 CORRELATION / REGRESSION DIAGNOSTICS
# Source: so_sealevel_paper_fig.py
# File name: Table_S5_correlation_regression_diagnostics_RECOMPUTED_detailed.csv
# File name: Table_S5_correlation_regression_diagnostics_RECOMPUTED_publication.csv
# File name: Table_S5_regression_recomputation_audit.csv

# TABLE S5 — SEA-WISE CORRELATION AND REGRESSION DIAGNOSTICS
# Period: 2008–2025
#
# Why this recomputation is required
# ----------------------------------
# The existing Figure 13 regression-coefficient CSV files contain:
#   beta, raw p, FDR p/q, adjusted R², VIF and n
#
# They do NOT contain 95% confidence intervals for standardized beta.
# This script therefore refits the exact sea-wise standardized OLS models
# using HAC/Newey–West covariance with maxlags=1, which reproduces the
# original Figure 13 p values.
#
# Responses and predictors
# ------------------------
# Thermosteric:
#   Qnet*, WSC, MLD, SIC, AAO
#
# Halosteric:
#   FWC, SIC, Snowfall, MLD, AAO
#
# Correlation values are merged from the existing Figure 13 correlation CSVs.
#
# Outputs
# -------
# Table_S5_correlation_regression_diagnostics_RECOMPUTED_detailed.csv
# Table_S5_correlation_regression_diagnostics_RECOMPUTED_publication.csv
# Table_S5_regression_recomputation_audit.csv
# Table_S5_recomputation_summary.txt


# 0. INSTALL MISSING PACKAGES
import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "statsmodels": "statsmodels",
}

missing_packages = [
    pip_name
    for import_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]

if missing_packages:
    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )


# 1. IMPORTS
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# 3. PATHS
INPUT_DIRECTORY = Path(
    "/content/drive/MyDrive/SAM_Thesis/paper2"
)

OUTPUT_DIRECTORY = INPUT_DIRECTORY
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

TIMESERIES_FILE = (
    INPUT_DIRECTORY
    / "Fig13_sector_driver_timeseries_LOW_RAM.csv"
)

THERMO_CORRELATION_FILE = (
    INPUT_DIRECTORY
    / "Fig13b_thermosteric_driver_correlations.csv"
)

HALO_CORRELATION_FILE = (
    INPUT_DIRECTORY
    / "Fig13c_halosteric_driver_correlations.csv"
)

ORIGINAL_THERMO_REGRESSION_FILE = (
    INPUT_DIRECTORY
    / "Fig13d_thermosteric_regression_coefficients.csv"
)

ORIGINAL_HALO_REGRESSION_FILE = (
    INPUT_DIRECTORY
    / "Fig13e_halosteric_regression_coefficients.csv"
)

OUT_DETAILED_CSV = (
    OUTPUT_DIRECTORY
    / "Table_S5_correlation_regression_diagnostics_RECOMPUTED_detailed.csv"
)

OUT_PUBLICATION_CSV = (
    OUTPUT_DIRECTORY
    / "Table_S5_correlation_regression_diagnostics_RECOMPUTED_publication.csv"
)

OUT_AUDIT_CSV = (
    OUTPUT_DIRECTORY
    / "Table_S5_regression_recomputation_audit.csv"
)

OUT_SUMMARY_TXT = (
    OUTPUT_DIRECTORY
    / "Table_S5_recomputation_summary.txt"
)


# 4. SETTINGS
YEAR_START = 2008
YEAR_END = 2025

HAC_MAXLAGS = 1
CONFIDENCE_LEVEL = 0.95
FDR_ALPHA = 0.05

# Figure 13 applied Benjamini–Hochberg correction over all 65 regression
# coefficients separately for each response model.
FDR_SCOPE = "within_response_all_65_coefficients"

SEA_ORDER = [
    "WED",
    "KHV",
    "RLS",
    "LAZ",
    "COS",
    "COO",
    "DAV",
    "MAW",
    "DUR",
    "SOM",
    "ROS",
    "AMU",
    "BEL",
]

SEA_NAMES = {
    "WED": "Weddell Sea",
    "KHV": "King Haakon VII Sea",
    "RLS": "Riiser-Larsen Sea",
    "LAZ": "Lazarev Sea",
    "COS": "Cosmonauts Sea",
    "COO": "Cooperation Sea",
    "DAV": "Davis Sea",
    "MAW": "Mawson Sea",
    "DUR": "D'Urville Sea",
    "SOM": "Somov Sea",
    "ROS": "Ross Sea",
    "AMU": "Amundsen Sea",
    "BEL": "Bellingshausen Sea",
}

MODEL_DEFINITIONS = {
    "Thermosteric": {
        "predictors": [
            "Qnet*",
            "WSC",
            "MLD",
            "SIC",
            "AAO",
        ],
        "correlation_file": THERMO_CORRELATION_FILE,
        "original_regression_file": ORIGINAL_THERMO_REGRESSION_FILE,
    },
    "Halosteric": {
        "predictors": [
            "FWC",
            "SIC",
            "Snowfall",
            "MLD",
            "AAO",
        ],
        "correlation_file": HALO_CORRELATION_FILE,
        "original_regression_file": ORIGINAL_HALO_REGRESSION_FILE,
    },
}


# 5. HELPERS
def verify_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{file_path}"
        )


def zscore_sample(values):
    """
    Standardize with sample standard deviation (ddof=1).
    The standardized regression coefficients are unaffected by choosing
    ddof=0 versus ddof=1 when the same convention is used for all columns,
    but ddof=1 is used here explicitly.
    """
    values = np.asarray(values, dtype=float)

    mean_value = np.nanmean(values)
    standard_deviation = np.nanstd(
        values,
        ddof=1,
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0.0
    ):
        raise ValueError(
            "A response or predictor has zero or invalid variance."
        )

    return (
        values - mean_value
    ) / standard_deviation


def calculate_vif(standardized_predictors):
    """
    Calculate one VIF value for each predictor.
    """
    matrix = np.asarray(
        standardized_predictors,
        dtype=float,
    )

    return np.array(
        [
            variance_inflation_factor(
                matrix,
                predictor_index,
            )
            for predictor_index in range(
                matrix.shape[1]
            )
        ],
        dtype=float,
    )


def format_p_value(value):
    if not np.isfinite(value):
        return "N/A"

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def significance_symbol(q_value):
    if not np.isfinite(q_value):
        return ""

    if q_value < 0.001:
        return "***"

    if q_value < 0.01:
        return "**"

    if q_value < 0.05:
        return "*"

    return ""


# 6. VERIFY INPUTS
verify_file(TIMESERIES_FILE)

for response, specification in MODEL_DEFINITIONS.items():
    verify_file(specification["correlation_file"])
    verify_file(specification["original_regression_file"])


# 7. READ INPUTS
timeseries = pd.read_csv(
    TIMESERIES_FILE,
    parse_dates=["time"],
)

timeseries = timeseries.loc[
    (
        timeseries["time"].dt.year
        >= YEAR_START
    )
    & (
        timeseries["time"].dt.year
        <= YEAR_END
    )
].copy()

if len(timeseries) != 216:
    print(
        "Warning: Expected 216 monthly records, "
        f"but found {len(timeseries)}."
    )

print(
    "Analysis period:",
    timeseries["time"].min(),
    "to",
    timeseries["time"].max(),
)

print(
    "Monthly records:",
    len(timeseries),
)


# 8. RECOMPUTE STANDARDIZED REGRESSION MODELS
recomputed_rows = []
audit_rows = []

for response, specification in MODEL_DEFINITIONS.items():

    predictors = specification["predictors"]

    correlation_table = pd.read_csv(
        specification["correlation_file"]
    )

    original_regression = pd.read_csv(
        specification["original_regression_file"]
    )

    # Fast lookup tables
    correlation_lookup = (
        correlation_table
        .set_index(
            [
                "sea",
                "driver",
            ]
        )
    )

    original_lookup = (
        original_regression
        .set_index(
            [
                "sea",
                "predictor",
            ]
        )
    )

    response_rows = []

    for sea in SEA_ORDER:

        response_column = (
            f"{sea}|{response}"
        )

        predictor_columns = {
            predictor: (
                f"{sea}|{predictor}"
            )
            for predictor in predictors
        }

        required_columns = [
            response_column,
            *predictor_columns.values(),
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in timeseries.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing columns for {sea} {response}:\n"
                + "\n".join(missing_columns)
            )

        model_data = pd.DataFrame(
            {
                "response": timeseries[
                    response_column
                ],
                **{
                    predictor: timeseries[column]
                    for predictor, column
                    in predictor_columns.items()
                },
            }
        )

        model_data = (
            model_data
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        number_of_observations = len(
            model_data
        )

        minimum_required = (
            len(predictors) + 3
        )

        if (
            number_of_observations
            < minimum_required
        ):
            raise RuntimeError(
                f"Insufficient observations for "
                f"{sea} {response}: "
                f"{number_of_observations}"
            )

        standardized_response = zscore_sample(
            model_data["response"].values
        )

        standardized_predictors = pd.DataFrame(
            {
                predictor: zscore_sample(
                    model_data[predictor].values
                )
                for predictor in predictors
            },
            index=model_data.index,
        )

        design_matrix = sm.add_constant(
            standardized_predictors,
            has_constant="add",
        )

        # The original Figure 13 p values are reproduced using
        # HAC/Newey–West covariance with maxlags=1.
        fitted_model = sm.OLS(
            standardized_response,
            design_matrix,
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": HAC_MAXLAGS,
                "use_correction": False,
            },
        )

        confidence_intervals = (
            fitted_model.conf_int(
                alpha=1.0 - CONFIDENCE_LEVEL
            )
        )

        vif_values = calculate_vif(
            standardized_predictors.values
        )

        for predictor_index, predictor in enumerate(
            predictors
        ):

            coefficient_index = (
                predictor_index + 1
            )

            beta_value = float(
                fitted_model.params.iloc[
                    coefficient_index
                ]
            )

            beta_ci_lower = float(
                confidence_intervals.iloc[
                    coefficient_index,
                    0,
                ]
            )

            beta_ci_upper = float(
                confidence_intervals.iloc[
                    coefficient_index,
                    1,
                ]
            )

            regression_p_raw = float(
                fitted_model.pvalues.iloc[
                    coefficient_index
                ]
            )

            correlation_key = (
                sea,
                predictor,
            )

            if correlation_key not in correlation_lookup.index:
                raise KeyError(
                    f"No correlation result for "
                    f"{sea}, {response}, {predictor}"
                )

            correlation_row = (
                correlation_lookup.loc[
                    correlation_key
                ]
            )

            if isinstance(
                correlation_row,
                pd.DataFrame,
            ):
                correlation_row = (
                    correlation_row.iloc[0]
                )

            correlation_r = float(
                correlation_row["r"]
            )

            correlation_p_raw = float(
                correlation_row["p_raw"]
            )

            correlation_p_fdr = float(
                correlation_row["p_fdr"]
            )

            row = {
                "sea_code": sea,
                "sea_name": SEA_NAMES[sea],
                "response": response,
                "predictor": predictor,
                "n": int(
                    number_of_observations
                ),
                "correlation_r": correlation_r,
                "correlation_p_raw": correlation_p_raw,
                "correlation_p_fdr": correlation_p_fdr,
                "standardized_beta": beta_value,
                "beta_ci95_lower": beta_ci_lower,
                "beta_ci95_upper": beta_ci_upper,
                "regression_p_raw": regression_p_raw,
                "vif": float(
                    vif_values[
                        predictor_index
                    ]
                ),
                "adjusted_r2": float(
                    fitted_model.rsquared_adj
                ),
            }

            response_rows.append(row)

            original_key = (
                sea,
                predictor,
            )

            if original_key not in original_lookup.index:
                raise KeyError(
                    f"No original regression result for "
                    f"{sea}, {response}, {predictor}"
                )

            original_row = (
                original_lookup.loc[
                    original_key
                ]
            )

            if isinstance(
                original_row,
                pd.DataFrame,
            ):
                original_row = (
                    original_row.iloc[0]
                )

            audit_rows.append(
                {
                    "sea_code": sea,
                    "response": response,
                    "predictor": predictor,
                    "original_beta": float(
                        original_row["beta"]
                    ),
                    "recomputed_beta": beta_value,
                    "absolute_beta_difference": abs(
                        float(
                            original_row["beta"]
                        )
                        - beta_value
                    ),
                    "original_p_raw": float(
                        original_row["p_raw"]
                    ),
                    "recomputed_p_raw": regression_p_raw,
                    "absolute_p_difference": abs(
                        float(
                            original_row["p_raw"]
                        )
                        - regression_p_raw
                    ),
                    "original_adjusted_r2": float(
                        original_row[
                            "adjusted_r2"
                        ]
                    ),
                    "recomputed_adjusted_r2": float(
                        fitted_model.rsquared_adj
                    ),
                    "absolute_adjusted_r2_difference": abs(
                        float(
                            original_row[
                                "adjusted_r2"
                            ]
                        )
                        - float(
                            fitted_model.rsquared_adj
                        )
                    ),
                    "original_vif": float(
                        original_row["vif"]
                    ),
                    "recomputed_vif": float(
                        vif_values[
                            predictor_index
                        ]
                    ),
                    "absolute_vif_difference": abs(
                        float(
                            original_row["vif"]
                        )
                        - float(
                            vif_values[
                                predictor_index
                            ]
                        )
                    ),
                }
            )

    response_table = pd.DataFrame(
        response_rows
    )

    # FDR across all 13 seas × 5 predictors = 65 tests,
    # separately for each response.
    reject, q_values, _, _ = multipletests(
        response_table[
            "regression_p_raw"
        ].to_numpy(
            dtype=float
        ),
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )

    response_table[
        "regression_p_fdr"
    ] = q_values

    response_table[
        "significant_fdr"
    ] = reject

    response_table[
        "significance_symbol"
    ] = [
        significance_symbol(value)
        for value in q_values
    ]

    recomputed_rows.append(
        response_table
    )


# 9. COMBINE AND ORDER RESULTS
detailed_table = pd.concat(
    recomputed_rows,
    ignore_index=True,
)

sea_order_mapping = {
    sea: index
    for index, sea in enumerate(
        SEA_ORDER
    )
}

response_order_mapping = {
    "Thermosteric": 0,
    "Halosteric": 1,
}

predictor_order_mapping = {
    (
        response,
        predictor,
    ): predictor_index
    for response, specification
    in MODEL_DEFINITIONS.items()
    for predictor_index, predictor
    in enumerate(
        specification["predictors"]
    )
}

detailed_table[
    "_sea_order"
] = detailed_table[
    "sea_code"
].map(
    sea_order_mapping
)

detailed_table[
    "_response_order"
] = detailed_table[
    "response"
].map(
    response_order_mapping
)

detailed_table[
    "_predictor_order"
] = [
    predictor_order_mapping[
        (
            response,
            predictor,
        )
    ]
    for response, predictor
    in zip(
        detailed_table["response"],
        detailed_table["predictor"],
    )
]

detailed_table = (
    detailed_table
    .sort_values(
        [
            "_sea_order",
            "_response_order",
            "_predictor_order",
        ]
    )
    .drop(
        columns=[
            "_sea_order",
            "_response_order",
            "_predictor_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# 10. PUBLICATION-READY TABLE
publication_table = pd.DataFrame(
    {
        "Sea": [
            f"{name} ({code})"
            for name, code
            in zip(
                detailed_table[
                    "sea_name"
                ],
                detailed_table[
                    "sea_code"
                ],
            )
        ],
        "Response": detailed_table[
            "response"
        ],
        "Predictor": detailed_table[
            "predictor"
        ],
        "Correlation (r)": [
            f"{value:.2f}"
            for value in detailed_table[
                "correlation_r"
            ]
        ],
        "Standardized beta": [
            f"{value:.2f}"
            for value in detailed_table[
                "standardized_beta"
            ]
        ],
        "95% CI": [
            (
                f"[{lower:.2f}, "
                f"{upper:.2f}]"
            )
            for lower, upper
            in zip(
                detailed_table[
                    "beta_ci95_lower"
                ],
                detailed_table[
                    "beta_ci95_upper"
                ],
            )
        ],
        # Use FDR-adjusted p/q in the publication table because Figure 13
        # interpreted significance after BH correction.
        "FDR-adjusted p (q)": [
            format_p_value(value)
            for value in detailed_table[
                "regression_p_fdr"
            ]
        ],
        "VIF": [
            f"{value:.2f}"
            for value in detailed_table[
                "vif"
            ]
        ],
        "Adjusted R2": [
            f"{value:.2f}"
            for value in detailed_table[
                "adjusted_r2"
            ]
        ],
    }
)


# 11. SAVE OUTPUTS
detailed_table.to_csv(
    OUT_DETAILED_CSV,
    index=False,
)

publication_table.to_csv(
    OUT_PUBLICATION_CSV,
    index=False,
    encoding="utf-8-sig",
)

audit_table = pd.DataFrame(
    audit_rows
)

audit_table.to_csv(
    OUT_AUDIT_CSV,
    index=False,
)


# 12. AUDIT AND VALIDATION
maximum_beta_difference = float(
    audit_table[
        "absolute_beta_difference"
    ].max()
)

maximum_p_difference = float(
    audit_table[
        "absolute_p_difference"
    ].max()
)

maximum_adjusted_r2_difference = float(
    audit_table[
        "absolute_adjusted_r2_difference"
    ].max()
)

maximum_vif_difference = float(
    audit_table[
        "absolute_vif_difference"
    ].max()
)

if maximum_p_difference > 1.0e-8:
    raise RuntimeError(
        "The recomputed HAC p values do not match the "
        "original Figure 13 outputs closely enough."
    )

if maximum_adjusted_r2_difference > 1.0e-8:
    raise RuntimeError(
        "The recomputed adjusted R² values do not match "
        "the original Figure 13 outputs."
    )

if maximum_vif_difference > 1.0e-8:
    raise RuntimeError(
        "The recomputed VIF values do not match "
        "the original Figure 13 outputs."
    )

summary_lines = [
    "=" * 92,
    "TABLE S5 — CORRELATION AND REGRESSION DIAGNOSTICS, RECOMPUTED",
    "=" * 92,
    "",
    f"Period                       : {YEAR_START}–{YEAR_END}",
    f"Monthly records              : {len(timeseries)}",
    f"Seas                         : {len(SEA_ORDER)}",
    f"Responses                    : {len(MODEL_DEFINITIONS)}",
    f"Predictors per response      : 5",
    f"Publication rows             : {len(publication_table)}",
    "",
    "Regression method:",
    "• Response and predictors standardized within each sea.",
    "• Multiple OLS regression with an intercept.",
    f"• HAC/Newey–West covariance with maxlags={HAC_MAXLAGS}.",
    f"• {int(CONFIDENCE_LEVEL * 100)}% HAC confidence intervals for standardized beta.",
    "• VIF calculated from the standardized predictor matrix.",
    "• Adjusted R² repeated for all predictors in each sea-response model.",
    "• Benjamini–Hochberg FDR applied across all 65 coefficients",
    "  separately for thermosteric and halosteric models.",
    "",
    "Correlation:",
    "• Pearson r was merged from the existing Figure 13 correlation outputs.",
    "",
    "Audit versus original Figure 13 regression CSVs:",
    f"Maximum absolute beta difference        : {maximum_beta_difference:.12g}",
    f"Maximum absolute raw-p difference       : {maximum_p_difference:.12g}",
    f"Maximum absolute adjusted-R² difference : {maximum_adjusted_r2_difference:.12g}",
    f"Maximum absolute VIF difference         : {maximum_vif_difference:.12g}",
    "",
    "Outputs:",
    f"Detailed CSV    : {OUT_DETAILED_CSV}",
    f"Publication CSV : {OUT_PUBLICATION_CSV}",
    f"Audit CSV       : {OUT_AUDIT_CSV}",
]

OUT_SUMMARY_TXT.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\n".join(summary_lines))

print("\nPublication-table preview:")
print(
    publication_table
    .head(20)
    .to_string(
        index=False
    )
)
