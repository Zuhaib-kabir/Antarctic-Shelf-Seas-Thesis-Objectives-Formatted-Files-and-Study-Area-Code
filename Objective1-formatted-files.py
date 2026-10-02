#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

%pip install -q numpy pandas xarray "dask[array]" netCDF4 h5netcdf scipy gsw



"""
Build the five derived input files used by the Antarctic sea-ice analyses.

Outputs
-------
1. Argo_profile_OHC_0_100m.csv
2. OHC_0_100m_monthly_anomalies_local_objective_analysis_2008_2025.nc
3. SIC_seawise_monthly_2008_2025_LOW_RAM.csv
4. Argo_monthly_OHC_0_100m_GJ_m2.csv
5. Argo_monthly_OHC_profile_counts.csv

The file is intentionally divided into five clearly marked output sections so
it can be uploaded to GitHub as the complete formatting/processing record.

Colab installation (run once before this script if required):
    %pip install -q numpy pandas xarray "dask[array]" netCDF4 h5netcdf scipy gsw

Scientific definitions
----------------------
* Analysis period: January 2008 through December 2025 (216 months).
* OHC: integral of rho * cp0 * max(CT - CT_freezing, 0) over 0–100 dbar,
  reported in GJ m-2. TEOS-10 is used through the gsw package.
* Local objective analysis: Gaussian covariance C(d)=exp[-(d/L)^2], with
  source-bin, search-radius, observation-error, support, and mapping-error
  limits recorded in the output NetCDF attributes.
* SIC: cosine-latitude-weighted mean within the 13 fixed Antarctic marginal-
  sea longitude sectors, reported as percent.
* Missing OHC months are never temporally interpolated.
"""

from pathlib import Path
import gc
import warnings

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, num2date
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")



# COMMON PATHS, SETTINGS AND HELPERS
DATA_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Data")
FIG_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Fig")
OHC_DIR = FIG_DIR / "Fig5_OHC_supplementary"
OI_DIR = FIG_DIR / "P1_Corrected/Fig4_5/new_correlation_significance_files"

ARGO_SOURCE_NC = DATA_DIR / "argo_SO_profiles_2001_2025_cleaned_gridded.nc"
SIC_SOURCE_NC = DATA_DIR / "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"

ARGO_PROFILE_OHC_CSV = OHC_DIR / "Argo_profile_OHC_0_100m.csv"
OHC_OI_MONTHLY_NC = OI_DIR / (
    "OHC_0_100m_monthly_anomalies_local_objective_analysis_2008_2025.nc"
)
SIC_SEAWISE_CSV = DATA_DIR / "SIC_seawise_monthly_2008_2025_LOW_RAM.csv"
ARGO_MONTHLY_OHC_CSV = OHC_DIR / "Argo_monthly_OHC_0_100m_GJ_m2.csv"
ARGO_MONTHLY_COUNTS_CSV = OHC_DIR / "Argo_monthly_OHC_profile_counts.csv"

START_YEAR = 2008
END_YEAR = 2025
START_TIME = pd.Timestamp(f"{START_YEAR}-01-01")
END_TIME = pd.Timestamp(f"{END_YEAR}-12-31")
FULL_MONTH_INDEX = pd.date_range(
    f"{START_YEAR}-01-01",
    f"{END_YEAR}-12-01",
    freq="MS",
)

# Set any item to False when only selected products need rebuilding.
BUILD_PROFILE_OHC_CSV = True
BUILD_OHC_OI_NETCDF = True
BUILD_SIC_SEAWISE_CSV = True
BUILD_MONTHLY_OHC_CSV = True
BUILD_MONTHLY_PROFILE_COUNTS_CSV = True

# OHC definition and profile-level quality requirements.
CP0 = 3991.86795711963  # TEOS-10 reference heat capacity, J kg-1 K-1
OHC_PRESSURE_GRID_DBAR = np.arange(0.0, 101.0, 1.0)
MAX_SHALLOWEST_VALID_DBAR = 10.0
MIN_DEEPEST_VALID_DBAR = 90.0
MIN_VALID_LEVELS_OHC = 8

# A sampled sea-month requires at least this many accepted Argo profiles.
MIN_ARGO_PROFILES_PER_SEA_MONTH = 1

# Local objective-analysis settings retained from the corrected Fig. 5 code.
OHC_SOURCE_LAT_STEP_DEGREES = 2.0
OHC_SOURCE_LON_STEP_DEGREES = 4.0
OHC_OI_LAT_STEP_DEGREES = 1.0
OHC_OI_LON_STEP_DEGREES = 2.0
OHC_OI_LENGTH_SCALE_KM = 500.0
OHC_OI_SEARCH_RADIUS_KM = 1200.0
OHC_OI_NOISE_TO_SIGNAL_VARIANCE = 0.15
OHC_OI_MIN_OBSERVATIONS = 3
OHC_OI_MAX_NEIGHBORS = 16
OHC_OI_MAX_NORMALIZED_ERROR = 0.70
OHC_CLIMATOLOGY_MIN_YEARS = 3
OHC_OI_NUMERICAL_NUGGET = 1.0e-8
EARTH_RADIUS_KM = 6371.0

SEA_INFO = [
    ("WED", "Weddell Sea", -60.0, -20.0),
    ("KHV", "King Haakon VII Sea", -20.0, 0.0),
    ("RLS", "Riiser-Larsen Sea", 0.0, 10.0),
    ("LAZ", "Lazarev Sea", 10.0, 30.0),
    ("COS", "Cosmonauts Sea", 30.0, 50.0),
    ("COO", "Cooperation Sea", 50.0, 70.0),
    ("DAV", "Davis Sea", 70.0, 90.0),
    ("MAW", "Mawson Sea", 90.0, 130.0),
    ("DUR", "D'Urville Sea", 130.0, 150.0),
    ("SOM", "Somov Sea", 150.0, 170.0),
    ("ROS", "Ross Sea", 170.0, -130.0),  # crosses the date line
    ("AMU", "Amundsen Sea", -130.0, -100.0),
    ("BEL", "Bellingshausen Sea", -100.0, -60.0),
]
SEA_CODES = [item[0] for item in SEA_INFO]
SEA_NAMES = {item[0]: item[1] for item in SEA_INFO}


def ensure_output_directories():
    for directory in [DATA_DIR, OHC_DIR, OI_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def require_file(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def normalize_longitude(longitude):
    """Normalize scalar, NumPy or xarray longitudes to [-180, 180)."""
    return ((longitude + 180.0) % 360.0) - 180.0


def longitude_in_sector(longitude, west, east):
    """Return a half-open longitude-sector mask, including date-line sectors."""
    if west < east:
        return (longitude >= west) & (longitude < east)
    return (longitude >= west) | (longitude < east)


def sea_code_from_longitude(longitude):
    longitude = float(normalize_longitude(longitude))
    for code, _, west, east in SEA_INFO:
        if west < east:
            if west <= longitude < east:
                return code
        elif longitude >= west or longitude < east:
            return code
    return None


def filled_float_array(values):
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)


def choose_xarray_variable(dataset, candidates, description):
    lower_lookup = {
        str(variable).lower(): variable for variable in dataset.data_vars
    }
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return dataset[lower_lookup[candidate.lower()]]
    raise ValueError(
        f"Could not identify {description}. Tried {candidates}. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def standardize_monthly_grid(data_array):
    """Standardize a regular time/latitude/longitude xarray field."""
    aliases = {
        "valid_time": "time",
        "time_counter": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    available = set(data_array.coords) | set(data_array.dims)
    rename = {
        old: new
        for old, new in aliases.items()
        if old in available and new not in available
    }
    if rename:
        data_array = data_array.rename(rename)

    missing = {"time", "lat", "lon"}.difference(data_array.coords)
    if missing:
        raise ValueError(
            f"Missing grid coordinates {sorted(missing)} for {data_array.name}."
        )
    if data_array["lat"].ndim != 1 or data_array["lon"].ndim != 1:
        raise ValueError("The SIC source must use one-dimensional lat/lon grids.")

    for dimension in list(data_array.dims):
        if str(dimension).lower() in {"number", "expver", "member"}:
            data_array = data_array.mean(dimension, skipna=True)

    extra = set(data_array.dims).difference({"time", "lat", "lon"})
    if extra:
        raise ValueError(
            f"Unexpected dimensions in {data_array.name}: {sorted(extra)}"
        )

    data_array = data_array.assign_coords(
        lon=normalize_longitude(data_array["lon"].astype(float))
    )
    data_array = data_array.sortby("lon").sortby("lat").sortby("time")

    _, unique_lon_indices = np.unique(
        np.asarray(data_array["lon"].values), return_index=True
    )
    data_array = data_array.isel(lon=np.sort(unique_lon_indices))

    month_time = (
        pd.DatetimeIndex(pd.to_datetime(data_array["time"].values))
        .to_period("M")
        .to_timestamp()
    )
    data_array = data_array.assign_coords(time=month_time)
    if month_time.duplicated().any():
        data_array = data_array.groupby("time").mean("time", skipna=True)

    data_array = data_array.sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )
    data_array = data_array.where(data_array["lat"] <= -60.0, drop=True)
    return data_array.transpose("time", "lat", "lon").astype("float32")


def decode_argo_juld(juld_variable):
    """Decode Argo JULD using its stored units/calendar, with a safe fallback."""
    numeric = filled_float_array(juld_variable[:]).reshape(-1)
    output = np.full(numeric.size, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = np.isfinite(numeric)
    if not valid.any():
        return pd.DatetimeIndex(output)

    units = getattr(juld_variable, "units", "days since 1950-01-01 00:00:00")
    calendar = getattr(juld_variable, "calendar", "standard")
    try:
        decoded = num2date(
            numeric[valid],
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        decoded = pd.to_datetime(np.asarray(decoded), errors="coerce")
    except Exception:
        decoded = pd.to_datetime(
            numeric[valid],
            unit="D",
            origin="1950-01-01",
            errors="coerce",
        )
    output[valid] = decoded.to_numpy(dtype="datetime64[ns]")
    return pd.DatetimeIndex(output)


def profile_vector(variable, profile_index, number_of_profiles):
    """Read one vertical profile whether the profile dimension is first/last."""
    if variable.ndim == 1:
        return filled_float_array(variable[:]).reshape(-1)
    if variable.ndim != 2:
        raise ValueError(
            f"Variable {variable.name} must have one or two dimensions; "
            f"found shape {variable.shape}."
        )
    if variable.shape[0] == number_of_profiles:
        return filled_float_array(variable[profile_index, :]).reshape(-1)
    if variable.shape[1] == number_of_profiles:
        return filled_float_array(variable[:, profile_index]).reshape(-1)
    raise ValueError(
        f"Cannot identify profile dimension for {variable.name}: {variable.shape}"
    )


def calculate_profile_ohc(p_raw, t_raw, sp_raw, latitude, longitude):
    """Calculate 0–100-dbar OHC above local seawater freezing, in GJ m-2."""
    pressure = np.asarray(p_raw, dtype=float)
    temperature = np.asarray(t_raw, dtype=float)
    practical_salinity = np.asarray(sp_raw, dtype=float)

    common_size = min(pressure.size, temperature.size, practical_salinity.size)
    pressure = pressure[:common_size]
    temperature = temperature[:common_size]
    practical_salinity = practical_salinity[:common_size]

    valid = (
        np.isfinite(pressure)
        & np.isfinite(temperature)
        & np.isfinite(practical_salinity)
        & (pressure >= 0.0)
        & (pressure <= 130.0)
        & (temperature > -3.0)
        & (temperature < 20.0)
        & (practical_salinity > 20.0)
        & (practical_salinity < 42.0)
    )
    if int(valid.sum()) < MIN_VALID_LEVELS_OHC:
        return np.nan

    pressure = pressure[valid]
    temperature = temperature[valid]
    practical_salinity = practical_salinity[valid]
    order = np.argsort(pressure)
    pressure = pressure[order]
    temperature = temperature[order]
    practical_salinity = practical_salinity[order]

    pressure, unique_indices = np.unique(pressure, return_index=True)
    temperature = temperature[unique_indices]
    practical_salinity = practical_salinity[unique_indices]

    if pressure.size < MIN_VALID_LEVELS_OHC:
        return np.nan
    if float(np.nanmin(pressure)) > MAX_SHALLOWEST_VALID_DBAR:
        return np.nan
    if float(np.nanmax(pressure)) < MIN_DEEPEST_VALID_DBAR:
        return np.nan

    temperature_i = np.interp(OHC_PRESSURE_GRID_DBAR, pressure, temperature)
    salinity_i = np.interp(
        OHC_PRESSURE_GRID_DBAR, pressure, practical_salinity
    )

    try:
        absolute_salinity = gsw.SA_from_SP(
            salinity_i,
            OHC_PRESSURE_GRID_DBAR,
            longitude,
            latitude,
        )
        conservative_temperature = gsw.CT_from_t(
            absolute_salinity,
            temperature_i,
            OHC_PRESSURE_GRID_DBAR,
        )
        freezing_temperature = gsw.CT_freezing(
            absolute_salinity,
            OHC_PRESSURE_GRID_DBAR,
            0.0,
        )
        density = gsw.rho(
            absolute_salinity,
            conservative_temperature,
            OHC_PRESSURE_GRID_DBAR,
        )
        depth_m = -gsw.z_from_p(OHC_PRESSURE_GRID_DBAR, latitude)
    except Exception:
        return np.nan

    temperature_excess = np.maximum(
        conservative_temperature - freezing_temperature, 0.0
    )
    volumetric_heat = density * CP0 * temperature_excess
    if hasattr(np, "trapezoid"):
        ohc_j_m2 = np.trapezoid(volumetric_heat, depth_m)
    else:
        ohc_j_m2 = np.trapz(volumetric_heat, depth_m)

    if not np.isfinite(ohc_j_m2) or ohc_j_m2 < 0.0:
        return np.nan
    return float(ohc_j_m2 / 1.0e9)



# 1. Argo_profile_OHC_0_100m.csv

def build_argo_profile_ohc_csv():
    """Calculate and save accepted profile-level upper-ocean heat content."""
    require_file(ARGO_SOURCE_NC, "cleaned gridded Argo profile NetCDF")
    print("\n" + "=" * 72)
    print("1. Building Argo_profile_OHC_0_100m.csv")
    print("=" * 72)

    records = []
    with Dataset(ARGO_SOURCE_NC, mode="r") as dataset:
        required = ["PRES_GRID", "LATITUDE", "LONGITUDE", "JULD", "TEMP", "PSAL"]
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ValueError(
                f"Argo source is missing variables {missing}. "
                f"Available: {list(dataset.variables)}"
            )

        latitude = filled_float_array(dataset.variables["LATITUDE"][:]).reshape(-1)
        longitude = normalize_longitude(
            filled_float_array(dataset.variables["LONGITUDE"][:]).reshape(-1)
        )
        time = decode_argo_juld(dataset.variables["JULD"])
        number_of_profiles = latitude.size

        if longitude.size != number_of_profiles or time.size != number_of_profiles:
            raise ValueError("Argo LATITUDE, LONGITUDE and JULD sizes do not match.")

        pressure_variable = dataset.variables["PRES_GRID"]
        temperature_variable = dataset.variables["TEMP"]
        salinity_variable = dataset.variables["PSAL"]

        valid_metadata = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & (latitude >= -90.0)
            & (latitude < -60.0)
            & (~pd.isna(time))
            & (time >= START_TIME)
            & (time <= END_TIME)
        )
        candidate_indices = np.flatnonzero(valid_metadata)
        print(f"Candidate profiles: {candidate_indices.size:,}")

        shared_pressure = None
        if pressure_variable.ndim == 1:
            shared_pressure = filled_float_array(pressure_variable[:]).reshape(-1)

        for processed, profile_index in enumerate(candidate_indices, start=1):
            pressure = (
                shared_pressure
                if shared_pressure is not None
                else profile_vector(
                    pressure_variable, profile_index, number_of_profiles
                )
            )
            temperature = profile_vector(
                temperature_variable, profile_index, number_of_profiles
            )
            salinity = profile_vector(
                salinity_variable, profile_index, number_of_profiles
            )

            ohc = calculate_profile_ohc(
                pressure,
                temperature,
                salinity,
                float(latitude[profile_index]),
                float(longitude[profile_index]),
            )
            if np.isfinite(ohc):
                timestamp = pd.Timestamp(time[profile_index])
                sea = sea_code_from_longitude(longitude[profile_index])
                if sea is not None:
                    records.append(
                        {
                            "time": timestamp,
                            "month": timestamp.to_period("M").to_timestamp(),
                            "sea": sea,
                            "lat": float(latitude[profile_index]),
                            "lon": float(longitude[profile_index]),
                            "OHC_0_100m_GJ_m2": ohc,
                        }
                    )

            if processed % 5000 == 0:
                print(
                    f"  processed {processed:,}/{candidate_indices.size:,}; "
                    f"accepted OHC profiles={len(records):,}"
                )

    if not records:
        raise RuntimeError("No valid profile-level OHC records were produced.")

    profiles = pd.DataFrame(records).sort_values(["time", "sea", "lat", "lon"])
    profiles = profiles[
        ["time", "month", "sea", "lat", "lon", "OHC_0_100m_GJ_m2"]
    ].reset_index(drop=True)
    profiles.to_csv(
        ARGO_PROFILE_OHC_CSV,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.8g",
    )

    print(f"Accepted profiles: {len(profiles):,}")
    print("Saved:", ARGO_PROFILE_OHC_CSV)
    return profiles



# 2. OHC_0_100m_monthly_anomalies_local_objective_analysis_2008_2025.nc

def spherical_xyz(longitude, latitude):
    longitude_radians = np.deg2rad(np.asarray(longitude, dtype=float))
    latitude_radians = np.deg2rad(np.asarray(latitude, dtype=float))
    return np.column_stack(
        [
            np.cos(latitude_radians) * np.cos(longitude_radians),
            np.cos(latitude_radians) * np.sin(longitude_radians),
            np.sin(latitude_radians),
        ]
    )


def great_circle_distance_matrix_km(first_xyz, second_xyz):
    cosine_angle = np.clip(
        np.asarray(first_xyz) @ np.asarray(second_xyz).T, -1.0, 1.0
    )
    return EARTH_RADIUS_KM * np.arccos(cosine_angle)


def read_profile_csv_for_mapping():
    require_file(ARGO_PROFILE_OHC_CSV, "profile-level OHC CSV")
    profiles = pd.read_csv(ARGO_PROFILE_OHC_CSV)
    required = {"lat", "lon", "OHC_0_100m_GJ_m2"}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Profile OHC CSV is missing columns: {sorted(missing)}")

    time_column = "month" if "month" in profiles.columns else "time"
    profiles["month"] = (
        pd.to_datetime(profiles[time_column], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    for column in ["lat", "lon", "OHC_0_100m_GJ_m2"]:
        profiles[column] = pd.to_numeric(profiles[column], errors="coerce")
    profiles = profiles.dropna(
        subset=["month", "lat", "lon", "OHC_0_100m_GJ_m2"]
    ).copy()
    profiles["lon"] = normalize_longitude(profiles["lon"].to_numpy())
    profiles = profiles.loc[
        (profiles["month"] >= START_TIME)
        & (profiles["month"] <= END_TIME)
        & (profiles["lat"] >= -90.0)
        & (profiles["lat"] < -60.0)
    ].copy()
    if profiles.empty:
        raise ValueError("No usable profile OHC records remain for 2008–2025.")
    return profiles


def build_source_bin_monthly_anomalies():
    """Average profiles into source bins and remove calendar-month means."""
    profiles = read_profile_csv_for_mapping()
    profiles["lat_bin"] = (
        -90.0
        + (
            np.floor(
                (profiles["lat"].to_numpy() + 90.0)
                / OHC_SOURCE_LAT_STEP_DEGREES
            )
            + 0.5
        )
        * OHC_SOURCE_LAT_STEP_DEGREES
    )
    profiles["lon_bin"] = (
        -180.0
        + (
            np.floor(
                (profiles["lon"].to_numpy() + 180.0)
                / OHC_SOURCE_LON_STEP_DEGREES
            )
            + 0.5
        )
        * OHC_SOURCE_LON_STEP_DEGREES
    )

    grouped = (
        profiles.groupby(["month", "lat_bin", "lon_bin"], as_index=False)
        .agg(
            OHC_0_100m_GJ_m2=("OHC_0_100m_GJ_m2", "mean"),
            profile_count=("OHC_0_100m_GJ_m2", "size"),
        )
    )
    source_latitude = np.arange(
        -90.0 + OHC_SOURCE_LAT_STEP_DEGREES / 2.0,
        -60.0,
        OHC_SOURCE_LAT_STEP_DEGREES,
    )
    source_longitude = np.arange(
        -180.0 + OHC_SOURCE_LON_STEP_DEGREES / 2.0,
        180.0,
        OHC_SOURCE_LON_STEP_DEGREES,
    )
    source = (
        grouped.set_index(["month", "lat_bin", "lon_bin"])[
            "OHC_0_100m_GJ_m2"
        ]
        .to_xarray()
        .rename({"month": "time", "lat_bin": "lat", "lon_bin": "lon"})
        .reindex(time=FULL_MONTH_INDEX, lat=source_latitude, lon=source_longitude)
        .transpose("time", "lat", "lon")
        .astype("float32")
    )

    climatology = source.groupby("time.month").mean("time", skipna=True)
    climatology_count = source.groupby("time.month").count("time")
    pieces = []
    for timestamp in source["time"].values:
        month_number = pd.Timestamp(timestamp).month
        anomaly = source.sel(time=timestamp) - climatology.sel(month=month_number)
        anomaly = anomaly.where(
            climatology_count.sel(month=month_number)
            >= OHC_CLIMATOLOGY_MIN_YEARS
        )
        pieces.append(anomaly.expand_dims(time=[timestamp]))

    source_anomaly = xr.concat(pieces, dim="time").transpose("time", "lat", "lon")
    source_anomaly.name = "source_OHC_anomaly"
    source_anomaly.attrs.update(
        {
            "long_name": "Source-bin monthly Argo OHC anomaly, 0-100 m",
            "units": "GJ m-2",
            "climatology_period": f"{START_YEAR}-{END_YEAR}",
            "minimum_climatology_years": OHC_CLIMATOLOGY_MIN_YEARS,
        }
    )
    print("Source anomaly grid:", dict(source_anomaly.sizes))
    return source_anomaly.astype("float32")


def local_objective_analysis_one_month(
    source_field,
    target_longitude_2d,
    target_latitude_2d,
    target_xyz,
):
    """Map one source-bin anomaly field by local optimal interpolation."""
    source_longitude_2d, source_latitude_2d = np.meshgrid(
        source_field["lon"].values, source_field["lat"].values
    )
    source_values_2d = np.asarray(source_field.values, dtype=float)
    valid = np.isfinite(source_values_2d)

    target_shape = target_longitude_2d.shape
    mapped = np.full(target_shape, np.nan, dtype=np.float32)
    mapping_error = np.full(target_shape, np.nan, dtype=np.float32)
    local_count = np.zeros(target_shape, dtype=np.int16)
    if int(valid.sum()) < OHC_OI_MIN_OBSERVATIONS:
        return mapped, mapping_error, local_count

    observation_values = source_values_2d[valid]
    observation_xyz = spherical_xyz(
        source_longitude_2d[valid], source_latitude_2d[valid]
    )
    observation_tree = cKDTree(observation_xyz)
    chord_radius = 2.0 * np.sin(
        OHC_OI_SEARCH_RADIUS_KM / (2.0 * EARTH_RADIUS_KM)
    )
    number_of_neighbors = min(
        OHC_OI_MAX_NEIGHBORS, observation_values.size
    )
    try:
        chord_distance, neighbor_index = observation_tree.query(
            target_xyz,
            k=number_of_neighbors,
            distance_upper_bound=chord_radius,
            workers=-1,
        )
    except TypeError:
        chord_distance, neighbor_index = observation_tree.query(
            target_xyz,
            k=number_of_neighbors,
            distance_upper_bound=chord_radius,
        )
    if number_of_neighbors == 1:
        chord_distance = chord_distance[:, None]
        neighbor_index = neighbor_index[:, None]

    mapped_flat = mapped.ravel()
    error_flat = mapping_error.ravel()
    count_flat = local_count.ravel()
    inverse_cache = {}

    for target_index in range(target_xyz.shape[0]):
        indices = neighbor_index[target_index]
        usable = (
            np.isfinite(chord_distance[target_index])
            & (indices < observation_values.size)
        )
        indices = np.sort(indices[usable])
        if indices.size < OHC_OI_MIN_OBSERVATIONS:
            continue

        neighborhood_key = tuple(int(index) for index in indices)
        cached = inverse_cache.get(neighborhood_key)
        if cached is None:
            local_xyz = observation_xyz[indices]
            local_values = observation_values[indices]
            observation_distance = great_circle_distance_matrix_km(
                local_xyz, local_xyz
            )
            observation_covariance = np.exp(
                -(observation_distance / OHC_OI_LENGTH_SCALE_KM) ** 2
            )
            system = observation_covariance.copy()
            system.flat[:: system.shape[0] + 1] += (
                OHC_OI_NOISE_TO_SIGNAL_VARIANCE + OHC_OI_NUMERICAL_NUGGET
            )
            try:
                system_inverse = np.linalg.inv(system)
            except np.linalg.LinAlgError:
                system_inverse = np.linalg.pinv(system, hermitian=True)
            inverse_cache[neighborhood_key] = (
                local_xyz, local_values, system_inverse
            )
        else:
            local_xyz, local_values, system_inverse = cached

        grid_distance = great_circle_distance_matrix_km(
            target_xyz[target_index : target_index + 1], local_xyz
        ).ravel()
        covariance_grid_observation = np.exp(
            -(grid_distance / OHC_OI_LENGTH_SCALE_KM) ** 2
        )
        weights = system_inverse @ covariance_grid_observation
        normalized_error = float(
            1.0 - np.dot(covariance_grid_observation, weights)
        )
        normalized_error = float(np.clip(normalized_error, 0.0, 1.0))
        if normalized_error > OHC_OI_MAX_NORMALIZED_ERROR:
            continue

        mapped_flat[target_index] = float(np.dot(weights, local_values))
        error_flat[target_index] = normalized_error
        count_flat[target_index] = int(indices.size)

    return mapped, mapping_error, local_count


def remove_residual_calendar_month_mean(data_array):
    support = np.isfinite(data_array)
    climatology = data_array.groupby("time.month").mean("time", skipna=True)
    return (data_array.groupby("time.month") - climatology).where(support)


def build_ohc_local_oi_netcdf():
    """Create the 216-month Local-OI anomaly cube and diagnostics."""
    print("\n" + "=" * 72)
    print("2. Building Local-OI monthly OHC anomaly NetCDF")
    print("=" * 72)
    source_anomaly = build_source_bin_monthly_anomalies()

    target_latitude = np.arange(
        -90.0 + OHC_OI_LAT_STEP_DEGREES / 2.0,
        -60.0,
        OHC_OI_LAT_STEP_DEGREES,
    )
    target_longitude = np.arange(
        -180.0 + OHC_OI_LON_STEP_DEGREES / 2.0,
        180.0,
        OHC_OI_LON_STEP_DEGREES,
    )
    target_longitude_2d, target_latitude_2d = np.meshgrid(
        target_longitude, target_latitude
    )
    target_xyz = spherical_xyz(
        target_longitude_2d.ravel(), target_latitude_2d.ravel()
    )

    output_shape = (
        source_anomaly.sizes["time"],
        target_latitude.size,
        target_longitude.size,
    )
    mapped_values = np.full(output_shape, np.nan, dtype=np.float32)
    mapping_errors = np.full(output_shape, np.nan, dtype=np.float32)
    local_counts = np.zeros(output_shape, dtype=np.int16)

    for time_index, timestamp in enumerate(source_anomaly["time"].values):
        if time_index % 12 == 0:
            print(
                f"  Local OI {time_index + 1}/{output_shape[0]} "
                f"({pd.Timestamp(timestamp):%Y-%m})"
            )
        mapped, error, count = local_objective_analysis_one_month(
            source_anomaly.isel(time=time_index),
            target_longitude_2d,
            target_latitude_2d,
            target_xyz,
        )
        mapped_values[time_index] = mapped
        mapping_errors[time_index] = error
        local_counts[time_index] = count

    coordinates = {
        "time": source_anomaly["time"].values,
        "lat": target_latitude,
        "lon": target_longitude,
    }
    mapped_anomaly = xr.DataArray(
        mapped_values,
        dims=("time", "lat", "lon"),
        coords=coordinates,
        name="OHC_anomaly_OI",
        attrs={
            "long_name": "Local-OI monthly Argo OHC anomaly, 0-100 m",
            "units": "GJ m-2",
        },
    )
    mapped_anomaly = remove_residual_calendar_month_mean(mapped_anomaly)
    mapped_anomaly.name = "OHC_anomaly_OI"
    mapped_anomaly.attrs.update(
        {
            "long_name": "Local-OI monthly Argo OHC anomaly, 0-100 m",
            "units": "GJ m-2",
        }
    )
    mapping_error = xr.DataArray(
        mapping_errors,
        dims=("time", "lat", "lon"),
        coords=coordinates,
        name="normalized_mapping_error",
        attrs={
            "long_name": "Normalized local objective-analysis mapping error",
            "units": "1",
        },
    )
    local_observation_count = xr.DataArray(
        local_counts,
        dims=("time", "lat", "lon"),
        coords=coordinates,
        name="local_observation_count",
        attrs={
            "long_name": "Number of local source-bin observations used",
            "units": "1",
        },
    )

    output_dataset = xr.Dataset(
        {
            "OHC_anomaly_OI": mapped_anomaly.astype("float32"),
            "normalized_mapping_error": mapping_error.astype("float32"),
            "local_observation_count": local_observation_count.astype("int16"),
        }
    )
    output_dataset.attrs.update(
        {
            "title": "Monthly 0-100 m Argo OHC anomalies mapped by Local OI",
            "method": "Local objective analysis / optimal interpolation",
            "covariance_model": "C(d) = exp[-(d/L)^2]",
            "length_scale_km": OHC_OI_LENGTH_SCALE_KM,
            "search_radius_km": OHC_OI_SEARCH_RADIUS_KM,
            "noise_to_signal_variance": OHC_OI_NOISE_TO_SIGNAL_VARIANCE,
            "minimum_local_observations": OHC_OI_MIN_OBSERVATIONS,
            "maximum_local_neighbors": OHC_OI_MAX_NEIGHBORS,
            "maximum_normalized_mapping_error": OHC_OI_MAX_NORMALIZED_ERROR,
            "minimum_climatology_years": OHC_CLIMATOLOGY_MIN_YEARS,
            "source_latitude_bin_degrees": OHC_SOURCE_LAT_STEP_DEGREES,
            "source_longitude_bin_degrees": OHC_SOURCE_LON_STEP_DEGREES,
            "OI_target_latitude_step_degrees": OHC_OI_LAT_STEP_DEGREES,
            "OI_target_longitude_step_degrees": OHC_OI_LON_STEP_DEGREES,
            "analysis_period": f"{START_YEAR}-{END_YEAR}",
            "temporal_interpolation": "none",
            "source_profile_csv": ARGO_PROFILE_OHC_CSV.name,
        }
    )
    encoding = {
        "OHC_anomaly_OI": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, target_latitude.size, target_longitude.size),
        },
        "normalized_mapping_error": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (1, target_latitude.size, target_longitude.size),
        },
        "local_observation_count": {
            "zlib": True,
            "complevel": 4,
            "dtype": "int16",
            "_FillValue": np.int16(-32767),
            "chunksizes": (1, target_latitude.size, target_longitude.size),
        },
    }
    output_dataset.to_netcdf(
        OHC_OI_MONTHLY_NC,
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )

    valid_fraction = float(np.isfinite(mapped_anomaly.values).mean() * 100.0)
    print(f"Valid Local-OI space-time coverage: {valid_fraction:.2f}%")
    print("Saved:", OHC_OI_MONTHLY_NC)
    return output_dataset



# 3. SIC_seawise_monthly_2008_2025_LOW_RAM.csv

def open_sic_low_ram():
    require_file(SIC_SOURCE_NC, "monthly OSTIA sea-ice-fraction NetCDF")
    try:
        dataset = xr.open_dataset(
            SIC_SOURCE_NC,
            engine="netcdf4",
            decode_times=True,
            decode_timedelta=False,
            chunks={"time": 1, "lat": 150, "lon": 720},
        )
    except ValueError as error:
        if "chunk" not in str(error).lower() and "dask" not in str(error).lower():
            raise
        dataset = xr.open_dataset(
            SIC_SOURCE_NC,
            engine="netcdf4",
            decode_times=True,
            decode_timedelta=False,
            chunks=None,
        )
    sea_ice_fraction = choose_xarray_variable(
        dataset,
        ["SIF", "sif", "sea_ice_fraction", "sic", "SIC"],
        "sea-ice fraction/concentration variable",
    )
    return dataset, standardize_monthly_grid(sea_ice_fraction)


def determine_sic_scale(data_array):
    sample = data_array.isel(
        time=slice(0, min(12, data_array.sizes["time"])),
        lat=slice(None, None, 20),
        lon=slice(None, None, 40),
    )
    maximum = sample.max(dim=list(sample.dims), skipna=True)
    sample_maximum = float(maximum.compute().item())
    return 100.0 if sample_maximum <= 1.5 else 1.0


def build_sic_seawise_csv():
    """Create the 216-row cosine-weighted monthly SIC sector table."""
    print("\n" + "=" * 72)
    print("3. Building SIC_seawise_monthly_2008_2025_LOW_RAM.csv")
    print("=" * 72)
    dataset, sea_ice_fraction = open_sic_low_ram()
    scale = determine_sic_scale(sea_ice_fraction)
    if scale == 100.0:
        print("SIF detected as fraction; converting to percent.")
    else:
        print("SIF detected as percent.")
    sea_ice_percent = (sea_ice_fraction * scale).clip(min=0.0, max=100.0)

    series = {}
    for code, name, west, east in SEA_INFO:
        print(f"  Processing {code}: {name}")
        sector = sea_ice_percent.where(
            longitude_in_sector(sea_ice_percent["lon"], west, east),
            drop=True,
        )
        if sector.sizes.get("lon", 0) == 0:
            raise ValueError(f"No SIC longitude cells found for {code}.")
        latitude_weights = np.cos(np.deg2rad(sector["lat"]))
        monthly_mean = sector.weighted(latitude_weights).mean(
            dim=("lat", "lon"), skipna=True
        )
        loaded = monthly_mean.compute()
        time_index = (
            pd.DatetimeIndex(pd.to_datetime(loaded["time"].values))
            .to_period("M")
            .to_timestamp()
        )
        values = pd.Series(
            np.asarray(loaded.values, dtype=float), index=time_index, name=code
        )
        series[code] = values.groupby(level=0).mean().sort_index()
        del sector, monthly_mean, loaded
        gc.collect()

    output = pd.DataFrame(series).reindex(FULL_MONTH_INDEX)
    output = output.reindex(columns=SEA_CODES)
    output.index.name = "time"

    if output.shape != (216, 13):
        raise ValueError(f"Unexpected SIC output shape: {output.shape}")
    if output.index.duplicated().any():
        raise ValueError("Duplicate SIC monthly timestamps were produced.")
    finite = output.to_numpy(dtype=float)
    if np.nanmin(finite) < 0.0 or np.nanmax(finite) > 100.0:
        raise ValueError("SIC values outside 0–100% were produced.")
    if output.isna().any().any():
        missing = output.isna().sum()
        print("WARNING: missing SIC values by sector:\n", missing[missing > 0])

    output.to_csv(
        SIC_SEAWISE_CSV,
        index=True,
        date_format="%Y-%m-%d",
        float_format="%.8g",
    )
    dataset.close()
    print("Shape:", output.shape)
    print("Saved:", SIC_SEAWISE_CSV)
    return output



# 4. Argo_monthly_OHC_0_100m_GJ_m2.csv

def read_valid_profile_ohc_csv():
    require_file(ARGO_PROFILE_OHC_CSV, "profile-level OHC CSV")
    profiles = pd.read_csv(ARGO_PROFILE_OHC_CSV)
    required = {"sea", "OHC_0_100m_GJ_m2"}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Profile OHC CSV is missing columns: {sorted(missing)}")
    time_column = "month" if "month" in profiles.columns else "time"
    if time_column not in profiles.columns:
        raise ValueError("Profile OHC CSV requires a 'month' or 'time' column.")
    profiles["month"] = (
        pd.to_datetime(profiles[time_column], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    profiles["OHC_0_100m_GJ_m2"] = pd.to_numeric(
        profiles["OHC_0_100m_GJ_m2"], errors="coerce"
    )
    profiles = profiles.dropna(
        subset=["month", "sea", "OHC_0_100m_GJ_m2"]
    ).copy()
    profiles = profiles.loc[
        (profiles["month"] >= START_TIME)
        & (profiles["month"] <= END_TIME)
        & (profiles["sea"].isin(SEA_CODES))
    ].copy()
    return profiles


def monthly_ohc_and_count_tables():
    profiles = read_valid_profile_ohc_csv()
    grouped = (
        profiles.groupby(["month", "sea"])["OHC_0_100m_GJ_m2"]
        .agg(["mean", "size"])
        .reset_index()
    )
    ohc = grouped.pivot(index="month", columns="sea", values="mean")
    counts = grouped.pivot(index="month", columns="sea", values="size")
    ohc = ohc.reindex(index=FULL_MONTH_INDEX, columns=SEA_CODES)
    counts = (
        counts.reindex(index=FULL_MONTH_INDEX, columns=SEA_CODES)
        .fillna(0)
        .astype("int32")
    )
    ohc = ohc.where(counts >= MIN_ARGO_PROFILES_PER_SEA_MONTH)
    ohc.index.name = "time"
    counts.index.name = "time"
    return ohc, counts


def build_argo_monthly_ohc_csv():
    """Average accepted profile OHC values within each sea and month."""
    print("\n" + "=" * 72)
    print("4. Building Argo_monthly_OHC_0_100m_GJ_m2.csv")
    print("=" * 72)
    ohc, _ = monthly_ohc_and_count_tables()
    ohc.to_csv(
        ARGO_MONTHLY_OHC_CSV,
        index=True,
        date_format="%Y-%m-%d",
        float_format="%.8g",
    )
    print("Shape:", ohc.shape)
    for sea in SEA_CODES:
        print(f"  {sea}: {int(ohc[sea].notna().sum())}/216 valid months")
    print("Saved:", ARGO_MONTHLY_OHC_CSV)
    return ohc



# 5. Argo_monthly_OHC_profile_counts.csv

def build_argo_monthly_profile_counts_csv():
    """Count accepted Argo OHC profiles within every sea-month."""
    print("\n" + "=" * 72)
    print("5. Building Argo_monthly_OHC_profile_counts.csv")
    print("=" * 72)
    _, counts = monthly_ohc_and_count_tables()
    counts.to_csv(
        ARGO_MONTHLY_COUNTS_CSV,
        index=True,
        date_format="%Y-%m-%d",
    )
    print("Shape:", counts.shape)
    print("Total accepted profiles by sea:")
    print(counts.sum(axis=0).to_string())
    print("Saved:", ARGO_MONTHLY_COUNTS_CSV)
    return counts



# RUN ALL REQUESTED BUILDERS

def main():
    ensure_output_directories()

    # Build the common profile-level source first.
    if BUILD_PROFILE_OHC_CSV:
        build_argo_profile_ohc_csv()
    else:
        require_file(ARGO_PROFILE_OHC_CSV, "existing profile-level OHC CSV")

    # Execute in the same numbered order as the five documented sections.
    if BUILD_OHC_OI_NETCDF:
        build_ohc_local_oi_netcdf()
    if BUILD_SIC_SEAWISE_CSV:
        build_sic_seawise_csv()
    if BUILD_MONTHLY_OHC_CSV:
        build_argo_monthly_ohc_csv()
    if BUILD_MONTHLY_PROFILE_COUNTS_CSV:
        build_argo_monthly_profile_counts_csv()

    print("\n" + "=" * 72)
    print("DONE — requested derived input files")
    print("=" * 72)
    requested = [
        ARGO_PROFILE_OHC_CSV,
        OHC_OI_MONTHLY_NC,
        SIC_SEAWISE_CSV,
        ARGO_MONTHLY_OHC_CSV,
        ARGO_MONTHLY_COUNTS_CSV,
    ]
    for path in requested:
        print(f"{'FOUND' if path.exists() else 'MISSING'}: {path}")


if __name__ == "__main__":
    main()
