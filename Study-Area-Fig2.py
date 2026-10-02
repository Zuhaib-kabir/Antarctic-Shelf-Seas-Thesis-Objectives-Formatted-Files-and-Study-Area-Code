# INTEGRATED THESIS STUDY-AREA MAP
# Antarctic marginal seas + GEBCO bathymetry + SIC contours
# + Argo hydrography points + GBIF occurrence points
# + integrated thesis concept annotations
#
# INPUT PATHS: restored from your previous working codes
# OUTPUT PATH: /content/drive/MyDrive/Southern_Ocean_Paper
#
# OUTPUT:
#   1) 1080-dpi PNG
#   2) Vector PDF

#Google Colab installation (run once if needed):
!apt-get -qq install -y libproj-dev proj-data proj-bin libgeos-dev
!pip -q install cartopy xarray netCDF4 h5netcdf matplotlib numpy pandas dask

# mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

import os
import gc
import warnings
import textwrap

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath

from matplotlib.patches import FancyBboxPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=UserWarning)


# 1. GLOBAL GRAPHIC SETTINGS
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# 2. INPUT FILE PATHS

bathy_file = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "GEBCO_2024_CF.nc"
)

sic_file = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"
)

argo_file = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "argo_SO_profiles_2001_2025_cleaned_gridded.nc"
)

occurrence_file = (
    "/content/drive/MyDrive/SAM_Thesis/Data/Biodiversity_Indices/Processed/"
    "GBIF_clean_gridded_MONTHLY5000_2008_2025_1deg.csv"
)


# 3. OUTPUT PATHS
out_dir = "/content/drive/MyDrive/Southern_Ocean_Paper"
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(
    out_dir,
    "Integrated_Thesis_Study_Area_Map_1080dpi.png"
)

out_pdf = os.path.join(
    out_dir,
    "Integrated_Thesis_Study_Area_Map_Vector.pdf"
)

EXPORT_DPI = 1080

BATHY_SOURCE_LABEL = "GEBCO 2024"
OCCURRENCE_SOURCE_LABEL = "GBIF cleaned occurrence records"

MAX_ARGO_POINTS = 6500
ARGO_ROUNDING_DEG = 0.10

MAX_OCCURRENCE_POINTS = 950
RANDOM_SEED = 42


# 4. COLOURS
GREEN_CONCEPT  = "#187A20"
TEAL_CONCEPT   = "#007C7C"
ORANGE_CONCEPT = "#FF8500"
RED_CONCEPT    = "#F02020"
BLUE_CONCEPT   = "#1847F2"
GRAY_CONCEPT   = "#666666"

SIC_RED = "#D7191C"

# Requested dark-brown 2000 m isobath
SHELFBREAK_COLOR = "#5A3526"


# 5. EXACT 13 ANTARCTIC SHELF-SEA SECTORS

sea_info = [
    ("WED", "Weddell Sea",          "60°W–20°W",    -60,  -20),
    ("KHV", "King Haakon VII Sea",  "20°W–0°",      -20,    0),
    ("RLS", "Riiser-Larsen Sea",    "0°–10°E",        0,   10),
    ("LAZ", "Lazarev Sea",          "10°E–30°E",     10,   30),
    ("COS", "Cosmonauts Sea",       "30°E–50°E",     30,   50),
    ("COO", "Cooperation Sea",      "50°E–70°E",     50,   70),
    ("DAV", "Davis Sea",            "70°E–90°E",     70,   90),
    ("MAW", "Mawson Sea",           "90°E–130°E",    90,  130),
    ("DUR", "D'Urville Sea",        "130°E–150°E",  130,  150),
    ("SOM", "Somov Sea",            "150°E–170°E",  150,  170),
    ("ROS", "Ross Sea",             "170°E–130°W",  170, -130),
    ("AMU", "Amundsen Sea",         "130°W–100°W", -130, -100),
    ("BEL", "Bellingshausen Sea",   "100°W–60°W",  -100,  -60),
]

sector_colors = [
    "#E76F51",
    "#F4A261",
    "#E9D8A6",
    "#90BE6D",
    "#43AA8B",
    "#76C7C0",
    "#4EA8DE",
    "#7469C4",
    "#B23A88",
    "#D45AAB",
    "#9B6A36",
    "#D6A45F",
    "#63B1AD",
]


# 6. LABEL POSITIONS AND ARROW TARGETS

label_positions = {
    "WED": (-43,  -64.8),
    "KHV": (-11,  -61.7),
    "RLS": (6,    -61.5),
    "LAZ": (21,   -62.4),
    "COS": (42,   -63.0),
    "COO": (62,   -64.5),
    "DAV": (82,   -66.3),
    "MAW": (110,  -64.2),
    "DUR": (140,  -66.8),
    "SOM": (160,  -68.8),
    "ROS": (-155, -71.0),
    "AMU": (-118, -67.0),
    "BEL": (-85,  -66.0),
}

arrow_targets = {
    "WED": (-48,  -63.6),
    "KHV": (-14,  -60.9),
    "RLS": (5,    -60.75),

    "LAZ": (20,   -61.6),
    "COS": (42,   -62.0),
    "COO": (62,   -63.5),
    "DAV": (82,   -65.7),

    "BEL": (-87,  -65.5),

    "AMU": (-120, -66.2),
    "ROS": (-151, -70.3),

    "MAW": (111,  -63.8),
    "DUR": (141,  -66.0),
    "SOM": (160,  -68.2),
}

outside_lon_labels = {
    "0°":     (0.500,  1.012),
    "60°E":   (0.920,  0.815),
    "120°E":  (0.920,  0.175),
    "180°":   (0.500, -0.020),
    "120°W":  (0.080,  0.175),
    "60°W":   (0.080,  0.815),
}


# 7. PROFESSIONAL FIXED LAYOUT
MAP_RECT = [0.115, 0.145, 0.585, 0.785]
INSET_RECT = [0.010, 0.760, 0.125, 0.185]

GREEN_BOX_RECT  = [0.155, 0.900, 0.245, 0.090]
TEAL_BOX_RECT   = [0.535, 0.900, 0.235, 0.090]
ORANGE_BOX_RECT = [0.006, 0.455, 0.155, 0.145]
RED_BOX_RECT    = [0.025, 0.180, 0.220, 0.105]
BLUE_BOX_RECT   = [0.535, 0.145, 0.205, 0.105]

COMMON_BOX_RECT = [0.770, 0.875, 0.220, 0.075]

NOTE_BOX_RECT = [0.008, 0.028, 0.265, 0.105]

# Bathymetry scale centered
COLORBAR_RECT = [0.285, 0.050, 0.390, 0.030]

MAP_LEGEND_RECT     = [0.755, 0.530, 0.238, 0.255]
SECTOR_LEGEND_RECT  = [0.755, 0.250, 0.238, 0.255]
CONCEPT_LEGEND_RECT = [0.755, 0.055, 0.238, 0.165]


# 8. HELPER FUNCTIONS

def require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{description} was not found.\n"
            f"Expected path:\n{path}\n"
        )


def normalize_lon(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def get_coord_names(ds):

    all_names = (
        list(ds.coords)
        + list(ds.dims)
        + list(ds.data_vars)
    )

    lat_candidates = []
    lon_candidates = []

    for name in all_names:

        lname = name.lower()

        if lname in [
            "lat",
            "latitude",
            "nav_lat",
            "y"
        ]:
            lat_candidates.insert(
                0,
                name
            )

        elif "lat" in lname:
            lat_candidates.append(
                name
            )

        if lname in [
            "lon",
            "longitude",
            "nav_lon",
            "x"
        ]:
            lon_candidates.insert(
                0,
                name
            )

        elif "lon" in lname:
            lon_candidates.append(
                name
            )

    if (
        not lat_candidates
        or not lon_candidates
    ):
        raise ValueError(
            "Could not detect latitude/longitude names.\n"
            f"Available names: {all_names}"
        )

    return (
        lat_candidates[0],
        lon_candidates[0]
    )


def get_time_name(ds):

    for name in (
        list(ds.coords)
        + list(ds.dims)
        + list(ds.data_vars)
    ):

        lname = name.lower()

        if (
            "time" in lname
            or "juld" in lname
            or "date" in lname
        ):
            return name

    return None


def get_first_numeric_var(ds):

    for var_name in ds.data_vars:

        if np.issubdtype(
            ds[var_name].dtype,
            np.number
        ):
            return var_name

    raise ValueError(
        "No numeric data variable found."
    )


def subset_antarctic(
    ds,
    lat_name,
    south=-90,
    north=-58
):

    lat_values = np.asarray(
        ds[lat_name].values
    )

    if lat_values.ndim != 1:
        return ds

    if lat_values[0] < lat_values[-1]:

        return ds.sel({
            lat_name: slice(
                south,
                north
            )
        })

    return ds.sel({
        lat_name: slice(
            north,
            south
        )
    })


def maybe_sort_lon(
    data_array,
    lon_name
):

    lon_values = np.asarray(
        data_array[lon_name].values
    )

    if lon_values.ndim != 1:
        return data_array

    normalized = normalize_lon(
        lon_values
    )

    sort_index = np.argsort(
        normalized
    )

    data_array = data_array.assign_coords({
        lon_name: normalized
    })

    data_array = data_array.isel({
        lon_name: sort_index
    })

    return data_array


def get_sector_parts(
    lon1,
    lon2
):

    if lon1 > lon2:

        return [
            (lon1, 180),
            (-180, lon2)
        ]

    return [
        (lon1, lon2)
    ]


def make_sector_polygon(
    lon1,
    lon2,
    lat_inner=-82.0,
    lat_outer=-58.0,
    n=1500
):

    lons = np.linspace(
        lon1,
        lon2,
        n
    )

    outer_lats = np.full_like(
        lons,
        lat_outer,
        dtype=float
    )

    inner_lats = np.full_like(
        lons,
        lat_inner,
        dtype=float
    )

    polygon_lons = np.concatenate([
        lons,
        lons[::-1]
    ])

    polygon_lats = np.concatenate([
        outer_lats,
        inner_lats[::-1]
    ])

    return (
        polygon_lons,
        polygon_lats
    )


def wrap_to_width(
    text,
    width
):

    pieces = str(
        text
    ).split("\n")

    wrapped = []

    for piece in pieces:

        wrapped.append(
            textwrap.fill(
                piece,
                width=width,
                break_long_words=False,
                break_on_hyphens=False
            )
        )

    return "\n".join(
        wrapped
    )


def fit_text_in_axes(
    ax,
    text,
    x,
    y,
    max_width_fraction,
    max_height_fraction,
    fontsize_start,
    min_fontsize=6.8,
    step=0.20,
    **kwargs
):

    artist = ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize_start,
        **kwargs
    )

    fig = ax.figure

    fig.canvas.draw()

    renderer = (
        fig.canvas.get_renderer()
    )

    axes_bbox = (
        ax.get_window_extent(
            renderer
        )
    )

    max_width = (
        axes_bbox.width
        * max_width_fraction
    )

    max_height = (
        axes_bbox.height
        * max_height_fraction
    )

    current_size = fontsize_start

    while current_size > min_fontsize:

        text_bbox = (
            artist.get_window_extent(
                renderer
            )
        )

        if (
            text_bbox.width <= max_width
            and text_bbox.height <= max_height
        ):
            break

        current_size -= step

        artist.set_fontsize(
            current_size
        )

        fig.canvas.draw()

        renderer = (
            fig.canvas.get_renderer()
        )

    return artist


def add_callout_box(
    fig,
    rect,
    title,
    subtitle,
    color,
    title_size=12.0,
    subtitle_size=9.2,
    title_wrap=31,
    subtitle_wrap=40
):

    box_ax = fig.add_axes(
        rect,
        zorder=100
    )

    box_ax.set_xlim(
        0,
        1
    )

    box_ax.set_ylim(
        0,
        1
    )

    box_ax.axis(
        "off"
    )

    background = FancyBboxPatch(
        (0.018, 0.018),
        0.964,
        0.964,
        boxstyle=(
            "round,pad=0.020,"
            "rounding_size=0.050"
        ),
        transform=box_ax.transAxes,
        facecolor="white",
        edgecolor=color,
        linewidth=1.9,
        alpha=0.99,
        clip_on=False
    )

    box_ax.add_patch(
        background
    )

    wrapped_title = wrap_to_width(
        title,
        title_wrap
    )

    wrapped_subtitle = wrap_to_width(
        subtitle,
        subtitle_wrap
    )

    fit_text_in_axes(
        box_ax,
        wrapped_title,
        0.50,
        0.73,
        max_width_fraction=0.86,
        max_height_fraction=0.26,
        fontsize_start=title_size,
        ha="center",
        va="center",
        fontweight="bold",
        color=color,
        linespacing=1.02
    )

    fit_text_in_axes(
        box_ax,
        wrapped_subtitle,
        0.50,
        0.34,
        max_width_fraction=0.86,
        max_height_fraction=0.47,
        fontsize_start=subtitle_size,
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
        linespacing=1.07
    )

    return box_ax


def add_common_box(
    fig,
    rect
):

    box_ax = fig.add_axes(
        rect,
        zorder=100
    )

    box_ax.set_xlim(
        0,
        1
    )

    box_ax.set_ylim(
        0,
        1
    )

    box_ax.axis(
        "off"
    )

    background = FancyBboxPatch(
        (0.018, 0.018),
        0.964,
        0.964,
        boxstyle=(
            "round,pad=0.018,"
            "rounding_size=0.045"
        ),
        transform=box_ax.transAxes,
        facecolor="white",
        edgecolor=GRAY_CONCEPT,
        linewidth=1.6,
        alpha=0.99,
        clip_on=False
    )

    box_ax.add_patch(
        background
    )

    fit_text_in_axes(
        box_ax,
        "Common framework",
        0.50,
        0.68,
        max_width_fraction=0.88,
        max_height_fraction=0.24,
        fontsize_start=12.8,
        ha="center",
        va="center",
        fontweight="bold",
        color="black"
    )

    fit_text_in_axes(
        box_ax,
        (
            "All four objectives use the same\n"
            "13 sector-based Antarctic shelf-sea domain"
        ),
        0.50,
        0.31,
        max_width_fraction=0.88,
        max_height_fraction=0.42,
        fontsize_start=9.4,
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
        linespacing=1.08
    )

    return box_ax


def box_point(
    rect,
    x_fraction,
    y_fraction
):

    x0, y0, width, height = rect

    return (
        x0 + width * x_fraction,
        y0 + height * y_fraction
    )


def add_box_to_map_arrow(
    ax,
    start_fig,
    target_lonlat,
    color,
    curve=0.0,
    linewidth=2.35,
    mutation_scale=17,
    zorder=20
):

    ax.annotate(
        "",
        xy=target_lonlat,
        xycoords=(
            ccrs.PlateCarree()
            ._as_mpl_transform(ax)
        ),
        xytext=start_fig,
        textcoords=ax.figure.transFigure,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=1,
            shrinkB=6,
            connectionstyle=(
                f"arc3,rad={curve}"
            ),
            capstyle="round",
            joinstyle="round"
        ),
        annotation_clip=False,
        zorder=zorder
    )


def make_arrow_legend_handle(
    color,
    label
):

    return mlines.Line2D(
        [0, 1],
        [0, 0],
        color=color,
        linewidth=2.6,
        marker=">",
        markevery=[1],
        markersize=7,
        label=label
    )


def set_legend_bold(
    legend
):

    legend.get_title().set_fontweight(
        "bold"
    )

    for text_object in legend.get_texts():

        text_object.set_fontweight(
            "bold"
        )


# 9. ARGO FUNCTIONS

def get_argo_years_from_juld(
    ds
):

    if "JULD" not in ds:
        return None

    juld_da = ds[
        "JULD"
    ]

    juld_values = np.asarray(
        juld_da.values
    )

    years = np.full(
        juld_values.shape,
        -9999,
        dtype=np.int32
    )

    if np.issubdtype(
        juld_values.dtype,
        np.datetime64
    ):

        valid = ~np.isnat(
            juld_values
        )

        if np.any(valid):

            years[valid] = (
                juld_values[valid]
                .astype("datetime64[Y]")
                .astype(int)
                + 1970
            )

        return years

    try:

        numeric_juld = (
            juld_values.astype(float)
        )

    except (
        TypeError,
        ValueError
    ):

        return years

    valid = np.isfinite(
        numeric_juld
    )

    if not np.any(valid):

        return years

    units = str(
        juld_da.attrs.get(
            "units",
            ""
        )
    ).lower()

    if "since" in units:

        try:

            decoded = xr.decode_cf(
                ds[["JULD"]]
            )["JULD"].values

            if np.issubdtype(
                decoded.dtype,
                np.datetime64
            ):

                decoded_valid = ~np.isnat(
                    decoded
                )

                years[decoded_valid] = (
                    decoded[decoded_valid]
                    .astype("datetime64[Y]")
                    .astype(int)
                    + 1970
                )

                return years

        except Exception:
            pass

    seconds = np.rint(
        numeric_juld[valid]
        * 86400.0
    ).astype(
        "timedelta64[s]"
    )

    dates = (
        np.datetime64("1950-01-01")
        + seconds
    )

    years[valid] = (
        dates
        .astype("datetime64[Y]")
        .astype(int)
        + 1970
    )

    return years


def _detect_profile_coordinate(
    ds,
    preferred_names,
    token
):

    available = (
        list(ds.data_vars)
        + list(ds.coords)
    )

    for name in preferred_names:

        if name in available:

            return name

    for name in available:

        if token in name.lower():

            return name

    return None


def load_argo_points(
    path,
    start_year=2008,
    end_year=2025
):

    print(
        "\nLoading Argo profile/support positions..."
    )

    ds = xr.open_dataset(
        path,
        decode_times=False
    )

    lat_name = _detect_profile_coordinate(
        ds,
        [
            "LATITUDE",
            "latitude",
            "lat",
            "profile_lat",
            "PROFILE_LATITUDE"
        ],
        "lat"
    )

    lon_name = _detect_profile_coordinate(
        ds,
        [
            "LONGITUDE",
            "longitude",
            "lon",
            "profile_lon",
            "PROFILE_LONGITUDE"
        ],
        "lon"
    )

    if (
        lat_name is None
        or lon_name is None
    ):

        ds.close()

        raise ValueError(
            "Could not locate paired "
            "Argo LATITUDE/LONGITUDE variables."
        )

    lat_da = ds[
        lat_name
    ]

    lon_da = ds[
        lon_name
    ]

    if lat_da.shape != lon_da.shape:

        ds.close()

        raise ValueError(
            "Argo LATITUDE and LONGITUDE "
            "do not have matching shapes."
        )

    argo_lats = np.asarray(
        lat_da.values,
        dtype=float
    ).ravel()

    argo_lons = normalize_lon(
        np.asarray(
            lon_da.values,
            dtype=float
        ).ravel()
    )

    profile_mask = (
        np.isfinite(argo_lats)
        & np.isfinite(argo_lons)
        & (argo_lats >= -90.0)
        & (argo_lats <= -58.0)
    )

    years = get_argo_years_from_juld(
        ds
    )

    if years is not None:

        years = np.asarray(
            years
        ).ravel()

        if years.size == profile_mask.size:

            profile_mask &= (
                (years >= start_year)
                & (years <= end_year)
            )

    profile_dim = (
        lat_da.dims[0]
        if lat_da.dims
        else None
    )

    for candidate in [
        "TEMP",
        "PSAL",
        "temperature",
        "salinity"
    ]:

        if (
            candidate not in ds.data_vars
            or profile_dim is None
        ):
            continue

        data = ds[
            candidate
        ]

        if profile_dim not in data.dims:
            continue

        reduce_dims = [
            dim
            for dim in data.dims
            if dim != profile_dim
        ]

        finite_data = (
            data.notnull()
        )

        if reduce_dims:

            finite_profile = (
                finite_data.any(
                    dim=reduce_dims
                ).values
            )

        else:

            finite_profile = (
                finite_data.values
            )

        finite_profile = np.asarray(
            finite_profile,
            dtype=bool
        ).ravel()

        if finite_profile.size == profile_mask.size:

            profile_mask &= (
                finite_profile
            )

            break

    point_lons = (
        argo_lons[
            profile_mask
        ]
    )

    point_lats = (
        argo_lats[
            profile_mask
        ]
    )

    ds.close()

    del ds

    gc.collect()

    if point_lons.size == 0:

        raise ValueError(
            "No valid Argo profiles remained "
            "after filtering."
        )

    position_table = pd.DataFrame({
        "lon": (
            np.round(
                point_lons
                / ARGO_ROUNDING_DEG
            )
            * ARGO_ROUNDING_DEG
        ),

        "lat": (
            np.round(
                point_lats
                / ARGO_ROUNDING_DEG
            )
            * ARGO_ROUNDING_DEG
        ),
    }).dropna().drop_duplicates()

    if len(
        position_table
    ) > MAX_ARGO_POINTS:

        position_table = (
            position_table.sample(
                n=MAX_ARGO_POINTS,
                random_state=RANDOM_SEED
            )
        )

    print(
        "Argo plotting points:",
        f"{len(position_table):,}"
    )

    return (
        position_table["lon"].to_numpy(),
        position_table["lat"].to_numpy()
    )


# 10. BIODIVERSITY OCCURRENCE LOADER
def load_occurrence_points(
    path,
    max_points=950,
    random_seed=42
):

    print(
        "\nLoading biodiversity occurrence points..."
    )

    header_columns = pd.read_csv(
        path,
        nrows=0
    ).columns.tolist()

    if (
        "lat" in header_columns
        and "lon" in header_columns
    ):

        lat_col = "lat"
        lon_col = "lon"

    elif (
        "latitude" in header_columns
        and "longitude" in header_columns
    ):

        lat_col = "latitude"
        lon_col = "longitude"

    elif (
        "lat_bin" in header_columns
        and "lon_bin" in header_columns
    ):

        lat_col = "lat_bin"
        lon_col = "lon_bin"

    else:

        raise ValueError(
            "Occurrence CSV must contain "
            "lat/lon, latitude/longitude, "
            "or lat_bin/lon_bin."
        )

    use_columns = [
        lat_col,
        lon_col
    ]

    if "year" in header_columns:

        use_columns.append(
            "year"
        )

    points = pd.read_csv(
        path,
        usecols=use_columns
    )

    points[lat_col] = pd.to_numeric(
        points[lat_col],
        errors="coerce"
    )

    points[lon_col] = pd.to_numeric(
        points[lon_col],
        errors="coerce"
    )

    if "year" in points.columns:

        points["year"] = pd.to_numeric(
            points["year"],
            errors="coerce"
        )

    points = points.dropna(
        subset=[
            lat_col,
            lon_col
        ]
    )

    points[lon_col] = normalize_lon(
        points[lon_col].to_numpy()
    )

    spatial_mask = (
        (points[lat_col] >= -90.0)
        & (points[lat_col] <= -58.0)
        & (points[lon_col] >= -180.0)
        & (points[lon_col] <= 180.0)
    )

    points = points.loc[
        spatial_mask
    ].copy()

    if "year" in points.columns:

        points = points.loc[
            (points["year"] >= 2008)
            & (points["year"] <= 2025)
        ].copy()

    points = points.drop_duplicates(
        subset=[
            lat_col,
            lon_col
        ]
    )

    if len(
        points
    ) > max_points:

        points = points.sample(
            n=max_points,
            random_state=random_seed,
            replace=False
        )

    print(
        "Biodiversity plotting points:",
        f"{len(points):,}"
    )

    return (
        points[lon_col].to_numpy(
            dtype=np.float32
        ),
        points[lat_col].to_numpy(
            dtype=np.float32
        )
    )


# 11. VERIFY REQUIRED FILES
required_files = {
    "Bathymetry": bathy_file,
    "Sea ice": sic_file,
    "Argo": argo_file,
    "Biodiversity occurrence": occurrence_file,
}

missing_files = []

for label, path in required_files.items():

    exists = os.path.isfile(
        path
    )

    print(
        f"{label} file exists: {exists}"
    )

    print(
        f"  {path}"
    )

    if not exists:

        missing_files.append(
            f"{label}: {path}"
        )

if missing_files:

    raise FileNotFoundError(
        "\nThe following required file(s) "
        "were not found:\n\n"
        + "\n".join(
            missing_files
        )
        + "\n\nThese are the paths from "
        "your previous working codes."
    )


# 12. LOAD GEBCO BATHYMETRY
print(
    "\nLoading bathymetry..."
)

ds_bathy = xr.open_dataset(
    bathy_file,
    chunks={}
)

bathy_lat, bathy_lon = get_coord_names(
    ds_bathy
)

if "elevation" in ds_bathy.data_vars:

    bathy_var = "elevation"

else:

    bathy_var = get_first_numeric_var(
        ds_bathy
    )

ds_bathy = subset_antarctic(
    ds_bathy,
    bathy_lat,
    south=-90,
    north=-58
)

bathy_skip = 16

bathy_da = ds_bathy[
    bathy_var
].isel({
    bathy_lat: slice(
        None,
        None,
        bathy_skip
    ),

    bathy_lon: slice(
        None,
        None,
        bathy_skip
    )

}).astype(
    "float32"
).load()

bathy_da = maybe_sort_lon(
    bathy_da,
    bathy_lon
)

bathy_lats = np.asarray(
    bathy_da[
        bathy_lat
    ].values
)

bathy_lons = np.asarray(
    bathy_da[
        bathy_lon
    ].values
)

z_ocean = bathy_da.where(
    bathy_da < 0
)

ds_bathy.close()

del ds_bathy

gc.collect()

print(
    "Bathymetry loaded:",
    z_ocean.shape
)


# 13. LOAD 2008-2025 OSTIA SIC
print(
    "\nLoading sea-ice concentration..."
)

ds_sic = xr.open_dataset(
    sic_file,
    chunks={}
)

sic_lat, sic_lon = get_coord_names(
    ds_sic
)

sic_time = get_time_name(
    ds_sic
)

ds_sic = subset_antarctic(
    ds_sic,
    sic_lat,
    south=-90,
    north=-58
)

possible_sic_variables = [
    "siconc",
    "sif",
    "SIF",
    "sea_ice_fraction",
    "sea_ice_concentration",
    "ice_conc",
    "sic",
    "SIC",
]

sic_var = next(
    (
        variable
        for variable in possible_sic_variables
        if variable in ds_sic.data_vars
    ),
    None
)

if sic_var is None:

    sic_var = get_first_numeric_var(
        ds_sic
    )

sic_skip = 4

sic = ds_sic[
    sic_var
].isel({
    sic_lat: slice(
        None,
        None,
        sic_skip
    ),

    sic_lon: slice(
        None,
        None,
        sic_skip
    )

}).astype(
    "float32"
)

if float(
    sic.max(
        skipna=True
    )
) <= 1.5:

    sic = (
        sic * 100.0
    )

if (
    sic_time is not None
    and sic_time in sic.dims
):

    try:

        sic = sic.sel({
            sic_time: slice(
                "2008-01-01",
                "2025-12-31"
            )
        })

    except Exception:
        pass

    sic_mean = sic.mean(
        dim=sic_time,
        skipna=True
    ).load()

else:

    sic_mean = sic.load()

sic_mean = maybe_sort_lon(
    sic_mean,
    sic_lon
)

sic_lats = np.asarray(
    sic_mean[
        sic_lat
    ].values
)

sic_lons = np.asarray(
    sic_mean[
        sic_lon
    ].values
)

ds_sic.close()

del ds_sic, sic

gc.collect()

print(
    "SIC variable:",
    sic_var
)

print(
    "Annual SIC shape:",
    sic_mean.shape
)


# 14. LOAD ARGO AND GBIF POINTS
argo_plot_lon, argo_plot_lat = load_argo_points(
    argo_file,
    start_year=2008,
    end_year=2025
)

occurrence_lons, occurrence_lats = load_occurrence_points(
    occurrence_file,
    max_points=MAX_OCCURRENCE_POINTS,
    random_seed=RANDOM_SEED
)


# 15. CREATE FIGURE
fig = plt.figure(
    figsize=(13.0, 9.5),
    dpi=180,
    facecolor="white"
)

projection = ccrs.SouthPolarStereo()

ax = fig.add_axes(
    MAP_RECT,
    projection=projection,
    zorder=5
)

ax.set_extent(
    [-180, 180, -90, -58],
    crs=ccrs.PlateCarree()
)


# 16. CIRCULAR POLAR BOUNDARY
theta = np.linspace(
    0,
    2 * np.pi,
    720
)

circle_vertices = np.vstack([
    np.sin(theta),
    np.cos(theta)
]).T

circle_vertices = (
    circle_vertices
    * 0.5
    + [0.5, 0.5]
)

ax.set_boundary(
    mpath.Path(
        circle_vertices
    ),
    transform=ax.transAxes
)


# 17. BATHYMETRY BACKGROUND
bathy_levels = [
    -8000,
    -7000,
    -6000,
    -5000,
    -4000,
    -3000,
    -2000,
    -1500,
    -1000,
    -500,
    -200,
    0,
]

bath = ax.contourf(
    bathy_lons,
    bathy_lats,
    z_ocean,
    levels=bathy_levels,
    cmap="Blues_r",
    extend="min",
    transform=ccrs.PlateCarree(),
    zorder=1
)

try:

    bath.set_rasterized(
        True
    )

except Exception:
    pass


# 18. SECTOR SHADING
sector_handles = []

for index, sector in enumerate(
    sea_info
):

    (
        abbreviation,
        full_name,
        longitude_text,
        lon1,
        lon2
    ) = sector

    sector_color = (
        sector_colors[index]
    )

    for (
        part_lon1,
        part_lon2
    ) in get_sector_parts(
        lon1,
        lon2
    ):

        (
            polygon_lons,
            polygon_lats
        ) = make_sector_polygon(
            part_lon1,
            part_lon2,
            lat_inner=-82.0,
            lat_outer=-58.0,
            n=1500
        )

        ax.fill(
            polygon_lons,
            polygon_lats,
            facecolor=sector_color,
            edgecolor="white",
            linewidth=0.28,
            alpha=0.18,
            transform=ccrs.PlateCarree(),
            zorder=2.5
        )

    sector_handles.append(
        mpatches.Patch(
            facecolor=sector_color,
            edgecolor=sector_color,
            alpha=0.72,
            label=(
                f"{abbreviation} – "
                f"{full_name} – "
                f"({longitude_text})"
            )
        )
    )


# 19. BATHYMETRIC CONTOURS
ax.contour(
    bathy_lons,
    bathy_lats,
    z_ocean,
    levels=[-2000],
    colors=[
        SHELFBREAK_COLOR
    ],
    linewidths=1.45,
    linestyles="-",
    transform=ccrs.PlateCarree(),
    zorder=5
)

ax.contour(
    bathy_lons,
    bathy_lats,
    z_ocean,
    levels=[
        -4000,
        -3000,
        -1000,
        -500
    ],
    colors="dimgray",
    linewidths=0.32,
    alpha=0.28,
    transform=ccrs.PlateCarree(),
    zorder=4
)


# 20. ANNUAL-MEAN SIC CONTOURS
ax.contour(
    sic_lons,
    sic_lats,
    sic_mean,
    levels=[15],
    colors=SIC_RED,
    linewidths=1.65,
    linestyles="-",
    transform=ccrs.PlateCarree(),
    zorder=7
)

ax.contour(
    sic_lons,
    sic_lats,
    sic_mean,
    levels=[50],
    colors=SIC_RED,
    linewidths=1.25,
    linestyles="--",
    transform=ccrs.PlateCarree(),
    zorder=7
)

ax.contour(
    sic_lons,
    sic_lats,
    sic_mean,
    levels=[80],
    colors=SIC_RED,
    linewidths=1.10,
    linestyles=":",
    transform=ccrs.PlateCarree(),
    zorder=7
)


# 21. ARGO AND BIODIVERSITY POINTS
# Argo = small black dots
ax.scatter(
    argo_plot_lon,
    argo_plot_lat,
    s=3.0,
    marker="o",
    c="black",
    alpha=0.72,
    linewidths=0,
    transform=ccrs.PlateCarree(),
    zorder=8,
    rasterized=True
)

# Biodiversity occurrence =
# yellow with black outline and larger than Argo
ax.scatter(
    occurrence_lons,
    occurrence_lats,
    s=22,
    marker="o",
    facecolors="#FFD700",
    edgecolors="black",
    linewidths=0.65,
    alpha=0.92,
    transform=ccrs.PlateCarree(),
    zorder=9,
    rasterized=True
)


# 22. LAND, COASTLINES AND GRIDLINES
ax.add_feature(
    cfeature.LAND,
    facecolor="#D7D7D7",
    edgecolor="black",
    linewidth=0.60,
    zorder=10
)

ax.coastlines(
    resolution="110m",
    linewidth=0.70,
    color="black",
    zorder=11
)

ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.30,
    color="gray",
    alpha=0.22,
    linestyle="--",
    zorder=3
)


# 23. SEA LABELS

for (
    abbreviation,
    _,
    _,
    _,
    _
) in sea_info:

    (
        label_lon,
        label_lat
    ) = label_positions[
        abbreviation
    ]

    ax.text(
        label_lon,
        label_lat,
        abbreviation,
        transform=ccrs.PlateCarree(),
        fontsize=11.0,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="black",
            linewidth=0.55,
            alpha=0.98
        ),
        zorder=30,
        clip_on=False
    )


# 24. LATITUDE AND LONGITUDE LABELS
ax.text(
    178,
    -80,
    "80°S",
    transform=ccrs.PlateCarree(),
    fontsize=12,
    fontweight="bold",
    ha="right",
    va="center",
    color="dimgray",
    zorder=31,
    clip_on=False
)

ax.text(
    178,
    -60,
    "60°S",
    transform=ccrs.PlateCarree(),
    fontsize=10.5,
    fontweight="bold",
    ha="right",
    va="center",
    color="dimgray",
    zorder=31,
    clip_on=False
)

for (
    longitude_text,
    position
) in outside_lon_labels.items():

    (
        x_position,
        y_position
    ) = position

    ax.text(
        x_position,
        y_position,
        longitude_text,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=31,
        clip_on=False
    )


# 25. LOCATION INSET
inset_ax = fig.add_axes(
    INSET_RECT,
    projection=ccrs.Orthographic(
        0,
        -90
    ),
    zorder=50
)

inset_ax.set_global()

inset_ax.add_feature(
    cfeature.OCEAN,
    facecolor="#DDEEFF",
    zorder=0
)

inset_ax.add_feature(
    cfeature.LAND,
    facecolor="#D4D4D4",
    edgecolor="black",
    linewidth=0.35,
    zorder=1
)

inset_ax.coastlines(
    resolution="110m",
    linewidth=0.35,
    zorder=2
)

belt_lons = np.linspace(
    -180,
    180,
    720
)

belt_outer = np.full_like(
    belt_lons,
    -58.0
)

belt_inner = np.full_like(
    belt_lons,
    -90.0
)

belt_polygon_lons = np.concatenate([
    belt_lons,
    belt_lons[::-1]
])

belt_polygon_lats = np.concatenate([
    belt_outer,
    belt_inner[::-1]
])

inset_ax.fill(
    belt_polygon_lons,
    belt_polygon_lats,
    facecolor="none",
    edgecolor="red",
    linewidth=1.15,
    transform=ccrs.PlateCarree(),
    zorder=4
)

inset_ax.plot(
    [0, 0],
    [-90, -58],
    color="red",
    linewidth=1.15,
    transform=ccrs.PlateCarree(),
    zorder=5
)

inset_ax.set_title(
    "Location",
    fontsize=13.5,
    fontweight="bold",
    pad=2
)

inset_theta = np.linspace(
    0,
    2 * np.pi,
    360
)

inset_vertices = np.vstack([
    np.sin(inset_theta),
    np.cos(inset_theta)
]).T

inset_vertices = (
    inset_vertices
    * 0.5
    + [0.5, 0.5]
)

inset_ax.set_boundary(
    mpath.Path(
        inset_vertices
    ),
    transform=inset_ax.transAxes
)


# 26. INTEGRATED CONCEPT CALLOUT BOXES

add_callout_box(
    fig,
    GREEN_BOX_RECT,
    "Atlantic-sector coupled hotspot",
    (
        "WED–KHV–RLS: strong sea-ice seasonality, "
        "active upper-ocean coupling, productive summer shelf zones, "
        "and comparatively high biodiversity support"
    ),
    GREEN_CONCEPT,
    title_size=12.0,
    subtitle_size=9.2,
    title_wrap=31,
    subtitle_wrap=42
)

add_callout_box(
    fig,
    TEAL_BOX_RECT,
    "East Antarctic transition corridor",
    (
        "LAZ–COS–COO–DAV: marked current and sea-ice gradients, "
        "variable sea-level support, mixed bloom timing, "
        "and regional biodiversity contrasts"
    ),
    TEAL_CONCEPT,
    title_size=11.7,
    subtitle_size=9.0,
    title_wrap=33,
    subtitle_wrap=40
)

add_callout_box(
    fig,
    ORANGE_BOX_RECT,
    "Marginal-ice biological edge",
    (
        "BEL: open-water access near the marginal-ice zone supports "
        "strong sampling, productive blooms, and elevated "
        "biodiversity indicators"
    ),
    ORANGE_CONCEPT,
    title_size=11.2,
    subtitle_size=8.7,
    title_wrap=26,
    subtitle_wrap=28
)

add_callout_box(
    fig,
    RED_BOX_RECT,
    "Pacific-sector contrast",
    (
        "ROS–AMU: pronounced retreat and freshwater-structure contrasts, "
        "distinct hydrographic and sea-level behaviour, "
        "and strong productivity–carbon responses"
    ),
    RED_CONCEPT,
    title_size=11.5,
    subtitle_size=8.9,
    title_wrap=30,
    subtitle_wrap=36
)

add_callout_box(
    fig,
    BLUE_BOX_RECT,
    "Sector-dependent eastern gradient",
    (
        "MAW–DUR–SOM: varying current influence, steric structure, "
        "retreat timing, and biodiversity response across East Antarctica"
    ),
    BLUE_CONCEPT,
    title_size=10.9,
    subtitle_size=8.8,
    title_wrap=31,
    subtitle_wrap=35
)

add_common_box(
    fig,
    COMMON_BOX_RECT
)


# 27. CONCEPT ARROWS
# GREEN
add_box_to_map_arrow(
    ax,
    box_point(
        GREEN_BOX_RECT,
        0.20,
        0.00
    ),
    arrow_targets["WED"],
    GREEN_CONCEPT,
    curve=0.10
)

add_box_to_map_arrow(
    ax,
    box_point(
        GREEN_BOX_RECT,
        0.50,
        0.00
    ),
    arrow_targets["KHV"],
    GREEN_CONCEPT,
    curve=0.025
)

add_box_to_map_arrow(
    ax,
    box_point(
        GREEN_BOX_RECT,
        0.82,
        0.00
    ),
    arrow_targets["RLS"],
    GREEN_CONCEPT,
    curve=-0.035
)

# TEAL
add_box_to_map_arrow(
    ax,
    box_point(
        TEAL_BOX_RECT,
        0.16,
        0.00
    ),
    arrow_targets["LAZ"],
    TEAL_CONCEPT,
    curve=-0.08
)

add_box_to_map_arrow(
    ax,
    box_point(
        TEAL_BOX_RECT,
        0.39,
        0.00
    ),
    arrow_targets["COS"],
    TEAL_CONCEPT,
    curve=-0.04
)

add_box_to_map_arrow(
    ax,
    box_point(
        TEAL_BOX_RECT,
        0.63,
        0.00
    ),
    arrow_targets["COO"],
    TEAL_CONCEPT,
    curve=0.005
)

add_box_to_map_arrow(
    ax,
    box_point(
        TEAL_BOX_RECT,
        0.86,
        0.00
    ),
    arrow_targets["DAV"],
    TEAL_CONCEPT,
    curve=0.050
)

# ORANGE
add_box_to_map_arrow(
    ax,
    box_point(
        ORANGE_BOX_RECT,
        0.995,
        0.52
    ),
    arrow_targets["BEL"],
    ORANGE_CONCEPT,
    curve=-0.095,
    linewidth=2.45
)

# RED
add_box_to_map_arrow(
    ax,
    box_point(
        RED_BOX_RECT,
        1.00,
        0.70
    ),
    arrow_targets["AMU"],
    RED_CONCEPT,
    curve=-0.10,
    linewidth=2.50,
    mutation_scale=18
)

add_box_to_map_arrow(
    ax,
    box_point(
        RED_BOX_RECT,
        1.00,
        0.30
    ),
    arrow_targets["ROS"],
    RED_CONCEPT,
    curve=-0.045,
    linewidth=2.50,
    mutation_scale=18
)

# BLUE
add_box_to_map_arrow(
    ax,
    box_point(
        BLUE_BOX_RECT,
        0.18,
        1.00
    ),
    arrow_targets["SOM"],
    BLUE_CONCEPT,
    curve=0.13
)

add_box_to_map_arrow(
    ax,
    box_point(
        BLUE_BOX_RECT,
        0.50,
        1.00
    ),
    arrow_targets["DUR"],
    BLUE_CONCEPT,
    curve=0.055
)

add_box_to_map_arrow(
    ax,
    box_point(
        BLUE_BOX_RECT,
        0.82,
        1.00
    ),
    arrow_targets["MAW"],
    BLUE_CONCEPT,
    curve=-0.035
)


# 28. DATA NOTE BOX
note_ax = fig.add_axes(
    NOTE_BOX_RECT,
    zorder=100
)

note_ax.set_xlim(
    0,
    1
)

note_ax.set_ylim(
    0,
    1
)

note_ax.axis(
    "off"
)

note_background = FancyBboxPatch(
    (0.01, 0.01),
    0.98,
    0.98,
    boxstyle=(
        "round,pad=0.02,"
        "rounding_size=0.035"
    ),
    transform=note_ax.transAxes,
    facecolor="white",
    edgecolor="gray",
    linewidth=0.85,
    alpha=0.97
)

note_ax.add_patch(
    note_background
)

note_text = (
    f"Bathymetry: {BATHY_SOURCE_LABEL}\n"
    "Mean SIC: OSTIA 2008–2025\n"
    "SIC contours: 15%, 50%, 80%\n"
    "Hydrography and sea level: Argo, EN4, DUACS\n"
    "Biogeochemistry: Ocean-color products, GlobColour PFT\n"
    f"Biodiversity: {OCCURRENCE_SOURCE_LABEL}\n"
    "Study period: 2008–2025"
)

fit_text_in_axes(
    note_ax,
    note_text,
    0.04,
    0.52,
    max_width_fraction=0.90,
    max_height_fraction=0.88,
    fontsize_start=8.5,
    ha="left",
    va="center",
    fontweight="bold",
    color="black",
    linespacing=1.06
)


# 29. CENTERED BATHYMETRY COLORBAR

colorbar_ax = fig.add_axes(
    COLORBAR_RECT,
    zorder=80
)

colorbar = plt.colorbar(
    bath,
    cax=colorbar_ax,
    orientation="horizontal"
)

colorbar.set_ticks([
    -7000,
    -5000,
    -3000,
    -1500,
    -500,
    0
])

colorbar.set_label(
    "Bathymetry / Elevation (m)",
    fontsize=12.2,
    fontweight="bold",
    labelpad=4
)

colorbar.ax.tick_params(
    labelsize=9.5,
    width=0.9,
    length=4
)

for tick_label in (
    colorbar.ax.get_xticklabels()
):

    tick_label.set_fontweight(
        "bold"
    )


# 30. MAP-LAYERS LEGEND

land_handle = mpatches.Patch(
    facecolor="#D7D7D7",
    edgecolor="black",
    label="Antarctic land mask"
)

coastline_handle = mlines.Line2D(
    [],
    [],
    color="black",
    linewidth=0.8,
    label="Coastline"
)

shelf_handle = mlines.Line2D(
    [],
    [],
    color=SHELFBREAK_COLOR,
    linewidth=1.45,
    linestyle="-",
    label="2000 m isobath / shelf break"
)

sic15_handle = mlines.Line2D(
    [],
    [],
    color=SIC_RED,
    linewidth=1.65,
    linestyle="-",
    label=(
        "Annual mean 15% SIC /\n"
        "sea-ice edge"
    )
)

sic50_handle = mlines.Line2D(
    [],
    [],
    color=SIC_RED,
    linewidth=1.25,
    linestyle="--",
    label=(
        "Annual mean 50% SIC /\n"
        "intermediate ice zone"
    )
)

sic80_handle = mlines.Line2D(
    [],
    [],
    color=SIC_RED,
    linewidth=1.10,
    linestyle=":",
    label=(
        "Annual mean 80% SIC /\n"
        "compact ice zone"
    )
)

argo_handle = mlines.Line2D(
    [],
    [],
    linestyle="None",
    marker="o",
    markersize=3.8,
    markerfacecolor="black",
    markeredgecolor="black",
    label=(
        "Argo / hydrography support\n"
        "(2008–2025)"
    )
)

occurrence_handle = mlines.Line2D(
    [],
    [],
    linestyle="None",
    marker="o",
    markersize=7.2,
    markerfacecolor="#FFD700",
    markeredgecolor="black",
    markeredgewidth=0.8,
    label=(
        "Biodiversity occurrence support\n"
        "(2008–2025)"
    )
)

map_legend_ax = fig.add_axes(
    MAP_LEGEND_RECT,
    zorder=90
)

map_legend_ax.axis(
    "off"
)

map_legend = map_legend_ax.legend(
    handles=[
        land_handle,
        coastline_handle,
        shelf_handle,
        sic15_handle,
        sic50_handle,
        sic80_handle,
        argo_handle,
        occurrence_handle,
    ],
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    borderaxespad=0.0,
    title="Map layers",
    title_fontsize=13.0,
    fontsize=8.6,
    frameon=True,
    fancybox=True,
    framealpha=0.98,
    borderpad=0.65,
    labelspacing=0.35,
    handlelength=2.20
)

set_legend_bold(
    map_legend
)


# 31. SECTOR LEGEND

sector_legend_ax = fig.add_axes(
    SECTOR_LEGEND_RECT,
    zorder=90
)

sector_legend_ax.axis(
    "off"
)

sector_legend = (
    sector_legend_ax.legend(
        handles=sector_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.0,
        title="Antarctic Shelf Sea Sectors",
        title_fontsize=12.0,
        fontsize=7.8,
        frameon=True,
        fancybox=True,
        framealpha=0.98,
        borderpad=0.60,
        labelspacing=0.24,
        handlelength=1.50
    )
)

set_legend_bold(
    sector_legend
)


# 32. INTEGRATED CONCEPT LEGEND

concept_handles = [

    make_arrow_legend_handle(
        GREEN_CONCEPT,
        "Atlantic-sector coupled hotspot"
    ),

    make_arrow_legend_handle(
        TEAL_CONCEPT,
        "East Antarctic transition corridor"
    ),

    make_arrow_legend_handle(
        ORANGE_CONCEPT,
        "Marginal-ice biological edge"
    ),

    make_arrow_legend_handle(
        RED_CONCEPT,
        "Pacific-sector contrast"
    ),

    make_arrow_legend_handle(
        BLUE_CONCEPT,
        "Sector-dependent eastern gradient"
    ),

    make_arrow_legend_handle(
        GRAY_CONCEPT,
        "Common framework"
    ),
]

concept_legend_ax = fig.add_axes(
    CONCEPT_LEGEND_RECT,
    zorder=90
)

concept_legend_ax.axis(
    "off"
)

concept_legend = (
    concept_legend_ax.legend(
        handles=concept_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.0,
        title="Integrated concept-guided annotations",
        title_fontsize=10.7,
        fontsize=7.9,
        frameon=True,
        fancybox=True,
        framealpha=0.98,
        borderpad=0.62,
        labelspacing=0.34,
        handlelength=2.15
    )
)

set_legend_bold(
    concept_legend
)


# 33. FINAL DRAW AND EXPORT
fig.canvas.draw()

print(
    f"\nSaving {EXPORT_DPI}-dpi PNG..."
)

plt.savefig(
    out_png,
    dpi=EXPORT_DPI,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
    edgecolor="none"
)

print(
    "Saving vector PDF..."
)

plt.savefig(
    out_pdf,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
    edgecolor="none"
)

plt.show()

print(
    "\nCompleted successfully."
)

print(
    "Saved PNG:",
    out_png
)

print(
    "Saved PDF:",
    out_pdf
)
