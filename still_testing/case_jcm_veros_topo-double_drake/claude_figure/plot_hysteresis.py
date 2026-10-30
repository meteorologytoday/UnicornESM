"""
Plot the AMOC-like hysteresis curve for the 0-60E hosing basin at 40N.

Computes the meridional overturning streamfunction from the ocean 'v' field
at 40N, restricted to the 0-60E basin, takes the dominant (most negative)
cell as the overturning strength, and plots it against the freshwater
hosing forcing for the pre-hosing, ramp-up, and ramp-down periods.

Reads every 10th year of output to keep load time down; set STRIDE=1 for
full resolution.
"""
import glob
import re

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_DIR = "/home/tienyiao/projects_local/project_jax-esm/MetESM/case_jcm_veros_topo-double_drake"
OUTPUT_DIR = f"{CASE_DIR}/output_T31/hosing_run"
FIGURE_DIR = f"{CASE_DIR}/claude_figure"

STRIDE = 1           # use every Nth year of output
LAT_TARGET = 40.0     # degrees N
LON_MIN, LON_MAX = 0.0, 60.0   # hosing basin bounds
PLOT_YEAR_MIN = 0     # earliest year to include in the plot
RAMP_UP_START = 1000
RAMP_DOWN_START = 2000
EARTH_RADIUS = 6.371e6


def compute_amoc_strength():
    all_files = sorted(glob.glob(f"{OUTPUT_DIR}/ocn-*.nc"))
    files = [
        f for f in all_files
        if int(re.search(r"ocn-(\d+)\.nc", f).group(1)) % STRIDE == 0
    ]

    ds0 = xr.open_dataset(files[0], decode_cf=False)
    lon = ds0.lon.values
    lat = ds0.lat.values
    dzt = ds0.dzt.values
    mask_T = ds0.mask_T.values
    ds0.close()

    dlon_rad = np.deg2rad(3.75)
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    lat_idx = int(np.abs(lat - LAT_TARGET).argmin())
    dx = EARTH_RADIUS * np.cos(np.deg2rad(lat[lat_idx])) * dlon_rad
    mask_band = mask_T[lon_mask, lat_idx, :]

    years = []
    amoc_min = []
    for f in files:
        year = int(re.search(r"ocn-(\d+)\.nc", f).group(1))
        ds = xr.open_dataset(f, decode_cf=False)
        v = ds["v"].isel(time=0).values
        ds.close()
        v_band = v[lon_mask, lat_idx, :]
        transport_per_level = np.sum(v_band * mask_band * dx, axis=0) * dzt
        psi = np.cumsum(transport_per_level)
        years.append(year)
        amoc_min.append(psi.min() / 1e6)  # Sv

    years = np.array(years)
    amoc_min = np.array(amoc_min)

    forcing = np.full_like(years, np.nan, dtype=float)
    for i, year in enumerate(years):
        ds = xr.open_dataset(f"{OUTPUT_DIR}/hose-{year:05d}.nc", decode_cf=False)
        forcing[i] = float(ds["freshwater_forcing"].values[0])
        ds.close()

    return years, -amoc_min, forcing  # strength = -min(psi)


def plot_hysteresis(years, strength, forcing, out_path=f"{FIGURE_DIR}/hysteresis_060E_800.png"):
    mask = years > PLOT_YEAR_MIN
    y, f, s = years[mask], forcing[mask], strength[mask]

    pre = y <= RAMP_UP_START
    up = (y > RAMP_UP_START) & (y <= RAMP_DOWN_START)
    down = y >= RAMP_DOWN_START

    fig, ax = plt.subplots(figsize=(7.5, 6))
    # points within each segment are already in chronological order (years sorted ascending),
    # so connecting them traces the actual forcing path over time
    ax.plot(f[pre], s[pre], c="#718096", lw=0.8, alpha=0.6, zorder=1)
    ax.plot(f[up], s[up], c="#2b6cb0", lw=0.8, alpha=0.6, zorder=1)
    ax.plot(f[down], s[down], c="#c53030", lw=0.8, alpha=0.6, zorder=1)

    ax.scatter(f[pre], s[pre], c="#718096", s=14, alpha=0.8,
               label=f"pre-hosing (years {PLOT_YEAR_MIN}-{RAMP_UP_START})", zorder=2)
    ax.scatter(f[up], s[up], c="#2b6cb0", s=14, alpha=0.8,
               label=f"ramp-up (years {RAMP_UP_START}-{RAMP_DOWN_START})", zorder=2)
    ax.scatter(f[down], s[down], c="#c53030", s=14, alpha=0.8,
               label=f"ramp-down (years {RAMP_DOWN_START}-{int(y.max())})", zorder=2)

    ax.set_xlabel("Freshwater hosing forcing (Sv)")
    ax.set_ylabel("Overturning transport, dominant cell (Sv)")
    ax.set_title(
        f"AMOC-like hysteresis: {int(LON_MIN)}-{int(LON_MAX)}°E basin, {int(LAT_TARGET)}°N\n"
        f"(every {STRIDE}th year, years {PLOT_YEAR_MIN}-{int(y.max())})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_timeseries(years, strength, forcing, start_year=500,
                     out_path=f"{FIGURE_DIR}/amoc_strength_timeseries.png"):
    mask = years >= start_year
    y, f, s = years[mask], forcing[mask], strength[mask]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(y, s, c="#2b6cb0", lw=1.1)
    ax.axvline(RAMP_UP_START, color="#c53030", lw=0.8, ls=":")
    ax.axvline(RAMP_DOWN_START, color="#c53030", lw=0.8, ls=":")
    ax.text(RAMP_UP_START, ax.get_ylim()[1] * 0.98, " ramp-up\n start", fontsize=7.5,
            color="#c53030", va="top", ha="left")
    ax.text(RAMP_DOWN_START, ax.get_ylim()[1] * 0.98, " ramp-down\n start", fontsize=7.5,
            color="#c53030", va="top", ha="left")

    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Overturning transport, dominant cell (Sv)")
    ax.set_title(
        f"AMOC-like strength timeseries: {int(LON_MIN)}-{int(LON_MAX)}°E basin, {int(LAT_TARGET)}°N\n"
        f"(every {STRIDE}th year, years {start_year}-{int(y.max())})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    years, strength, forcing = compute_amoc_strength()
    plot_hysteresis(years, strength, forcing)
    plot_timeseries(years, strength, forcing)
