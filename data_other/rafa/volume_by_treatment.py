# -*- coding: utf-8 -*-
"""
Built volume change over time by GiveDirectly treatment intensity, Siaya.

Takes the Open Buildings 2.5D Temporal exports (one GeoTIFF per year, bands
building_height and building_presence, 4 m pixels, EPSG:32636) and the
beyond-nightlight treatment map (fig-map.csv, 0.005 degree cells), and:

  1. computes built volume per 4 m pixel as height x pixel_area, keeping only
     pixels where building_presence clears a threshold,
  2. aggregates that to the same 0.005 degree grid the treatment map uses,
  3. joins on treatment intensity bin,
  4. plots mean volume per cell over time, one line per bin, in levels and
     indexed to the first year.

Outputs, written next to the inputs:
  siaya_volume_by_cell.csv      cell x year panel, volume and built area
  siaya_volume_by_bin.csv       bin x year summary
  siaya_volume_by_treatment.png the figure

Requires: rasterio, pyproj, pandas, numpy, matplotlib.
    pip install rasterio pandas matplotlib

Usage:
    python volume_by_treatment.py
    python volume_by_treatment.py --presence-threshold 0.5 --weight binary
"""

import argparse
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

FOLDER = r"G:\Mi unidad\RESEARCH\beyond-nightlight"
TIF_GLOB = "siaya_buildings_*.tif"
CSV_NAME = "fig-map.csv"

GRID_DEG = 0.005          # must match fig-map.csv
NODATA_SENTINEL = -9999   # what the export used in .unmask()
BLOCK = 2048              # pixels per side when reading in chunks

BINS = ["0-3", "4-6", "7-9", "10-12", "13-"]
BIN_LABELS = {
    "0-3": "0 to 3",
    "4-6": "4 to 6",
    "7-9": "7 to 9",
    "10-12": "10 to 12",
    "13-": "13 or more",
}

# Latitude offset so cell indices stay non negative in the packed key.
LAT_OFFSET = 100_000


# ---------------------------------------------------------------------------
# Raster aggregation
# ---------------------------------------------------------------------------


def band_indices(src):
    """Find height and presence bands by description, fall back to 1 and 2."""
    desc = [d.lower() if d else "" for d in src.descriptions]
    h = next((i + 1 for i, d in enumerate(desc) if "height" in d), 1)
    p = next((i + 1 for i, d in enumerate(desc) if "presence" in d), 2)
    return h, p


def aggregate_raster(path, presence_threshold, weight, verbose=True):
    """Sum built volume and built area into 0.005 degree cells.

    Returns a dict keyed by packed cell index -> (volume_m3, built_area_m2).
    Only pixels that clear the presence threshold are counted, so empty land
    contributes nothing rather than contributing noise.
    """
    sums = {}

    with rasterio.open(path) as src:
        h_band, p_band = band_indices(src)
        pixel_area = abs(src.transform.a * src.transform.e)  # m2, CRS is metric
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        if verbose:
            print(f"  {os.path.basename(path)}: {src.width} x {src.height} px, "
                  f"{pixel_area:.1f} m2/px, bands h={h_band} p={p_band}")

        n_blocks = 0
        for row0 in range(0, src.height, BLOCK):
            for col0 in range(0, src.width, BLOCK):
                win = Window(col0, row0,
                             min(BLOCK, src.width - col0),
                             min(BLOCK, src.height - row0))

                h = src.read(h_band, window=win).astype("float64")
                p = src.read(p_band, window=win).astype("float64")

                valid = (
                    (h > 0) & (h != NODATA_SENTINEL) & np.isfinite(h)
                    & (p >= presence_threshold) & (p <= 1.0) & np.isfinite(p)
                )
                if not valid.any():
                    continue

                rr, cc = np.nonzero(valid)
                # Pixel centres in the raster CRS.
                xs, ys = rasterio.transform.xy(
                    src.transform, rr + row0, cc + col0, offset="center")
                lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))

                w = p[valid] if weight == "probability" else 1.0
                vol = h[valid] * pixel_area * w
                area = pixel_area * (p[valid] if weight == "probability" else 1.0)

                ix = np.floor(np.asarray(lon) / GRID_DEG).astype(np.int64)
                iy = np.floor(np.asarray(lat) / GRID_DEG).astype(np.int64)
                key = ix * 1_000_000 + (iy + LAT_OFFSET)

                uk, inv = np.unique(key, return_inverse=True)
                vsum = np.bincount(inv, weights=vol, minlength=len(uk))
                asum = np.bincount(inv, weights=np.broadcast_to(area, vol.shape),
                                   minlength=len(uk))

                for k, v, a in zip(uk, vsum, asum):
                    prev = sums.get(k)
                    if prev is None:
                        sums[k] = [v, a]
                    else:
                        prev[0] += v
                        prev[1] += a

                n_blocks += 1

        if verbose:
            print(f"    {n_blocks} blocks with buildings, {len(sums):,} cells hit")

    return sums


def sums_to_frame(sums, year):
    if not sums:
        return pd.DataFrame(columns=["lon", "lat", "year", "volume_m3",
                                     "built_area_m2"])
    keys = np.fromiter(sums.keys(), dtype=np.int64)
    vals = np.array(list(sums.values()), dtype="float64")

    ix = keys // 1_000_000
    iy = keys % 1_000_000 - LAT_OFFSET

    return pd.DataFrame({
        "lon": ((ix + 0.5) * GRID_DEG).round(4),
        "lat": ((iy + 0.5) * GRID_DEG).round(4),
        "year": year,
        "volume_m3": vals[:, 0],
        "built_area_m2": vals[:, 1],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(folder, presence_threshold, weight, dpi):
    csv_path = os.path.join(folder, CSV_NAME)
    tifs = sorted(glob.glob(os.path.join(folder, TIF_GLOB)))
    if not tifs:
        raise SystemExit(f"No rasters matching {TIF_GLOB} in {folder}")

    cells = pd.read_csv(csv_path)
    cells["lon"] = cells["lon"].round(4)
    cells["lat"] = cells["lat"].round(4)
    print(f"{len(cells):,} treatment map cells")

    frames = []
    for path in tifs:
        m = re.search(r"(\d{4})", os.path.basename(path))
        if not m:
            print(f"  skipping {path}, no year in filename")
            continue
        year = int(m.group(1))
        print(f"Aggregating {year}")
        frames.append(sums_to_frame(
            aggregate_raster(path, presence_threshold, weight), year))

    vol = pd.concat(frames, ignore_index=True)

    # Keep only cells that are in the treatment map. Everything outside the
    # study footprint is dropped, as are cells with no eligible households,
    # which were already excluded when fig-map.csv was built.
    panel = cells[["lon", "lat", "treatment_intensity"]].merge(
        vol, on=["lon", "lat"], how="left")
    panel[["volume_m3", "built_area_m2"]] = panel[
        ["volume_m3", "built_area_m2"]].fillna(0.0)

    years = sorted(panel["year"].dropna().unique().astype(int))

    matched = panel["volume_m3"].gt(0).groupby(panel["year"]).sum()
    print("\nCells with any built volume, by year:")
    print(matched.to_string())

    panel.to_csv(os.path.join(folder, "siaya_volume_by_cell.csv"), index=False)

    # --- summary by bin -----------------------------------------------------
    summary = (panel
               .groupby(["treatment_intensity", "year"], as_index=False)
               .agg(mean_volume_m3=("volume_m3", "mean"),
                    total_volume_m3=("volume_m3", "sum"),
                    mean_built_area_m2=("built_area_m2", "mean"),
                    n_cells=("volume_m3", "size")))
    summary.to_csv(os.path.join(folder, "siaya_volume_by_bin.csv"), index=False)
    print("\nMean built volume per cell (m3):")
    print(summary.pivot(index="year", columns="treatment_intensity",
                        values="mean_volume_m3").round(0).to_string())

    # --- figure -------------------------------------------------------------
    present = [b for b in BINS if b in set(summary["treatment_intensity"])]
    cmap = matplotlib.colormaps["viridis"]
    colors = {b: cmap(i / max(len(present) - 1, 1)) for i, b in enumerate(present)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for b in present:
        s = summary[summary["treatment_intensity"] == b].sort_values("year")
        axes[0].plot(s["year"], s["mean_volume_m3"], marker="o",
                     color=colors[b], label=BIN_LABELS.get(b, b))

        base = s["mean_volume_m3"].iloc[0]
        idx = 100 * s["mean_volume_m3"] / base if base else s["mean_volume_m3"] * 0
        axes[1].plot(s["year"], idx, marker="o", color=colors[b],
                     label=BIN_LABELS.get(b, b))

    axes[0].set_ylabel("Mean built volume per cell (m3)")
    axes[0].set_title("Levels")
    axes[1].axhline(100, color="#999999", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel(f"Index, {years[0]} = 100")
    axes[1].set_title("Indexed to first year")

    for ax in axes:
        ax.set_xlabel("Year")
        ax.set_xticks(years)
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[1].legend(title="Recipient households per cell", fontsize=9,
                   title_fontsize=9)
    fig.suptitle("Built volume by GiveDirectly treatment intensity, Siaya County",
                 fontsize=13)
    fig.tight_layout()

    out_png = os.path.join(folder, "siaya_volume_by_treatment.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", default=FOLDER)
    p.add_argument("--presence-threshold", type=float, default=0.5,
                   help="minimum building_presence for a pixel to count")
    p.add_argument("--weight", choices=["binary", "probability"],
                   default="binary",
                   help="count qualifying pixels fully, or weight by presence")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    main(a.folder, a.presence_threshold, a.weight, a.dpi)
