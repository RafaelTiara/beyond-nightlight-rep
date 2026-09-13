# -*- coding: utf-8 -*-
"""
Plots the GiveDirectly study area in Siaya County, Kenya.

Reads fig-map.csv from the beyond-nightlight replication data and draws:
  - the grid cells, coloured by treatment intensity (number of recipient
    households per 0.005 degree cell),
  - the towns in and near the study area,
  - a locator inset showing where this sits in western Kenya.

Outputs a PNG next to the csv.

Requires: pandas, matplotlib. Optionally contextily for a real basemap
(pip install contextily), which is used automatically if present.

Usage:
    python plot_siaya_map.py
    python plot_siaya_map.py --csv path/to/fig-map.csv --out path/to/map.png
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CSV = r"G:\Mi unidad\RESEARCH\beyond-nightlight\fig-map.csv"
DEFAULT_OUT = r"G:\Mi unidad\RESEARCH\beyond-nightlight\siaya_treatment_map.png"

# Treatment intensity bins as they appear in fig-map.csv, in order.
BINS = ["0-3", "4-6", "7-9", "10-12", "13-"]
BIN_LABELS = {
    "0-3": "0 to 3",
    "4-6": "4 to 6",
    "7-9": "7 to 9",
    "10-12": "10 to 12",
    "13-": "13 or more",
}

# Towns in and around the study area. Coordinates are approximate, to within
# a kilometre or so; adjust if you need them precise.
# in_study_area flags the ones inside the GiveDirectly footprint (Alego
# Usonga, Ugunja, Ugenya) as opposed to regional reference points.
TOWNS = [
    {"name": "Siaya", "lon": 34.2881, "lat": 0.0607, "in_study_area": True},
    {"name": "Ugunja", "lon": 34.3100, "lat": 0.1870, "in_study_area": True},
    {"name": "Ukwala", "lon": 34.1833, "lat": 0.2167, "in_study_area": True},
    {"name": "Yala", "lon": 34.5375, "lat": 0.0925, "in_study_area": True},
    {"name": "Sega", "lon": 34.2000, "lat": 0.1500, "in_study_area": True},
    {"name": "Bondo", "lon": 34.2667, "lat": -0.2417, "in_study_area": False},
    {"name": "Kisumu", "lon": 34.7680, "lat": -0.0917, "in_study_area": False},
    {"name": "Busia", "lon": 34.1117, "lat": 0.4608, "in_study_area": False},
]

# Rough extent of Siaya County, used for the locator inset.
SIAYA_BBOX = (33.90, -0.45, 34.65, 0.40)  # west, south, east, north

GRID_DEG = 0.005  # resolution of fig-map.csv


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def load(csv_path):
    df = pd.read_csv(csv_path)
    missing = {"lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    return df


def add_basemap(ax, crs="EPSG:4326"):
    """Add an OSM basemap if contextily is installed. Silently skip if not."""
    try:
        import contextily as cx
    except ImportError:
        return False
    try:
        cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron,
                       attribution_size=6)
        return True
    except Exception as exc:  # network off, tile server down, etc.
        print(f"Basemap skipped: {exc}")
        return False


def main(csv_path, out_path, dpi):
    df = load(csv_path)

    west, east = df["lon"].min(), df["lon"].max()
    south, north = df["lat"].min(), df["lat"].max()
    pad = 0.03

    print(f"{len(df):,} grid cells")
    print(f"Extent: lon {west:.4f} to {east:.4f}, lat {south:.4f} to {north:.4f}")

    fig, ax = plt.subplots(figsize=(11, 9))

    # --- treatment intensity cells -----------------------------------------
    has_intensity = "treatment_intensity" in df.columns
    if has_intensity:
        cmap = matplotlib.colormaps["Greens"]
        present = [b for b in BINS if b in set(df["treatment_intensity"])]
        colors = {b: cmap(0.2 + 0.8 * i / max(len(present) - 1, 1))
                  for i, b in enumerate(present)}

        # Marker size in points^2 so cells roughly tile the map.
        width_pts = (GRID_DEG / (east - west + 2 * pad)) * fig.get_size_inches()[0] * 72
        size = max(width_pts ** 2, 2)

        for b in present:
            sub = df[df["treatment_intensity"] == b]
            ax.scatter(sub["lon"], sub["lat"], s=size, marker="s",
                       color=colors[b], linewidths=0,
                       label=BIN_LABELS.get(b, b))

        handles = [Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=colors[b], markeredgecolor="none",
                          label=BIN_LABELS.get(b, b)) for b in present]
        legend = ax.legend(handles=handles, title="Recipient households\nper cell",
                           loc="upper left", frameon=True, framealpha=0.9,
                           fontsize=9, title_fontsize=9)
        ax.add_artist(legend)
    else:
        ax.scatter(df["lon"], df["lat"], s=3, color="#2b6a3f", linewidths=0)

    # --- towns --------------------------------------------------------------
    for t in TOWNS:
        inside = (west - pad <= t["lon"] <= east + pad
                  and south - pad <= t["lat"] <= north + pad)
        if not inside:
            continue
        ax.plot(t["lon"], t["lat"], marker="*", markersize=15,
                color="#b03030", markeredgecolor="white", markeredgewidth=0.8,
                zorder=5)
        ax.annotate(t["name"], (t["lon"], t["lat"]),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="#7a1f1f", zorder=6,
                    path_effects=None)

    # --- frame --------------------------------------------------------------
    ax.set_xlim(west - pad, east + pad)
    ax.set_ylim(south - pad, north + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GiveDirectly study area, Siaya County, Kenya\n"
                 "Treatment intensity at 0.005 degrees", fontsize=13)
    ax.grid(alpha=0.15, linewidth=0.5)

    # Equator reference, since the study area straddles it.
    if south - pad < 0 < north + pad:
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
        ax.annotate("equator", (west - pad, 0), xytext=(4, 3),
                    textcoords="offset points", fontsize=8, color="#666666")

    # Approximate scale bar. 1 degree of longitude is about 111.3 km at the
    # equator, which is where we are.
    km = 10
    deg = km / 111.3
    x0 = east + pad - deg - 0.01
    y0 = south - pad + 0.012
    ax.plot([x0, x0 + deg], [y0, y0], color="black", linewidth=2.5, zorder=6)
    ax.annotate(f"{km} km", ((x0 + x0 + deg) / 2, y0), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, zorder=6)

    add_basemap(ax)

    # --- locator inset ------------------------------------------------------
    inset = fig.add_axes([0.68, 0.66, 0.24, 0.24])
    w, s, e, n = SIAYA_BBOX
    inset.add_patch(Rectangle((w, s), e - w, n - s, facecolor="#e8e8e8",
                              edgecolor="#999999", linewidth=0.8))
    inset.add_patch(Rectangle((west, south), east - west, north - south,
                              facecolor="none", edgecolor="#b03030",
                              linewidth=1.6))
    for t in TOWNS:
        if w <= t["lon"] <= e and s <= t["lat"] <= n:
            inset.plot(t["lon"], t["lat"], marker=".", markersize=4,
                       color="#555555")
    inset.set_xlim(w - 0.05, e + 0.05)
    inset.set_ylim(s - 0.05, n + 0.05)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Siaya County", fontsize=9)
    add_basemap(inset)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    main(args.csv, args.out, args.dpi)
