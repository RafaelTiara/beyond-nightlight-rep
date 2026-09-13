# -*- coding: utf-8 -*-
"""
Built footprint area over time by GiveDirectly treatment intensity, Siaya.

Standalone companion to volume_by_treatment_v3.py. Same structure, same
spatial block bootstrap, but the outcome is surface area in m2 rather than
volume in m3, and it writes its own outputs so the two never overwrite each
other.

Area is the closer analogue to the building footprint outcome in Huang,
Hsiang and Gonzalez-Navarro, whose headline estimate is 7.9 m2 of footprint
per USD 1,000 transfer.

Only the building_presence band is read. Every 4 m pixel clearing the
threshold contributes its full pixel area, so the measure is quantised in
16 m2 steps and will round small structures up.

Takes the Open Buildings 2.5D Temporal exports (one GeoTIFF per year, 4 m
pixels, EPSG:32636) and the beyond-nightlight treatment map (fig-map.csv,
0.005 degree cells), aggregates footprint area to that grid, joins on
treatment intensity bin, and plots the mean per cell over time: one line per
bin, in levels and as level differences against the lowest intensity bin.

Note on the differences panel. 2016 is not a pre period, since transfers ran
from mid 2014 to early 2017, so the base year is itself partly treated. That
is why this plots level differences rather than an index: dividing by a
post treatment outcome can flip the sign of a real effect.

Outputs, written next to the inputs:
  siaya_area_by_cell.csv        cell x year panel
  siaya_area_by_bin.csv         bin x year summary with interval bounds
  siaya_area_by_treatment.png   the figure

Requires: rasterio, pyproj, pandas, numpy, matplotlib.
    pip install rasterio pandas matplotlib

Usage:
    python area_by_treatment.py
    python area_by_treatment.py --presence-threshold 0.6
    python area_by_treatment.py --ci iid --n-boot 2000
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


def presence_band(src):
    """Find the presence band by description, fall back to band 2."""
    desc = [d.lower() if d else "" for d in src.descriptions]
    return next((i + 1 for i, d in enumerate(desc) if "presence" in d), 2)


def aggregate_raster(path, presence_threshold, weight, verbose=True):
    """Sum built footprint area into 0.005 degree cells.

    Returns a dict keyed by packed cell index -> area_m2. Only pixels whose
    building_presence clears the threshold are counted, so empty land
    contributes nothing rather than contributing noise.
    """
    sums = {}

    with rasterio.open(path) as src:
        p_band = presence_band(src)
        pixel_area = abs(src.transform.a * src.transform.e)  # m2, CRS is metric
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        if verbose:
            print(f"  {os.path.basename(path)}: {src.width} x {src.height} px, "
                  f"{pixel_area:.1f} m2/px, presence band {p_band}")

        n_blocks = 0
        for row0 in range(0, src.height, BLOCK):
            for col0 in range(0, src.width, BLOCK):
                win = Window(col0, row0,
                             min(BLOCK, src.width - col0),
                             min(BLOCK, src.height - row0))

                p = src.read(p_band, window=win).astype("float64")

                valid = ((p >= presence_threshold) & (p <= 1.0)
                         & (p != NODATA_SENTINEL) & np.isfinite(p))
                if not valid.any():
                    continue

                rr, cc = np.nonzero(valid)
                # Pixel centres in the raster CRS.
                xs, ys = rasterio.transform.xy(
                    src.transform, rr + row0, cc + col0, offset="center")
                lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))

                # binary: a qualifying pixel counts in full.
                # probability: it counts in proportion to how confident the
                # model is that a building is there.
                area = pixel_area * (p[valid] if weight == "probability"
                                     else np.ones(valid.sum()))

                ix = np.floor(np.asarray(lon) / GRID_DEG).astype(np.int64)
                iy = np.floor(np.asarray(lat) / GRID_DEG).astype(np.int64)
                key = ix * 1_000_000 + (iy + LAT_OFFSET)

                uk, inv = np.unique(key, return_inverse=True)
                asum = np.bincount(inv, weights=area, minlength=len(uk))

                for k, a in zip(uk, asum):
                    sums[k] = sums.get(k, 0.0) + a

                n_blocks += 1

        if verbose:
            print(f"    {n_blocks} blocks with buildings, {len(sums):,} cells hit")

    return sums


def sums_to_frame(sums, year):
    if not sums:
        return pd.DataFrame(columns=["lon", "lat", "year", "area_m2"])
    keys = np.fromiter(sums.keys(), dtype=np.int64)
    vals = np.fromiter(sums.values(), dtype="float64")

    ix = keys // 1_000_000
    iy = keys % 1_000_000 - LAT_OFFSET

    return pd.DataFrame({
        "lon": ((ix + 0.5) * GRID_DEG).round(4),
        "lat": ((iy + 0.5) * GRID_DEG).round(4),
        "year": year,
        "area_m2": vals,
    })


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def spatial_block(lon, lat, block_deg):
    """Coarse spatial block id, for clustering neighbouring cells together."""
    bx = np.floor(np.asarray(lon) / block_deg).astype(np.int64)
    by = np.floor(np.asarray(lat) / block_deg).astype(np.int64)
    return bx * 1_000_000 + (by + LAT_OFFSET)


def compute_cis(panel, present, years, ref, method, n_boot, block_deg,
                alpha, seed):
    """Return {(bin, year): (lo, hi)} for levels and for differences vs ref.

    method='iid'   analytic, treats cells as independent. Understates the
                   width, since neighbouring cells are strongly correlated.
    method='block' bootstrap resampling coarse spatial blocks, which keeps
                   the within-block correlation intact. This is the honest
                   one and is the default.
    """
    keys = [(b, y) for b in present for y in years]
    kidx = {k: i for i, k in enumerate(keys)}

    if method == "iid":
        levels, diffs = {}, {}
        stats = (panel.groupby(["treatment_intensity", "year"])["area_m2"]
                 .agg(["mean", "std", "size"]))
        z = 1.959963985
        for b, y in keys:
            if (b, y) not in stats.index:
                continue
            m, sd, n = stats.loc[(b, y)]
            se = sd / np.sqrt(n) if n > 1 else 0.0
            levels[(b, y)] = (m - z * se, m + z * se)

            if (ref, y) in stats.index and b != ref:
                mr, sdr, nr = stats.loc[(ref, y)]
                ser = sdr / np.sqrt(nr) if nr > 1 else 0.0
                sed = np.sqrt(se ** 2 + ser ** 2)
                d = m - mr
                diffs[(b, y)] = (d - z * sed, d + z * sed)
        return levels, diffs

    # --- block bootstrap ----------------------------------------------------
    work = panel.copy()
    work["block"] = spatial_block(work["lon"], work["lat"], block_deg)

    blocks = np.sort(work["block"].unique())
    bidx = {b: i for i, b in enumerate(blocks)}

    S = np.zeros((len(blocks), len(keys)))
    N = np.zeros((len(blocks), len(keys)))
    grouped = work.groupby(["block", "treatment_intensity", "year"])["area_m2"]
    for (bl, tb, yr), sub in grouped:
        k = (tb, int(yr))
        if k in kidx:
            S[bidx[bl], kidx[k]] = sub.sum()
            N[bidx[bl], kidx[k]] = sub.size

    print(f"  block bootstrap: {len(blocks)} blocks of {block_deg} degrees, "
          f"{n_boot} draws")

    rng = np.random.default_rng(seed)
    means = np.full((n_boot, len(keys)), np.nan)
    for i in range(n_boot):
        draw = rng.integers(0, len(blocks), len(blocks))
        s = S[draw].sum(axis=0)
        n = N[draw].sum(axis=0)
        means[i] = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    levels, diffs = {}, {}
    for (b, y) in keys:
        col = means[:, kidx[(b, y)]]
        if np.all(np.isnan(col)):
            continue
        levels[(b, y)] = (np.nanpercentile(col, lo_q),
                          np.nanpercentile(col, hi_q))
        if b != ref and (ref, y) in kidx:
            d = col - means[:, kidx[(ref, y)]]
            diffs[(b, y)] = (np.nanpercentile(d, lo_q),
                             np.nanpercentile(d, hi_q))
    return levels, diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(folder, presence_threshold, weight, dpi, reference,
         ci_method, n_boot, block_deg, alpha, seed):
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

    areas = pd.concat(frames, ignore_index=True)

    # Keep only cells that are in the treatment map. Everything outside the
    # study footprint is dropped, as are cells with no eligible households,
    # which were already excluded when fig-map.csv was built.
    panel = cells[["lon", "lat", "treatment_intensity"]].merge(
        areas, on=["lon", "lat"], how="left")
    panel["area_m2"] = panel["area_m2"].fillna(0.0)

    years = sorted(panel["year"].dropna().unique().astype(int))

    matched = panel["area_m2"].gt(0).groupby(panel["year"]).sum()
    print("\nCells with any built area, by year:")
    print(matched.to_string())

    panel.to_csv(os.path.join(folder, "siaya_area_by_cell.csv"), index=False)

    # --- summary by bin -----------------------------------------------------
    summary = (panel
               .groupby(["treatment_intensity", "year"], as_index=False)
               .agg(mean_area_m2=("area_m2", "mean"),
                    sd_area_m2=("area_m2", "std"),
                    total_area_m2=("area_m2", "sum"),
                    n_cells=("area_m2", "size")))
    print("\nMean built footprint area per cell (m2):")
    print(summary.pivot(index="year", columns="treatment_intensity",
                        values="mean_area_m2").round(0).to_string())

    # --- figure -------------------------------------------------------------
    present = [b for b in BINS if b in set(summary["treatment_intensity"])]
    cmap = matplotlib.colormaps["viridis"]
    colors = {b: cmap(i / max(len(present) - 1, 1)) for i, b in enumerate(present)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Reference bin for the difference panel: the lowest intensity bin present.
    ref = reference if reference in present else present[0]
    ref_series = (summary[summary["treatment_intensity"] == ref]
                  .set_index("year")["mean_area_m2"])

    print(f"\nConfidence intervals: {ci_method}")
    levels_ci, diffs_ci = compute_cis(panel, present, years, ref, ci_method,
                                      n_boot, block_deg, alpha, seed)

    # Attach the interval bounds to the summary csv so the numbers are usable
    # outside this figure.
    summary["ci_lo"] = [levels_ci.get((b, int(y)), (np.nan, np.nan))[0]
                        for b, y in zip(summary["treatment_intensity"],
                                        summary["year"])]
    summary["ci_hi"] = [levels_ci.get((b, int(y)), (np.nan, np.nan))[1]
                        for b, y in zip(summary["treatment_intensity"],
                                        summary["year"])]
    summary["diff_vs_ref"] = [
        m - ref_series.get(int(y), np.nan)
        for m, y in zip(summary["mean_area_m2"], summary["year"])]
    summary["diff_ci_lo"] = [diffs_ci.get((b, int(y)), (np.nan, np.nan))[0]
                             for b, y in zip(summary["treatment_intensity"],
                                             summary["year"])]
    summary["diff_ci_hi"] = [diffs_ci.get((b, int(y)), (np.nan, np.nan))[1]
                             for b, y in zip(summary["treatment_intensity"],
                                             summary["year"])]
    summary.to_csv(os.path.join(folder, "siaya_area_by_bin.csv"), index=False)

    # Small horizontal offsets so overlapping error bars stay readable.
    offsets = np.linspace(-0.12, 0.12, len(present)) if len(present) > 1 else [0]

    for b, dx in zip(present, offsets):
        sb = summary[summary["treatment_intensity"] == b].sort_values("year")
        yr = sb["year"].values.astype(float)

        m = sb["mean_area_m2"].values
        lo = np.array([levels_ci.get((b, int(y)), (np.nan, np.nan))[0] for y in yr])
        hi = np.array([levels_ci.get((b, int(y)), (np.nan, np.nan))[1] for y in yr])
        axes[0].errorbar(yr + dx, m,
                         yerr=[np.nan_to_num(m - lo), np.nan_to_num(hi - m)],
                         marker="o", capsize=3, linewidth=1.6, elinewidth=1.1,
                         color=colors[b], label=BIN_LABELS.get(b, b))

        # Difference in levels against the reference bin. No division by an
        # outcome, so a positive effect shows up as a widening gap rather than
        # being absorbed into a denominator.
        d = m - ref_series.reindex(sb["year"]).values
        if b == ref:
            axes[1].axhline(0, color="#999999", linewidth=0.8, linestyle="--")
            continue
        dlo = np.array([diffs_ci.get((b, int(y)), (np.nan, np.nan))[0] for y in yr])
        dhi = np.array([diffs_ci.get((b, int(y)), (np.nan, np.nan))[1] for y in yr])
        axes[1].errorbar(yr + dx, d,
                         yerr=[np.nan_to_num(d - dlo), np.nan_to_num(dhi - d)],
                         marker="o", capsize=3, linewidth=1.6, elinewidth=1.1,
                         color=colors[b], label=BIN_LABELS.get(b, b))

    axes[0].set_ylabel("Mean built footprint area per cell (m2)")
    axes[0].set_title(f"Levels, {int((1 - alpha) * 100)}% CI")
    axes[1].set_ylabel(f"Difference vs. {BIN_LABELS.get(ref, ref)} (m2)")
    axes[1].set_title(f"Level difference against {BIN_LABELS.get(ref, ref)}, "
                      f"{int((1 - alpha) * 100)}% CI")

    for ax in axes:
        ax.set_xlabel("Year")
        ax.set_xticks(years)
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[1].legend(title="Recipient households per cell", fontsize=9,
                   title_fontsize=9)
    fig.suptitle("Built footprint area by GiveDirectly treatment intensity, "
                 "Siaya County", fontsize=13)
    fig.tight_layout()

    out_png = os.path.join(folder, "siaya_area_by_treatment.png")
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
    p.add_argument("--reference", default="0-3",
                   help="bin used as the comparison group in the right panel")
    p.add_argument("--ci", choices=["block", "iid"], default="block",
                   help="spatial block bootstrap, or naive iid intervals")
    p.add_argument("--n-boot", type=int, default=500, dest="n_boot")
    p.add_argument("--block-deg", type=float, default=0.05, dest="block_deg",
                   help="size of the bootstrap blocks in degrees")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    main(a.folder, a.presence_threshold, a.weight, a.dpi, a.reference,
         a.ci, a.n_boot, a.block_deg, a.alpha, a.seed)
