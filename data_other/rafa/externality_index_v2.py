# -*- coding: utf-8 -*-
"""
Spillover index from neighbouring cells, GiveDirectly study area, Siaya.

Each 0.005 degree cell has up to 8 neighbours: 4 orthogonal and 4 diagonal.
For every cell we build an externality index as the sum over those neighbours
of the midpoint of the neighbour's treatment intensity bin:

    0-3   ->  1.5
    4-6   ->  5
    7-9   ->  8
    10-12 -> 11
    13-   -> 20      (open ended, so a stand in value rather than a midpoint)

so the index is roughly the number of recipient households living in the ring
around a cell. We then restrict to cells that are themselves in the lowest
intensity bin and scatter their built volume against the index. The idea is
that own treatment is held roughly fixed while exposure to nearby cash varies,
which is the spillover margin Egger et al. estimate with distance rings.

v2: the outcome is in logs, so the slope is a semi elasticity, meaning percent
change in built volume per extra recipient household in the surrounding ring.
Confidence intervals come from a spatial block bootstrap and are shown three
ways: a shaded band around the fitted line, error bars on the decile means,
and the slope interval printed in each panel title.

Inputs, both expected in the folder:
    fig-map.csv                treatment intensity bins per cell
    siaya_volume_by_cell.csv   produced by volume_by_treatment_v3.py

Outputs:
    siaya_externality_index.csv   per cell index, neighbour counts, volume
    siaya_externality_scatter.png the figure

Usage:
    python externality_index_v2.py
    python externality_index_v2.py --own-bin 0-3 --min-neighbours 8
    python externality_index_v2.py --log-zeros log1p --n-boot 2000
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLDER = r"G:\Mi unidad\RESEARCH\beyond-nightlight"
CSV_NAME = "fig-map.csv"
PANEL_NAME = "siaya_volume_by_cell.csv"

GRID_DEG = 0.005

# Midpoint of each bin's range. The top bin is open ended, so 20 is a
# stand in rather than a midpoint; change it here to test sensitivity.
BIN_VALUE = {
    "0-3": 1.5,
    "4-6": 5.0,
    "7-9": 8.0,
    "10-12": 11.0,
    "13-": 20.0,
}

# The 8 neighbours: 4 orthogonal, 4 diagonal.
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1), (1, 0), (1, 1)]


def cell_index(lon, lat):
    """Integer grid indices from cell centroids."""
    ix = np.floor(np.asarray(lon) / GRID_DEG + 1e-9).astype(np.int64)
    iy = np.floor(np.asarray(lat) / GRID_DEG + 1e-9).astype(np.int64)
    return ix, iy


def build_index(cells):
    """Externality index and neighbour count for every cell in the map."""
    ix, iy = cell_index(cells["lon"], cells["lat"])
    cells = cells.assign(ix=ix, iy=iy)
    cells["own_value"] = cells["treatment_intensity"].map(BIN_VALUE)

    if cells["own_value"].isna().any():
        unknown = sorted(set(cells.loc[cells["own_value"].isna(),
                                       "treatment_intensity"]))
        raise ValueError(f"Bins with no assigned value: {unknown}")

    lookup = dict(zip(zip(cells["ix"], cells["iy"]), cells["own_value"]))

    ext = np.zeros(len(cells))
    n_present = np.zeros(len(cells), dtype=int)
    for k, (a, b) in enumerate(zip(cells["ix"].values, cells["iy"].values)):
        total, n = 0.0, 0
        for da, db in NEIGHBOURS:
            v = lookup.get((a + da, b + db))
            if v is not None:
                total += v
                n += 1
        ext[k] = total
        n_present[k] = n

    cells["externality_index"] = ext
    cells["n_neighbours"] = n_present
    # Mean over the neighbours that exist, for cells on the edge of the map.
    cells["externality_mean"] = np.where(n_present > 0, ext / np.maximum(n_present, 1),
                                         np.nan)
    return cells


def block_bootstrap(x, y, block, decile, n_deciles, x_grid, n_boot, alpha, seed):
    """Block bootstrap for the OLS slope, the fitted line, and decile means.

    Resamples coarse spatial blocks rather than individual cells, since
    neighbouring cells are strongly correlated. Decile edges are held fixed at
    their full sample values so the bins mean the same thing in every draw.
    Returns (slope_lo, slope_hi, band_lo, band_hi, dec_lo, dec_hi).
    """
    blocks = np.unique(block)
    idx_by_block = {b: np.flatnonzero(block == b) for b in blocks}
    rng = np.random.default_rng(seed)

    slopes = np.full(n_boot, np.nan)
    bands = np.full((n_boot, len(x_grid)), np.nan)
    decs = np.full((n_boot, n_deciles), np.nan)

    for i in range(n_boot):
        draw = rng.integers(0, len(blocks), len(blocks))
        rows = np.concatenate([idx_by_block[blocks[d]] for d in draw])
        xb, yb, db = x[rows], y[rows], decile[rows]

        if xb.std() > 0:
            slope, intercept = np.polyfit(xb, yb, 1)
            slopes[i] = slope
            bands[i] = intercept + slope * x_grid

        for k in range(n_deciles):
            m = db == k
            if m.any():
                decs[i, k] = yb[m].mean()

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (np.nanpercentile(slopes, lo_q), np.nanpercentile(slopes, hi_q),
            np.nanpercentile(bands, lo_q, axis=0),
            np.nanpercentile(bands, hi_q, axis=0),
            np.nanpercentile(decs, lo_q, axis=0),
            np.nanpercentile(decs, hi_q, axis=0))


def main(folder, own_bin, min_neighbours, use_mean, block_deg, n_boot, alpha,
         seed, dpi, log_zeros):
    cells = pd.read_csv(os.path.join(folder, CSV_NAME))
    cells["lon"] = cells["lon"].round(4)
    cells["lat"] = cells["lat"].round(4)

    panel_path = os.path.join(folder, PANEL_NAME)
    if not os.path.exists(panel_path):
        raise SystemExit(f"{PANEL_NAME} not found. Run volume_by_treatment_v3.py "
                         "first, it writes the per cell volume panel.")
    panel = pd.read_csv(panel_path)

    cells = build_index(cells)
    print(f"{len(cells):,} cells, "
          f"{(cells['n_neighbours'] == 8).sum():,} with a full ring of 8")

    df = panel.merge(
        cells[["lon", "lat", "externality_index", "externality_mean",
               "n_neighbours", "own_value"]],
        on=["lon", "lat"], how="left")

    df.to_csv(os.path.join(folder, "siaya_externality_index.csv"), index=False)

    sub = df[(df["treatment_intensity"] == own_bin)
             & (df["n_neighbours"] >= min_neighbours)].copy()
    xcol = "externality_mean" if use_mean else "externality_index"
    sub = sub[np.isfinite(sub[xcol])]

    years = sorted(sub["year"].dropna().unique().astype(int))
    print(f"{len(sub) // max(len(years), 1):,} cells in bin {own_bin} "
          f"with at least {min_neighbours} neighbours")

    # Spatial blocks for the bootstrap, same construction as the volume script.
    bx = np.floor(sub["lon"] / block_deg).astype(np.int64)
    by = np.floor(sub["lat"] / block_deg).astype(np.int64)
    sub["block"] = bx * 1_000_000 + (by + 100_000)

    fig, axes = plt.subplots(1, len(years), figsize=(5 * len(years), 4.8),
                             sharey=True, squeeze=False)
    axes = axes[0]

    n_dec = 10
    for ax, yr in zip(axes, years):
        s = sub[sub["year"] == yr].copy()

        # Log transform. Cells with zero volume have no log, so either drop
        # them or use log1p, which is fine here because volumes are large
        # enough that log1p and log agree everywhere except at zero.
        if log_zeros == "drop":
            n0 = (s["volume_m3"] <= 0).sum()
            s = s[s["volume_m3"] > 0]
            if n0:
                print(f"  {yr}: dropped {n0:,} cells with zero volume")
            y = np.log(s["volume_m3"].values.astype(float))
        else:
            y = np.log1p(s["volume_m3"].values.astype(float))

        x = s[xcol].values.astype(float)
        if len(x) < 3 or x.std() == 0:
            ax.set_title(str(yr))
            continue

        ax.scatter(x, y, s=12, alpha=0.3, linewidths=0, color="#3b6ea5")

        # Fixed decile edges, so the bootstrap bins mean the same thing in
        # every draw.
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_dec + 1)))
        decile = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        k_dec = len(edges) - 1

        dec_x = np.array([x[decile == k].mean() if (decile == k).any() else np.nan
                          for k in range(k_dec)])
        dec_y = np.array([y[decile == k].mean() if (decile == k).any() else np.nan
                          for k in range(k_dec)])

        slope, intercept = np.polyfit(x, y, 1)
        x_grid = np.linspace(x.min(), x.max(), 60)

        (s_lo, s_hi, b_lo, b_hi, d_lo, d_hi) = block_bootstrap(
            x, y, s["block"].values, decile, k_dec, x_grid, n_boot, alpha, seed)

        ax.fill_between(x_grid, b_lo, b_hi, color="black", alpha=0.12,
                        linewidth=0)
        ax.plot(x_grid, intercept + slope * x_grid, color="black",
                linewidth=1.4, linestyle="--", label="OLS")
        ax.errorbar(dec_x, dec_y, yerr=[dec_y - d_lo, d_hi - dec_y],
                    marker="o", color="#b03030", linewidth=1.6, markersize=5,
                    capsize=3, elinewidth=1.0, label="decile means")

        # A slope on a log outcome is a semi elasticity: percent change in
        # volume per extra unit of the index, i.e. per recipient household
        # in the surrounding ring.
        ax.set_title(f"{yr}\nslope {100 * slope:+.2f}%  "
                     f"[{100 * s_lo:+.2f}, {100 * s_hi:+.2f}] per unit")
        print(f"  {yr}: {100 * slope:+.3f}% per unit, "
              f"{int((1 - alpha) * 100)}% CI [{100 * s_lo:+.3f}, "
              f"{100 * s_hi:+.3f}], n={len(x)}")

        ax.set_xlabel("Externality index"
                      + (" (mean per neighbour)" if use_mean else ""))
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("log built volume per cell")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Log built volume vs. neighbouring treatment, cells in bin "
                 f"{own_bin}, Siaya County", fontsize=13)
    fig.tight_layout()

    out = os.path.join(folder, "siaya_externality_scatter.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", default=FOLDER)
    p.add_argument("--own-bin", default="0-3", dest="own_bin",
                   help="restrict to cells in this treatment intensity bin")
    p.add_argument("--min-neighbours", type=int, default=0,
                   dest="min_neighbours",
                   help="drop cells with fewer than this many neighbours in the map")
    p.add_argument("--use-mean", action="store_true", dest="use_mean",
                   help="index as the mean over existing neighbours, not the sum")
    p.add_argument("--block-deg", type=float, default=0.05, dest="block_deg")
    p.add_argument("--n-boot", type=int, default=500, dest="n_boot")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-zeros", choices=["drop", "log1p"],
                   default="drop", dest="log_zeros",
                   help="how to handle cells with zero volume under the log")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    main(a.folder, a.own_bin, a.min_neighbours, a.use_mean, a.block_deg,
         a.n_boot, a.alpha, a.seed, a.dpi, a.log_zeros)
