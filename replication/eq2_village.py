"""
Equation 2 of Egger, Haushofer, Miguel, Niehaus and Walker (2022), estimated by IV at the
village level, with Open Buildings 2.5D building stock as the outcome.

Second stage:

    y_v = alpha + beta Amt_v + sum_r beta_r Amt_{-v,r} + delta y_{v,t=0} + eps_v

First stage, one per endogenous regressor, on the full instrument set:

    Amt_v      = pi_0  + pi_1  Treat_v + sum_r pi_{1r}  s_{v,r} + delta' y_{v,t=0} + u_v
    Amt_{-v,r} = psi_0 + psi_1 Treat_v + sum_s psi_{1s} s_{v,s} + delta'' y_{v,t=0} + w_{v,r}

This version follows the paper more closely than the grid one: the unit is the village, the
instrument for the own amount is the village treatment indicator, the instrument for each
ring is the share of eligible households assigned to treatment in that ring, and the baseline
value of the outcome enters when available.

Data:
  data_other/GDFigure2/VillageTreatmentData.shp        653 GE villages with treat, hi_sat, amounts
  data_other/GDFigure2/GE_GD_villages_ConvexHull.shp   polygons for 744 villages
  GE_HH_Census_2017-07-17_cleanGPS.csv                 households, eligible and recipients by village
  data_other/rafa/siaya_buildings_YYYY.tif             Open Buildings 2.5D, height and presence
                                                       bands, 4 m, EPSG:32636

The treatment shapefile stores the village code rounded to thousands, so it cannot be used as
a key. It is merged to the census by coordinates: each village point matches the census
centroid with a median error of one meter. The polygons do carry the full code and merge to
the census directly.

Footprint: pixels with building_presence at or above 0.5 contribute their full 16 m2, the same
convention as area_by_treatment.py. Volume: height times pixel area over those same pixels.

Cross section: the outcome is the 2016 footprint, the earliest Open Buildings layer and the
only one close to the experiment, which ran from 2014 to 2017. There is no pre program
baseline, so the y_{v,t=0} term is left out, as in the paper's equation when the baseline is
not available.

Conley standard errors, uniform kernel and a 10 km cutoff. The outer limit R is selected by
BIC from 2 km to 20 km in steps of 2 km.

Requirements: pandas, numpy, scipy, geopandas, rasterio.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.spatial import cKDTree

# ============================================================================ SETTINGS

REPO = Path(r"C:\Users\Rafael Tiara\Documents\GitHub\beyond-nightlight-rep")
PROJECT = Path(r"G:\Mi unidad\RESEARCH\beyond-nightlight-replication")

SHP_DIR = REPO / "data_other" / "GDFigure2"
RASTER_DIR = REPO / "data_other" / "rafa"
CENSUS = PROJECT / "GiveDirectlyOriginalPaper" / "GE_HH_Census_2017-07-17_cleanGPS.csv"
OUTDIR = REPO / "output-replication"

OUTCOME_YEAR = 2016       # outcome layer, cross section
BASE_YEAR = OUTCOME_YEAR  # no pre program baseline is available
USE_BASELINE = False      # pure cross section, as in the paper equation
PRESENCE_THRESHOLD = 0.5  # same as area_by_treatment.py

BAND_KM = 2.0
R_GRID = range(2, 22, 2)
R_FIXED = None            # fixed outer limit in km, for example 2; None lets BIC choose
CUTOFF_KM = 10.0
TRANSFER_USD = 1.0        # transfer per recipient household, in thousands of dollars
AMOUNT_SOURCE = "census"  # "census": recipients counted in the census; "shapefile": cum_amou_9
MATCH_TOL_DEG = 0.01      # tolerance of the coordinate merge between shapefile and census

OUTCOMES = ["footprint_area", "building_volume"]
TEX_LABELS = {
    "footprint_area": "Building footprint (m$^2$)",
    "building_volume": "Building volume (m$^3$)",
}
BLOCK = 2048


# ============================================================================ data

def zonal_open_buildings(polygons: gpd.GeoDataFrame, year: int):
    """Footprint and volume summed inside each polygon, from that year's raster."""
    path = RASTER_DIR / f"siaya_buildings_{year}.tif"
    with rasterio.open(path) as src:
        poly = polygons.to_crs(src.crs)
        shapes = [(geom, i + 1) for i, geom in enumerate(poly.geometry)]
        pixel_area = abs(src.transform.a * src.transform.e)
        height_band = next((i + 1 for i, d in enumerate(src.descriptions or [])
                            if d and "height" in d), 1)
        presence_band = next((i + 1 for i, d in enumerate(src.descriptions or [])
                              if d and "presence" in d), 2)

        area = np.zeros(len(poly) + 1)
        volume = np.zeros(len(poly) + 1)
        for r0 in range(0, src.height, BLOCK):
            for c0 in range(0, src.width, BLOCK):
                win = Window(c0, r0, min(BLOCK, src.width - c0), min(BLOCK, src.height - r0))
                labels = rasterize(shapes, out_shape=(int(win.height), int(win.width)),
                                   transform=src.window_transform(win), fill=0, dtype="int32")
                if not labels.any():
                    continue
                p = src.read(presence_band, window=win)
                hgt = src.read(height_band, window=win)
                mask = (p >= PRESENCE_THRESHOLD) & (p <= 1.0) & np.isfinite(p) & (labels > 0)
                if not mask.any():
                    continue
                np.add.at(area, labels[mask], pixel_area)
                np.add.at(volume, labels[mask], hgt[mask] * pixel_area)
    return area[1:], volume[1:]


def build_villages() -> pd.DataFrame:
    treat = gpd.read_file(SHP_DIR / "VillageTreatmentData.shp")
    hulls = gpd.read_file(SHP_DIR / "GE_GD_villages_ConvexHull.shp")
    hulls["village_code"] = hulls["village_co"].round().astype("int64")

    d = pd.read_csv(CENSUS, low_memory=False)
    d["recipient"] = ((d["eligible"] == 1) & (d["treat"] == 1)).astype(int)
    d["elig"] = (d["eligible"] == 1).astype(int)
    cen = (d.groupby("village_code")
             .agg(hh=("raw_id", "size"), e=("elig", "sum"), x=("recipient", "sum"),
                  lat=("avglat", "first"), lon=("avglong", "first"))
             .reset_index().dropna(subset=["lat", "lon"]))

    # the treatment shapefile is merged to the census by coordinates
    tree = cKDTree(np.c_[cen["lon"].values, cen["lat"].values])
    dist, idx = tree.query(np.c_[treat["longitude"].values, treat["latitude"].values])
    treat = treat.assign(village_code=cen["village_code"].values[idx], match_dist=dist)
    bad = treat["match_dist"] > MATCH_TOL_DEG
    if bad.any():
        print(f"  dropped {int(bad.sum())} villages without a coordinate match")
        treat = treat[~bad]
    treat = treat.drop_duplicates(subset="village_code", keep="first")

    df = treat[["village_code", "treat", "hi_sat", "latitude", "longitude", "cum_amou_9"]]
    df = df.merge(cen, on="village_code", how="inner")
    df = df.merge(hulls[["village_code", "geometry"]], on="village_code", how="inner")
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    for year in dict.fromkeys((BASE_YEAR, OUTCOME_YEAR)):
        a, v = zonal_open_buildings(gdf, year)
        gdf[f"area_{year}"] = a
        gdf[f"volume_{year}"] = v

    gdf["footprint_area"] = gdf[f"area_{OUTCOME_YEAR}"]
    gdf["building_volume"] = gdf[f"volume_{OUTCOME_YEAR}"]
    gdf["base_area"] = gdf[f"area_{BASE_YEAR}"]
    gdf["base_volume"] = gdf[f"volume_{BASE_YEAR}"]

    print(f"villages: {len(gdf):,}   treated: {int(gdf['treat'].sum()):,}   "
          f"recipients: {int(gdf['x'].sum()):,}   households: {int(gdf['hh'].sum()):,}")
    print(f"mean footprint {gdf['footprint_area'].mean():,.0f} m2 in {OUTCOME_YEAR}, "
          f"{gdf['base_area'].mean():,.0f} m2 in {BASE_YEAR}")
    return pd.DataFrame(gdf.drop(columns="geometry"))


# ============================================================================ amounts and rings

def km_coords(df: pd.DataFrame) -> np.ndarray:
    lat0 = np.radians(df["latitude"].mean())
    return np.c_[df["longitude"].values * 111.320 * np.cos(lat0),
                 df["latitude"].values * 110.574]


def village_amount(df: pd.DataFrame) -> np.ndarray:
    """Transfers received by each village, in thousands of dollars."""
    if AMOUNT_SOURCE == "shapefile":
        return df["cum_amou_9"].values.astype(float) / 1000.0 * TRANSFER_USD
    return TRANSFER_USD * df["x"].values.astype(float)


def own_terms(df: pd.DataFrame):
    """Own village amount and its instrument, the treatment indicator."""
    return village_amount(df), df["treat"].values.astype(float)


def ring_terms(df: pd.DataFrame, r_max_km: float) -> dict:
    """Amount and instrument in each ring, excluding the village itself."""
    xy = km_coords(df)
    tree = cKDTree(xy)
    x = village_amount(df)
    e = df["e"].values.astype(float)
    et = (df["e"].values * df["treat"].values).astype(float)   # eligible in treated villages
    hh = df["hh"].values.astype(float)

    radii = np.arange(0.0, r_max_km + 1e-9, BAND_KM)
    disc = {0.0: np.c_[x, e, et, hh]}
    for r in radii[1:]:
        idx = tree.query_ball_point(xy, r=r)
        disc[r] = np.array([[x[j].sum(), e[j].sum(), et[j].sum(), hh[j].sum()] for j in idx])

    out = {}
    for r in radii[:-1]:
        tot = disc[r + BAND_KM] - disc[r if r > 0 else 0.0]
        xr, er, etr, hhr = tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3]
        out[int(r)] = {
            "amt": xr,
            "z": np.where(er > 0, etr / np.maximum(er, 1), 0.0),
        }
    return out


def build_matrices(df: pd.DataFrame, r_max_km: float, outcome: str):
    amt0, z0 = own_terms(df)
    rings = ring_terms(df, r_max_km)
    endog_names = ["Amt own village"] + [f"Amt {r} to {r + int(BAND_KM)} km" for r in rings]
    instr_names = ["Village treated"] + [f"Share treated {r} to {r + int(BAND_KM)} km" for r in rings]

    X = [np.ones(len(df)), amt0] + [rings[r]["amt"] for r in rings]
    Z = [np.ones(len(df)), z0] + [rings[r]["z"] for r in rings]
    if USE_BASELINE:
        base = df["base_area"].values if outcome == "footprint_area" else df["base_volume"].values
        X.append(base)
        Z.append(base)
    return np.c_[tuple(X)], np.c_[tuple(Z)], endog_names, instr_names


# ============================================================================ estimation

def conley_weights(df: pd.DataFrame, cutoff_km: float) -> np.ndarray:
    xy = km_coords(df)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    return (d2 <= cutoff_km ** 2).astype(float)


def tsls(y, X, Z, W):
    Pz_X = Z @ (np.linalg.pinv(Z.T @ Z) @ (Z.T @ X))
    A = np.linalg.pinv(Pz_X.T @ X)
    beta = A @ (Pz_X.T @ y)
    u = y - X @ beta
    Xu = Pz_X * u[:, None]
    V = A @ (Xu.T @ W @ Xu) @ A.T
    n, k = X.shape
    V *= n / (n - k)
    bic = n * np.log(float(u @ u) / n) + k * np.log(n)
    return beta, np.sqrt(np.diag(V)), bic


def ols_conley(y, X, W):
    A = np.linalg.pinv(X.T @ X)
    beta = A @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    V = A @ (Xu.T @ W @ Xu) @ A.T
    n, k = X.shape
    V *= n / (n - k)
    return beta, np.sqrt(np.diag(V))


def estimate(df: pd.DataFrame, outcome: str, W: np.ndarray, r_max: int):
    y = df[outcome].values.astype(float)
    X, Z, endog_names, instr_names = build_matrices(df, float(r_max), outcome)

    beta, se, bic = tsls(y, X, Z, W)
    second = pd.DataFrame({"term": endog_names, "coef": beta[1:1 + len(endog_names)],
                           "se": se[1:1 + len(endog_names)]})

    b1, se1 = ols_conley(X[:, 1], Z, W)
    first = pd.DataFrame({"term": instr_names, "coef": b1[1:1 + len(instr_names)],
                          "se": se1[1:1 + len(instr_names)]})

    resid_u = X[:, 1] - Z @ b1
    q = len(instr_names)
    n = len(y)
    resid_r = X[:, 1] - X[:, 1].mean()
    f_stat = ((resid_r @ resid_r - resid_u @ resid_u) / q) / (resid_u @ resid_u / (n - q - 1))

    return {"first": first, "second": second, "bic": bic, "r_max": r_max,
            "f_stat": f_stat, "total": beta[1:1 + len(endog_names)].sum(), "n": n,
            "mean": float(y.mean())}


def select_R(df: pd.DataFrame, outcome: str, W: np.ndarray):
    best = None
    for r_max in R_GRID:
        X, Z, _, _ = build_matrices(df, float(r_max), outcome)
        _, _, bic = tsls(df[outcome].values.astype(float), X, Z, W)
        if best is None or bic < best[1]:
            best = (r_max, bic)
    return best


# ============================================================================ table

def fmt(v, digits):
    return f"{v:,.{digits}f}"


def latex_table(res: dict, path: Path) -> None:
    digits = {o: 2 for o in OUTCOMES}

    def ordered(key):
        longest = max(OUTCOMES, key=lambda o: len(res[o][key]))
        return list(res[longest][key]["term"])

    def block(key, term_list):
        lines = []
        for t in term_list:
            coefs, ses = [], []
            for o in OUTCOMES:
                tab = res[o][key].set_index("term")
                if t in tab.index:
                    coefs.append(fmt(tab.loc[t, "coef"], 4 if key == "first" else digits[o]))
                    ses.append("(" + fmt(tab.loc[t, "se"], 4 if key == "first" else digits[o]) + ")")
                else:
                    coefs.append("")
                    ses.append("")
            lines.append(t + " & " + " & ".join(coefs) + r" \\")
            lines.append(" & " + " & ".join(ses) + r" \\")
        return lines

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Spillover effects of cash transfers on building stock, village level}",
        r"\label{tab:eq2-village}",
        r"\begin{tabular}{l" + "c" * len(OUTCOMES) + "}",
        r"\hline\hline",
        " & " + " & ".join(f"({i + 1})" for i in range(len(OUTCOMES))) + r" \\",
        " & " + " & ".join(TEX_LABELS[o] for o in OUTCOMES) + r" \\",
        r"\hline",
        r"\multicolumn{" + str(len(OUTCOMES) + 1)
        + r"}{l}{Panel A. First stage, dependent variable is own-village transfer amount} \\",
    ]
    lines += block("first", ordered("first"))
    lines.append("Excluded instruments $F$ & "
                 + " & ".join(fmt(res[o]["f_stat"], 1) for o in OUTCOMES) + r" \\")
    lines += [r"\hline",
              r"\multicolumn{" + str(len(OUTCOMES) + 1)
              + r"}{l}{Panel B. Second stage, two-stage least squares} \\"]
    lines += block("second", ordered("second"))
    lines.append("Sum of all amount terms & "
                 + " & ".join(fmt(res[o]["total"], digits[o]) for o in OUTCOMES) + r" \\")
    lines += [
        r"\hline",
        "Outcome mean & " + " & ".join(fmt(res[o]["mean"], digits[o]) for o in OUTCOMES) + r" \\",
        "Selected outer limit $\\bar{R}$ (km) & "
        + " & ".join(str(res[o]["r_max"]) for o in OUTCOMES) + r" \\",
        "Baseline control & "
        + " & ".join("Yes" if USE_BASELINE else "No" for _ in OUTCOMES) + r" \\",
        "Villages & " + " & ".join(f"{res[o]['n']:,}" for o in OUTCOMES) + r" \\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        f"Notes: each observation is one of the {res[OUTCOMES[0]]['n']} experimental villages. "
        f"Outcomes are the building footprint and building volume inside the village convex hull "
        f"in {OUTCOME_YEAR}, from Open Buildings 2.5D Temporal, counting 4 m pixels with building "
        f"presence at or above {PRESENCE_THRESHOLD}. "
        + (f"The {BASE_YEAR} value of the outcome is included as a control. " if USE_BASELINE else "")
        + f"Transfer amounts are in thousands of US dollars, one thousand per recipient household, "
        f"summed over the village or the ring, with no normalization. "
        f"The own-village amount is instrumented by village "
        r"treatment status and each ring amount by the share of eligible households assigned to "
        r"treatment in that ring, following Egger, Haushofer, Miguel, Niehaus and Walker (2022). "
        r"The outer limit $\bar{R}$ is selected by BIC over nested models from 2 km to 20 km. "
        f"Standard errors in parentheses are Conley standard errors with a uniform kernel and a "
        f"{CUTOFF_KM:.0f} km cutoff.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


# ============================================================================ run

if __name__ == "__main__":
    df = build_villages()
    W = conley_weights(df, CUTOFF_KM)

    res = {}
    for outcome in OUTCOMES:
        r_max = R_FIXED if R_FIXED is not None else select_R(df, outcome, W)[0]
        res[outcome] = estimate(df, outcome, W, r_max)

        print(f"\n=== {outcome}   R = {r_max} km   N = {res[outcome]['n']:,}   "
              f"mean = {res[outcome]['mean']:,.0f}")
        print("  Panel A, first stage of the own village amount")
        print(res[outcome]["first"].to_string(index=False, float_format=lambda v: f"{v:12,.4f}"))
        print(f"  excluded instruments F: {res[outcome]['f_stat']:.1f}")
        print("  Panel B, second stage")
        print(res[outcome]["second"].to_string(index=False, float_format=lambda v: f"{v:12,.1f}"))
        print(f"  sum of the amount terms: {res[outcome]['total']:,.1f}")

    latex_table(res, OUTDIR / "eq2_village_table.tex")
