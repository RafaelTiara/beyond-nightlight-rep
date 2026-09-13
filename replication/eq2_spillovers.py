"""
Spillover equation of Egger, Haushofer, Miguel, Niehaus and Walker (2022), their Equation 2,
estimated by IV on grid cells rather than households.

    y_i = alpha + beta Amt_i + sum_r beta_r Amt_{i,r} + eps_i

First stage, one per endogenous regressor, on the full instrument set:

    Amt_i     = pi_0  + pi_1  Z_i + sum_r pi_{1r}  Z_{i,r} + u_i
    Amt_{i,r} = psi_0 + psi_1 Z_i + sum_s psi_{1s} Z_{i,s} + v_{i,r}

Adaptation to the grid:
  the own village becomes the own cell, so Amt_i is the cash delivered inside the cell
  and Amt_{i,r} the cash delivered in the ring [r, r+2) km, excluding the cell itself;
  instruments are shares of eligible households assigned to treatment, in the cell and in
  the ring, as in the paper.

Differences with the paper that come from the data rather than the implementation:
  the unit is a 0.005 degree cell, not a household;
  y_i is the midpoint of the band published in fig-map.csv;
  the lagged outcome y_{i,t=0} and the missing indicator M_iv are absent, because
  fig-map.csv is a single cross section. With Open Buildings 2.5D (2016 as the baseline)
  that term can be included.

Amounts: each recipient household received USD 1,000, and the amount in a cell or ring is
the number of recipients measured in thousands of dollars. No normalization by households
or by GDP. Coefficients read as the effect of one thousand extra dollars in that cell or ring.

Conley standard errors, uniform kernel and a 10 km cutoff, as in the paper.
The outer limit R is selected by BIC from 2 km to 20 km in steps of 2 km.

Requirements: pandas, numpy, scipy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ============================================================================ SETTINGS

REPO = Path(r"C:\Users\Rafael Tiara\Documents\GitHub\beyond-nightlight-rep")
PROJECT = Path(r"G:\Mi unidad\RESEARCH\beyond-nightlight-replication")

FIG_MAP = REPO / "fig_raw_data" / "fig-map.csv"
CENSUS = PROJECT / "GiveDirectlyOriginalPaper" / "GE_HH_Census_2017-07-17_cleanGPS.csv"
OUTDIR = REPO / "output-replication"

CELL = 0.005              # degrees
BAND_KM = 2.0             # width of each ring, km
R_GRID = range(2, 22, 2)  # candidate outer limits, km
R_FIXED = 4            # fixed outer limit in km; None lets BIC choose
CUTOFF_KM = 10.0          # Conley uniform kernel cutoff, km
TRANSFER_USD = 1.0        # transfer per recipient household, in thousands of dollars
OUTCOMES = ["building_footprint", "tin_roof_area", "night_light"]

BAND_MID = {
    "building_footprint": {"0%-1%": 0.005, "1%-2%": 0.015, "2%-3%": 0.025, "3%-4%": 0.035},
    "tin_roof_area": {"0%-1%": 0.005, "1%-2%": 0.015, "2%-3%": 0.025},
    "night_light": {"0.0-0.4": 0.2, "0.4-0.8": 0.6, "0.8-1.2": 1.0},
}
IS_SHARE = {"building_footprint": True, "tin_roof_area": True, "night_light": False}

TEX_LABELS = {
    "building_footprint": "Building footprint",
    "tin_roof_area": "Tin-roof area",
    "night_light": "Night light",
}


# ============================================================================ data

def cell_area_m2(lat: np.ndarray) -> np.ndarray:
    return (CELL * 111_320 * np.cos(np.radians(lat))) * (CELL * 110_574)


def build_cells() -> pd.DataFrame:
    """Cells with households, eligible households, eligible in treated villages, recipients, outcomes."""
    m = pd.read_csv(FIG_MAP)
    m["lon"] = m["lon"].round(4)
    m["lat"] = m["lat"].round(4)

    d = pd.read_csv(CENSUS, low_memory=False).dropna(subset=["latitude", "longitude"])
    d["recipient"] = ((d["eligible"] == 1) & (d["treat"] == 1)).astype(int)
    d["elig_treat"] = ((d["eligible"] == 1) & (d["treat"] == 1)).astype(int)
    d["elig"] = (d["eligible"] == 1).astype(int)
    d["lon"] = (np.floor(d["longitude"] / CELL) * CELL + CELL / 2).round(4)
    d["lat"] = (np.floor(d["latitude"] / CELL) * CELL + CELL / 2).round(4)
    g = (d.groupby(["lon", "lat"])
           .agg(hh=("elig", "size"), e=("elig", "sum"),
                et=("elig_treat", "sum"), x=("recipient", "sum"))
           .reset_index())

    df = m.merge(g, on=["lon", "lat"], how="inner")
    area = cell_area_m2(df["lat"].values)
    for out in OUTCOMES:
        mid = df[out].map(BAND_MID[out]).astype(float)
        df["y_" + out] = mid * area if IS_SHARE[out] else mid

    print(f"cells: {len(df):,}   households: {df['hh'].sum():,}   "
          f"eligible: {df['e'].sum():,}   recipients: {df['x'].sum():,}")
    return df


def km_coords(df: pd.DataFrame) -> np.ndarray:
    lat0 = np.radians(df["lat"].mean())
    return np.c_[df["lon"].values * 111.320 * np.cos(lat0), df["lat"].values * 110.574]


def ring_aggregates(df: pd.DataFrame, r_max_km: float) -> dict:
    """Amounts and instruments for each ring [r, r+2) km, excluding the own cell."""
    xy = km_coords(df)
    tree = cKDTree(xy)
    x = df["x"].values.astype(float)
    e = df["e"].values.astype(float)
    et = df["et"].values.astype(float)
    hh = df["hh"].values.astype(float)

    radii = np.arange(0.0, r_max_km + 1e-9, BAND_KM)
    cum = {}
    for r in radii[1:]:
        idx = tree.query_ball_point(xy, r=r)
        cum[r] = np.array([[x[j].sum(), e[j].sum(), et[j].sum(), hh[j].sum()] for j in idx])
    # the disc of radius zero is the cell itself
    cum[0.0] = np.c_[x, e, et, hh]

    out = {}
    for r in radii[:-1]:
        lo, hi = (r, r + BAND_KM)
        tot = cum[hi] - cum[lo] if lo > 0 else cum[hi] - cum[0.0]
        xr, er, etr, hhr = tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 3]
        amt = TRANSFER_USD * xr
        z = np.where(er > 0, etr / np.maximum(er, 1), 0.0)
        out[int(r)] = {"amt": amt, "z": z, "n_elig": er}
    return out


def own_cell_terms(df: pd.DataFrame):
    amt = TRANSFER_USD * df["x"].values.astype(float)
    z = np.where(df["e"].values > 0, df["et"].values / np.maximum(df["e"].values, 1), 0.0)
    return amt, z


# ============================================================================ estimation

def conley_weights(df: pd.DataFrame, cutoff_km: float) -> np.ndarray:
    xy = km_coords(df)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    return (d2 <= cutoff_km ** 2).astype(float)


def tsls(y: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray):
    """2SLS with a Conley variance. X and Z include the constant."""
    ZZ_inv = np.linalg.pinv(Z.T @ Z)
    Pz_X = Z @ (ZZ_inv @ (Z.T @ X))          # first stage fitted values of X
    A = np.linalg.pinv(Pz_X.T @ X)
    beta = A @ (Pz_X.T @ y)
    u = y - X @ beta
    Xu = Pz_X * u[:, None]
    meat = Xu.T @ W @ Xu
    V = A @ meat @ A.T
    n, k = X.shape
    V *= n / (n - k)
    ssr = float(u @ u)
    bic = n * np.log(ssr / n) + k * np.log(n)
    return beta, np.sqrt(np.diag(V)), bic, ssr


def ols_conley(y: np.ndarray, X: np.ndarray, W: np.ndarray):
    A = np.linalg.pinv(X.T @ X)
    beta = A @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    V = A @ (Xu.T @ W @ Xu) @ A.T
    n, k = X.shape
    V *= n / (n - k)
    return beta, np.sqrt(np.diag(V))


def build_matrices(df: pd.DataFrame, r_max_km: float):
    amt0, z0 = own_cell_terms(df)
    rings = ring_aggregates(df, r_max_km)
    names = ["Amt own cell"] + [f"Amt {r} to {r + int(BAND_KM)} km" for r in rings]
    X = np.c_[np.ones(len(df)), amt0, np.c_[[rings[r]["amt"] for r in rings]].T]
    Z = np.c_[np.ones(len(df)), z0, np.c_[[rings[r]["z"] for r in rings]].T]
    return X, Z, names


def select_R(df: pd.DataFrame, outcome: str, W: np.ndarray):
    """Modelos anidados de 2 a 20 km, se queda con el de menor BIC."""
    y = df["y_" + outcome].values
    best = None
    for r_max in R_GRID:
        X, Z, names = build_matrices(df, float(r_max))
        _, _, bic, _ = tsls(y, X, Z, W)
        if best is None or bic < best[1]:
            best = (r_max, bic)
    return best


def estimate(df: pd.DataFrame, outcome: str, W: np.ndarray, r_max: int):
    y = df["y_" + outcome].values
    X, Z, names = build_matrices(df, float(r_max))

    beta, se, bic, _ = tsls(y, X, Z, W)
    second = pd.DataFrame({"term": names, "coef": beta[1:], "se": se[1:]})

    # primera etapa del monto de la celda, sobre todos los instrumentos
    b1, se1 = ols_conley(X[:, 1], Z, W)
    first = pd.DataFrame({"term": ["Own-cell eligible share treated"]
                                  + [n.replace("Amt", "Share treated") for n in names[1:]],
                          "coef": b1[1:], "se": se1[1:]})
    # joint F of the excluded instruments, without the spatial correction
    resid_u = X[:, 1] - Z @ b1
    resid_r = X[:, 1] - X[:, 1].mean()
    q = Z.shape[1] - 1
    n = len(y)
    f_stat = ((resid_r @ resid_r - resid_u @ resid_u) / q) / (resid_u @ resid_u / (n - q - 1))

    total = beta[1:].sum()
    return {"first": first, "second": second, "bic": bic, "r_max": r_max,
            "f_stat": f_stat, "total": total, "n": n}


# ============================================================================ table

def fmt(v, digits):
    return f"{v:,.{digits}f}"


def latex_table(res: dict, df: pd.DataFrame, path: Path) -> None:
    digits = {"building_footprint": 2, "tin_roof_area": 2, "night_light": 6}

    def ordered_terms(key):
        """Union of terms across columns, in the order of the longest model."""
        longest = max(OUTCOMES, key=lambda o: len(res[o][key]))
        return list(res[longest][key]["term"])

    terms = ordered_terms("second")
    first_terms = ordered_terms("first")

    def block(key, term_list, digs):
        out = []
        for t in term_list:
            coefs, ses = [], []
            for o in OUTCOMES:
                tab = res[o][key].set_index("term")
                if t in tab.index:
                    coefs.append(fmt(tab.loc[t, "coef"], digs[o]))
                    ses.append("(" + fmt(tab.loc[t, "se"], digs[o]) + ")")
                else:
                    coefs.append("")
                    ses.append("")
            out.append(t + " & " + " & ".join(coefs) + r" \\")
            out.append(" & " + " & ".join(ses) + r" \\")
        return out

    first_digits = {o: 4 for o in OUTCOMES}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Spillover effects of cash transfers on remotely sensed outcomes, "
        + f"{CELL}$^\\circ$ grid cells}}",
        r"\label{tab:eq2-spillovers}",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        " & " + " & ".join(f"({i + 1})" for i in range(len(OUTCOMES))) + r" \\",
        " & " + " & ".join(TEX_LABELS[o] for o in OUTCOMES) + r" \\",
        " & (m$^2$) & (m$^2$) & (nW cm$^{-2}$ sr$^{-1}$) " + r"\\",
        r"\hline",
        r"\multicolumn{4}{l}{Panel A. First stage, dependent variable is own-cell transfer amount} \\",
    ]
    lines += block("first", first_terms, first_digits)
    lines.append("Excluded instruments $F$ & "
                 + " & ".join(fmt(res[o]["f_stat"], 1) for o in OUTCOMES) + r" \\")

    lines += [r"\hline", r"\multicolumn{4}{l}{Panel B. Second stage, two-stage least squares} \\"]
    lines += block("second", terms, digits)
    lines.append("Sum of all amount terms & "
                 + " & ".join(fmt(res[o]["total"], digits[o]) for o in OUTCOMES) + r" \\")

    lines += [
        r"\hline",
        "Selected outer limit $\\bar{R}$ (km) & "
        + " & ".join(str(res[o]["r_max"]) for o in OUTCOMES) + r" \\",
        "Observations & " + " & ".join(f"{res[o]['n']:,}" for o in OUTCOMES) + r" \\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        f"Notes: each observation is a {CELL}$^\\circ$ grid cell. Transfer amounts are in thousands "
        f"of US dollars, one thousand per recipient household, summed over the cell or the ring, with "
        f"no normalization. Amounts in the own cell and in each {BAND_KM:.0f} km ring "
        r"are instrumented by the share of eligible households assigned to treatment in the "
        r"corresponding cell or ring, following Egger, Haushofer, Miguel, Niehaus and Walker (2022). "
        r"Outcomes are the midpoints of the bands published in \texttt{fig-map.csv}. The outer "
        r"limit $\bar{R}$ is selected by BIC over nested models from 2 km to 20 km. Standard errors "
        f"in parentheses are Conley standard errors with a uniform kernel and a {CUTOFF_KM:.0f} km cutoff.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


# ============================================================================ run

if __name__ == "__main__":
    df = build_cells()
    W = conley_weights(df, CUTOFF_KM)

    res = {}
    for outcome in OUTCOMES:
        if R_FIXED is None:
            r_max, bic = select_R(df, outcome, W)
            print(f"{outcome:20s} R selected by BIC: {r_max} km")
        else:
            r_max = R_FIXED
        res[outcome] = estimate(df, outcome, W, r_max)

        print(f"\n=== {outcome}   R = {r_max} km   N = {res[outcome]['n']:,}")
        print("  Panel A, first stage of the own-cell amount")
        print(res[outcome]["first"].to_string(index=False,
                                              float_format=lambda v: f"{v:12,.4f}"))
        print(f"  excluded instruments F: {res[outcome]['f_stat']:.1f}")
        print("  Panel B, second stage")
        print(res[outcome]["second"].to_string(index=False,
                                               float_format=lambda v: f"{v:12,.3f}"))
        print(f"  sum of the amount terms: {res[outcome]['total']:,.3f}\n")

    latex_table(res, df, OUTDIR / "eq2_spillovers_table.tex")
