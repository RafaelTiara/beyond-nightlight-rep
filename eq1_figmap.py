"""
Ecuaciones 1 y 2 estimadas con fig-map.csv, tomando el punto medio de cada banda.

    y_i = sum_k tau_k 1{x_i = k} + sum_m beta_m 1{e_i = m} + eps_i      (1)
    y_i = tau x_i          + sum_m beta_m 1{e_i = m} + eps_i            (2)

Diferencias con el paper, que vienen del dato y no de la implementación:
  la celda es de 0,005 grados y no de 0,001, porque así viene fig-map.csv;
  y_i es el punto medio de la banda publicada, no el valor observado.

x_i y e_i no salen de fig-map.csv, se calculan desde el censo de hogares:
receptor es elegible en aldea de tratamiento. El cruce con fig-map.csv es exacto
en las 2.501 celdas y el tramo de receptores coincide en 99,8 por ciento con
`treatment_intensity`, lo que sirve de chequeo.

Errores estándar a la Conley, kernel uniforme y corte de 3 km.

Requisitos: pandas, numpy, statsmodels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================================ SETTINGS

REPO = Path(r"C:\Users\Rafael Tiara\Documents\GitHub\beyond-nightlight-rep")
PROJECT = Path(r"G:\Mi unidad\RESEARCH\beyond-nightlight-replication")

FIG_MAP = REPO / "fig_raw_data" / "fig-map.csv"
CENSUS = PROJECT / "GiveDirectlyOriginalPaper" / "GE_HH_Census_2017-07-17_cleanGPS.csv"

OUTDIR = REPO / "output-replication"

CELL = 0.005            # tamaño de celda en grados, el de fig-map.csv
CUTOFF_KM = 3.0         # corte del kernel uniforme de Conley
X_BINS = "paper"        # "paper" para K = {0, 1, 2, 2+}; "figmap" para 0-3, 4-6, 7-9, 10-12, 13+
E_MAX = 20              # el control por elegibles se agrupa en este valor hacia arriba
OUTCOMES = ["building_footprint", "tin_roof_area", "night_light"]

# Puntos medios de cada banda publicada.
# Huella y techo vienen en porcentaje del área de la celda, se pasan a m2.
BAND_MID = {
    "building_footprint": {"0%-1%": 0.005, "1%-2%": 0.015, "2%-3%": 0.025, "3%-4%": 0.035},
    "tin_roof_area": {"0%-1%": 0.005, "1%-2%": 0.015, "2%-3%": 0.025},
    "night_light": {"0.0-0.4": 0.2, "0.4-0.8": 0.6, "0.8-1.2": 1.0},
}
IS_SHARE = {"building_footprint": True, "tin_roof_area": True, "night_light": False}


# ============================================================================ datos

def cell_area_m2(lat: np.ndarray) -> np.ndarray:
    """Área de una celda de CELL grados a esa latitud, en m2."""
    return (CELL * 111_320 * np.cos(np.radians(lat))) * (CELL * 110_574)


def build_panel() -> pd.DataFrame:
    m = pd.read_csv(FIG_MAP)
    m["lon"] = m["lon"].round(4)
    m["lat"] = m["lat"].round(4)

    d = pd.read_csv(CENSUS, low_memory=False).dropna(subset=["latitude", "longitude"])
    d["recipient"] = ((d["eligible"] == 1) & (d["treat"] == 1)).astype(int)
    d["lon"] = (np.floor(d["longitude"] / CELL) * CELL + CELL / 2).round(4)
    d["lat"] = (np.floor(d["latitude"] / CELL) * CELL + CELL / 2).round(4)
    g = (d.groupby(["lon", "lat"])
           .agg(x=("recipient", "sum"), e=("eligible", "sum"))
           .reset_index())

    df = m.merge(g, on=["lon", "lat"], how="inner")
    print(f"celdas en fig-map.csv: {len(m):,}; cruzadas con el censo: {len(df):,}")

    # chequeo contra la variable publicada
    lab = ["0-3", "4-6", "7-9", "10-12", "13-"]
    mybin = pd.cut(df["x"], bins=[-1, 3, 6, 9, 12, 10 ** 6], labels=lab).astype(str)
    print(f"coincidencia con treatment_intensity: {(mybin == df['treatment_intensity']).mean():.3f}")

    area = cell_area_m2(df["lat"].values)
    for out in OUTCOMES:
        mid = df[out].map(BAND_MID[out]).astype(float)
        df["y_" + out] = mid * area if IS_SHARE[out] else mid

    df = df[df["e"] > 0].copy()
    df["e_grp"] = df["e"].clip(upper=E_MAX).astype(int)
    return df


# ============================================================================ Conley

def conley_ols(y: np.ndarray, X: np.ndarray, lon: np.ndarray, lat: np.ndarray,
               cutoff_km: float = CUTOFF_KM):
    """OLS con varianza espacial de Conley, kernel uniforme y corte en cutoff_km."""
    res = sm.OLS(y, X).fit()
    u = res.resid
    n, k = X.shape

    # distancias en km, aproximación plana, suficiente en un área de medio grado
    lat0 = np.radians(lat.mean())
    xk = lon * 111.320 * np.cos(lat0)
    yk = lat * 110.574
    W = ((xk[:, None] - xk[None, :]) ** 2 + (yk[:, None] - yk[None, :]) ** 2) <= cutoff_km ** 2

    Xu = X * u[:, None]
    meat = Xu.T @ W @ Xu
    bread = np.linalg.inv(X.T @ X)
    V = bread @ meat @ bread
    V *= n / (n - k)                      # corrección de grados de libertad
    return res.params, np.sqrt(np.diag(V)), res


def design(df: pd.DataFrame, spec: str):
    """Matriz de diseño: tratamiento (en tramos o lineal) más dummies de elegibles."""
    e_dum = pd.get_dummies(df["e_grp"], prefix="e", drop_first=True).astype(float)
    if spec == "eq1":
        if X_BINS == "paper":
            lab = ["0", "1", "2", "2+"]
            xb = pd.cut(df["x"], bins=[-1, 0, 1, 2, 10 ** 6], labels=lab)
        else:
            lab = ["0-3", "4-6", "7-9", "10-12", "13+"]
            xb = pd.cut(df["x"], bins=[-1, 3, 6, 9, 12, 10 ** 6], labels=lab)
        t = pd.get_dummies(xb, prefix="tau").astype(float).drop(columns=f"tau_{lab[0]}")
    else:
        t = df[["x"]].astype(float).rename(columns={"x": "tau"})
    X = pd.concat([t, e_dum], axis=1)
    X.insert(0, "const", 1.0)
    return X, list(t.columns)


def estimate(df: pd.DataFrame, outcome: str, spec: str):
    X, tcols = design(df, spec)
    y = df["y_" + outcome].values
    b, se, res = conley_ols(y, X.values.astype(float),
                            df["lon"].values, df["lat"].values)
    idx = {c: i for i, c in enumerate(X.columns)}
    rows = []
    for c in tcols:
        i = idx[c]
        rows.append({"term": c, "coef": b[i], "conley_se": se[i], "t": b[i] / se[i]})
    return pd.DataFrame(rows), res



# ============================================================================ tabla

TEX_LABELS = {
    "building_footprint": "Building footprint",
    "tin_roof_area": "Tin-roof area",
    "night_light": "Night light",
}
ROW_LABELS = {"tau_1": "1 recipient", "tau_2": "2 recipients", "tau_2+": "More than 2 recipients",
              "tau_0-3": "0 to 3", "tau_4-6": "4 to 6", "tau_7-9": "7 to 9",
              "tau_10-12": "10 to 12", "tau_13+": "13 or more", "tau": "Recipients per cell"}


def fmt(v, digits):
    return f"{v:,.{digits}f}"


def latex_table(results: dict, df: pd.DataFrame, path: Path) -> None:
    """Tabla de dos paneles, ecuación 1 arriba y ecuación 2 abajo, en un fragmento .tex."""
    digits = {"building_footprint": 2, "tin_roof_area": 2, "night_light": 4}
    terms = list(results[OUTCOMES[0]]["eq1"]["tab"]["term"])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Program effects on remotely sensed outcomes, "
        + f"{CELL}$^\\circ$ grid cells}}",
        r"\label{tab:eq1-figmap}",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        " & " + " & ".join(f"({i + 1})" for i in range(len(OUTCOMES))) + r" \\",
        " & " + " & ".join(TEX_LABELS[o] for o in OUTCOMES) + r" \\",
        " & (m$^2$) & (m$^2$) & (nW cm$^{-2}$ sr$^{-1}$) " + r"\\",
        r"\hline",
        r"\multicolumn{4}{l}{Panel A. Equation 1, binned treatment intensity} \\",
    ]
    for t in terms:
        coefs, ses = [], []
        for o in OUTCOMES:
            r = results[o]["eq1"]["tab"].set_index("term").loc[t]
            coefs.append(fmt(r["coef"], digits[o]))
            ses.append("(" + fmt(r["conley_se"], digits[o]) + ")")
        lines.append(ROW_LABELS.get(t, t) + " & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")

    lines += [r"\hline", r"\multicolumn{4}{l}{Panel B. Equation 2, linear treatment intensity} \\"]
    coefs, ses = [], []
    for o in OUTCOMES:
        r = results[o]["eq2"]["tab"].set_index("term").loc["tau"]
        coefs.append(fmt(r["coef"], digits[o]))
        ses.append("(" + fmt(r["conley_se"], digits[o]) + ")")
    lines.append(ROW_LABELS["tau"] + " & " + " & ".join(coefs) + r" \\")
    lines.append(" & " + " & ".join(ses) + r" \\")

    lines += [
        r"\hline",
        "Outcome mean & " + " & ".join(fmt(df["y_" + o].mean(), digits[o]) for o in OUTCOMES)
        + r" \\",
        "Observations & " + " & ".join(f"{len(df):,}" for _ in OUTCOMES) + r" \\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        f"Notes: each observation is a {CELL}$^\\circ$ grid cell with at least one eligible "
        r"household. Outcomes are the midpoints of the published bands in \texttt{fig-map.csv}; "
        r"building footprint and tin-roof area are converted from shares of cell area to square "
        r"meters. Recipients and eligible households per cell are computed from the baseline "
        r"household census. All specifications control flexibly for the number of eligible "
        r"households. Standard errors in parentheses are Conley standard errors with a uniform "
        f"kernel and a {CUTOFF_KM:.0f} km cutoff.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("guardado", path)


# ============================================================================ ejecución

if __name__ == "__main__":
    df = build_panel()
    print(f"celdas con al menos un elegible: {len(df):,}; "
          f"receptores por celda: media {df['x'].mean():.1f}, máximo {df['x'].max():.0f}")
    print(f"elegibles por celda: media {df['e'].mean():.1f}, máximo {df['e'].max():.0f}\n")

    results = {}
    for outcome in OUTCOMES:
        unit = "m2 por celda" if IS_SHARE[outcome] else "nW cm-2 sr-1"
        print(f"=== {outcome} ({unit}), media {df['y_' + outcome].mean():,.2f}")
        results[outcome] = {}
        for spec, title in [("eq1", "Ecuación 1, tramos"), ("eq2", "Ecuación 2, lineal")]:
            tab, res = estimate(df, outcome, spec)
            results[outcome][spec] = {"tab": tab, "res": res}
            print(f"  {title}   N = {int(res.nobs):,}   R2 = {res.rsquared:.3f}")
            print(tab.to_string(index=False, float_format=lambda v: f"{v:12,.4f}"))
        print()

    latex_table(results, df, OUTDIR / "eq1_figmap_table.tex")
