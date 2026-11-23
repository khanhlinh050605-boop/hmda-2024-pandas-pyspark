"""Shared statistics (run only on small aggregated tables, so they are identical for every engine)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

Z95 = 1.959963984540054


# ------------------------------------------------------------------ confidence intervals
def wilson(k, n, z: float = Z95):
    """Wilson interval for the proportion k/n. Returns (p, lo, hi); n=0 -> NaN."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = np.where(n > 0, np.clip(centre - half, 0, 1), np.nan)
    hi = np.where(n > 0, np.clip(centre + half, 0, 1), np.nan)
    p = np.where(n > 0, p, np.nan)
    return p, lo, hi


def garwood(k, alpha: float = 0.05):
    """Exact Poisson interval (Garwood) for the count k."""
    k = np.asarray(k, dtype=float)
    lo = np.where(k > 0, sps.chi2.ppf(alpha / 2, 2 * np.maximum(k, 1)) / 2, 0.0)
    hi = sps.chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return lo, hi


def add_rate_ci(df: pd.DataFrame, k_col: str, n_col: str, prefix: str = "", small: int = 30) -> pd.DataFrame:
    """Add rate, ci_lo, ci_hi, small_cell to a count table."""
    p, lo, hi = wilson(df[k_col].values, df[n_col].values)
    df[f"{prefix}rate"] = p
    df[f"{prefix}ci_lo"] = lo
    df[f"{prefix}ci_hi"] = hi
    df[f"{prefix}small_cell"] = (df[n_col] < small).astype(int)
    return df


# ------------------------------------------------------------------ association
def cramers_v(table: np.ndarray) -> dict:
    """Cramér's V (and the Bergsma bias-corrected version) from an r x c cross-table."""
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0]
    t = t[:, t.sum(axis=0) > 0]
    r, c = t.shape
    n = t.sum()
    if r < 2 or c < 2 or n == 0:
        return {"v": np.nan, "v_bias_corrected": np.nan, "chi2": np.nan, "dof": 0, "n": n}
    exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    chi2 = float(((t - exp) ** 2 / exp).sum())
    k = min(r - 1, c - 1)
    v = float(np.sqrt(chi2 / (n * k)))
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    vc = float(np.sqrt(phi2c / max(min(rc - 1, cc - 1), 1e-12)))
    return {"v": v, "v_bias_corrected": vc, "chi2": chi2, "dof": (r - 1) * (c - 1), "n": float(n)}


def spearman(x, y) -> tuple[float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan, int(ok.sum())
    rho = sps.spearmanr(x[ok], y[ok]).statistic
    return float(rho), int(ok.sum())


def pearson_from_moments(n, sx, sy, sxx, syy, sxy) -> float:
    """Pearson r from sums (exact, can be assembled from any engine)."""
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def moments_to_summary(n, s, ss) -> dict:
    """mean, std (ddof=1) from n, sum and sum of squares."""
    n = float(n)
    if n <= 0:
        return {"mean": np.nan, "std": np.nan}
    mean = s / n
    var = (ss - n * mean * mean) / (n - 1) if n > 1 else np.nan
    return {"mean": mean, "std": float(np.sqrt(max(var, 0.0))) if var == var else np.nan}


# ------------------------------------------------------------------ dispersion
def dispersion(values, weights=None) -> dict:
    """Exact percentiles (linear interpolation), IQR, mean, std (ddof=1), weighted mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {"n": int(v.size)}
    if v.size == 0:
        return out
    for q in (5, 25, 50, 75, 95):
        out[f"p{q}"] = float(np.percentile(v, q))
    out["iqr"] = out["p75"] - out["p25"]
    out["mean"] = float(v.mean())
    out["std"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
    out["min"] = float(v.min())
    out["max"] = float(v.max())
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w[: v.size] if w.size != v.size else w
        out["weighted_mean"] = float(np.average(v, weights=w)) if w.sum() > 0 else np.nan
    return out


# ------------------------------------------------------------------ indirect standardisation
def indirect_standardise(df: pd.DataFrame, group_col: str | list, seg_cols: list[str],
                         n_col: str = "n_decided", k_col: str = "n_denied", ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """O/E by group, expected counts from the national rate of each segment (seg_cols). Garwood CI for O.
    `ref` (optional): national reference table by seg_cols (default: df itself)."""
    nat = (ref if ref is not None else df).groupby(seg_cols, as_index=False)[[n_col, k_col]].sum()
    nat["nat_rate"] = nat[k_col] / nat[n_col]
    m = df.merge(nat[seg_cols + ["nat_rate"]], on=seg_cols, how="left")
    m["expected_i"] = m[n_col] * m["nat_rate"]
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    g = m.groupby(keys, as_index=False).agg(n_decided=(n_col, "sum"), observed=(k_col, "sum"),
                                            expected=("expected_i", "sum"))
    g["oe_ratio"] = np.where(g["expected"] > 0, g["observed"] / g["expected"], np.nan)
    lo, hi = garwood(g["observed"].values)
    g["oe_lo"] = np.where(g["expected"] > 0, lo / g["expected"], np.nan)
    g["oe_hi"] = np.where(g["expected"] > 0, hi / g["expected"], np.nan)
    return g


# ------------------------------------------------------------------ OLS from per-cell sums
def ols_from_cells(cells: pd.DataFrame, cat_cols: list[str], cont_cols: list[str], y: str,
                   lender_cells: pd.DataFrame | None = None, cluster_col: str = "lei") -> dict:
    """OLS with coefficients = (X'X)^-1 X'y, built from per-cell sums.

    `cells` has columns: the categorical columns, `n`, `sum__<c>` (each continuous variable and y),
    `prod__<a>__<b>` for every pair a<=b in [y]+cont_cols. The reference level of each
    categorical variable is the level with the largest n (ties -> alphabetical order). Classical standard errors
    (homoskedastic); with millions of rows they mainly help read coefficient magnitudes.

    `lender_cells` (optional): same format as `cells` plus the column `cluster_col`. When given, adds
    cluster-robust CR1 standard errors (coefficients do NOT change): s_l = X_l'y_l - X_l'X_l beta; V = (X'X)^-1 (sum_l s_l s_l') (X'X)^-1
    * G/(G-1) * (N-1)/(N-K); columns se_cluster, t_cluster, ci_lo_cluster, ci_hi_cluster and G in the summary.
    """
    cells = cells[cells["n"] > 0].reset_index(drop=True)
    n_cell = cells["n"].to_numpy(float)
    ncell = len(cells)
    # --- dummy
    names, blocks = ["intercept"], [np.ones((ncell, 1))]
    ref, levels = {}, []          # levels: (column, level) in dummy order
    ref = {}
    for c in cat_cols:
        tot = cells.groupby(c)["n"].sum().sort_index()
        r = tot.idxmax()
        ref[c] = r
        for lvl in tot.index:
            if lvl == r:
                continue
            names.append(f"{c}[{lvl}]")
            levels.append((c, lvl))
            blocks.append((cells[c].to_numpy() == lvl).astype(float)[:, None])
    D = np.hstack(blocks)                       # ncell x q
    q = D.shape[1]
    k = len(cont_cols)
    p = q + k
    S1 = np.column_stack([cells[f"sum__{c}"].to_numpy(float) for c in cont_cols]) if k else np.zeros((ncell, 0))

    def prod(a, b):
        col = f"prod__{a}__{b}"
        if col not in cells:
            col = f"prod__{b}__{a}"
        return cells[col].to_numpy(float)

    XtX = np.zeros((p, p))
    XtX[:q, :q] = D.T @ (D * n_cell[:, None])
    if k:
        XtX[:q, q:] = D.T @ S1
        XtX[q:, :q] = XtX[:q, q:].T
        for i, a in enumerate(cont_cols):
            for j, b in enumerate(cont_cols):
                XtX[q + i, q + j] = prod(a, b).sum()
    sy = cells[f"sum__{y}"].to_numpy(float)
    Xty = np.zeros(p)
    Xty[:q] = D.T @ sy
    for i, a in enumerate(cont_cols):
        Xty[q + i] = prod(a, y).sum()
    yty = prod(y, y).sum()
    n = n_cell.sum()
    # --- solve with preconditioning (scaling by the square root of the diagonal) for numerical stability
    d = np.sqrt(np.diag(XtX))
    d[d == 0] = 1.0
    A = XtX / np.outer(d, d)
    b = Xty / d
    cond = float(np.linalg.cond(A))
    beta_s = np.linalg.solve(A, b)
    beta = beta_s / d
    rss = float(yty - beta @ Xty)
    dof = n - p
    sigma2 = rss / dof if dof > 0 else np.nan
    cov_s = np.linalg.inv(A) * sigma2
    cov = cov_s / np.outer(d, d)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ybar = sy.sum() / n
    tss = float(yty - n * ybar * ybar)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    tval = beta / se
    pval = 2 * sps.norm.sf(np.abs(tval))
    coef = pd.DataFrame({"term": names + list(cont_cols), "coef": beta, "se": se, "t": tval, "p_value": pval})
    coef["ci_lo"] = coef["coef"] - Z95 * coef["se"]
    coef["ci_hi"] = coef["coef"] + Z95 * coef["se"]
    extra = {}
    if lender_cells is not None:
        cl = _cluster_cr1(lender_cells, cells, cat_cols, cont_cols, y, cluster_col, levels, q, beta,
                          np.linalg.inv(A) / np.outer(d, d), n, p)
        coef["se_cluster"] = cl["se"]
        coef["t_cluster"] = beta / cl["se"]
        coef["ci_lo_cluster"] = beta - Z95 * cl["se"]
        coef["ci_hi_cluster"] = beta + Z95 * cl["se"]
        extra = {"G": cl["G"]}
    summary = {"y": y, "n": float(n), "n_terms": p, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan,
               "rmse": float(np.sqrt(rss / n)), "cond_scaled": cond, "n_cells": int(ncell),
               "reference_levels": "; ".join(f"{c}={v}" for c, v in ref.items()), **extra}
    return {"coef": coef, "summary": summary}


def _cluster_cr1(lc: pd.DataFrame, cells: pd.DataFrame, cat_cols, cont_cols, y, cluster_col, levels, q, beta, bread, n, p):
    """CR1 meat/sandwich from sums by (cell x cluster). `bread` = (X'X)^-1 (original scale)."""
    # keep only (cell x cluster) pairs whose cell was used when estimating the coefficients (min_cell_n)
    keep = cells[cat_cols].drop_duplicates()
    lc = lc[lc["n"] > 0].merge(keep, on=cat_cols, how="inner").reset_index(drop=True)
    n_l = lc["n"].to_numpy(float)
    if abs(n_l.sum() - n) > 1e-6 * max(n, 1.0):
        raise ValueError(f"total n by lender ({n_l.sum()}) differs from the regression n ({n})")
    m, k = len(lc), len(cont_cols)
    D = np.ones((m, q))
    for j, (c, lvl) in enumerate(levels, start=1):
        D[:, j] = (lc[c].to_numpy() == lvl)
    bq, bk = beta[:q], beta[q:]
    S1 = np.column_stack([lc[f"sum__{c}"].to_numpy(float) for c in cont_cols]) if k else np.zeros((m, 0))

    def prod(a, b):
        col = f"prod__{a}__{b}"
        if col not in lc:
            col = f"prod__{b}__{a}"
        return lc[col].to_numpy(float)

    sy = lc[f"sum__{y}"].to_numpy(float)
    dq = D @ bq
    S = np.zeros((m, p))
    S[:, :q] = D * (sy - n_l * dq - (S1 @ bk if k else 0.0))[:, None]
    for i, a in enumerate(cont_cols):
        Pb = sum(prod(a, b) * bk[j] for j, b in enumerate(cont_cols))
        S[:, q + i] = prod(a, y) - S1[:, i] * dq - Pb
    codes, uniq = pd.factorize(lc[cluster_col])
    G = len(uniq)
    Sg = np.zeros((G, p))
    np.add.at(Sg, codes, S)
    meat = Sg.T @ Sg
    V = bread @ meat @ bread * (G / (G - 1) * (n - 1) / (n - p) if G > 1 and n > p else np.nan)
    return {"se": np.sqrt(np.clip(np.diag(V), 0, None)), "G": int(G)}
