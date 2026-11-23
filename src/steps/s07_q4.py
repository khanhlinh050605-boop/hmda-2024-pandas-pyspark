"""Step 07 - RQ Q4: for originated loans, how do interest rate and rate spread differ by loan type, purpose, lien;
which loan characteristics are associated with them (OLS regression, association only)?

Pricing fields (interest_rate, rate_spread, loan_term, total_loan_costs) are used only to DESCRIBE originated loans.
Regression: OLS built from per-cell sums (normal equations); the engines return only per-cell sums -> one common solver for Pandas and Spark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import stats as ST
from ._util import FINE, PRODUCT, PROBS_DESC, raw_moments, raw_pct, raw_sum

NAME = "s07_q4"
ENGINE_BOUND = True
SECTIONS = ["regression", "plausibility", "thresholds"]

Y_FINE = ["interest_rate", "rate_spread", "loan_term", "total_loan_costs"]
Y_PRODUCT = ["interest_rate", "rate_spread"]


def _summ(m: pd.DataFrame, p: pd.DataFrame, y: str, keys: list[str]) -> pd.DataFrame:
    out = m.merge(p, on=keys, how="outer", suffixes=("", "_p")) if keys else pd.concat([m, p], axis=1)
    rows = []
    for _, r in out.iterrows():
        s = ST.moments_to_summary(r["n"], r[f"sum__{y}"], r[f"prod__{y}__{y}"])
        d = {k: r[k] for k in keys}
        d.update({"n_valid": int(r["n"]), "mean": s["mean"], "std": s["std"], "min": r[f"min__{y}"], "max": r[f"max__{y}"]})
        for c in p.columns:
            if c.startswith("p") and c not in keys and c != "n_valid":
                d[c] = r[c]
        d["iqr"] = d["p75"] - d["p25"]
        rows.append(d)
    return pd.DataFrame(rows)


def run(ctx):
    cfg = ctx.cfg
    # --- 1. description by segment
    for y in Y_FINE:
        w = ["is_orig", f"v_{y}"]
        m = raw_moments(ctx, f"q4_mom_{y}_fine", FINE, [y], where=w, extremes=True)
        p = raw_pct(ctx, f"q4_pct_{y}_fine", FINE, y, PROBS_DESC, where=w)
        ctx.save_table(f"q4_summary_{y}_by_segment", _summ(m, p.drop(columns=["n_valid", "min", "max"]), y, FINE))
    for y in Y_PRODUCT:
        w = ["is_orig", f"v_{y}"]
        m = raw_moments(ctx, f"q4_mom_{y}_product", PRODUCT, [y], where=w, extremes=True)
        p = raw_pct(ctx, f"q4_pct_{y}_product", PRODUCT, y, PROBS_DESC, where=w)
        ctx.save_table(f"q4_summary_{y}_by_product", _summ(m, p.drop(columns=["n_valid", "min", "max"]), y, PRODUCT))
        m = raw_moments(ctx, f"q4_mom_{y}_type", ["loan_type"], [y], where=w, extremes=True)
        p = raw_pct(ctx, f"q4_pct_{y}_type", ["loan_type"], y, PROBS_DESC, where=w)
        ctx.save_table(f"q4_summary_{y}_by_loan_type", _summ(m, p.drop(columns=["n_valid", "min", "max"]), y, ["loan_type"]))

    # --- 2. coverage: how many originated loans have a valid value
    cov_rows = []
    for y in Y_FINE:
        d = raw_sum(ctx, f"q4_status_{y}", FINE + [f"{y}_status"], ["v_" + y], where=["is_orig"])
        d = d.rename(columns={f"{y}_status": "status", f"v_{y}": "n_valid_flag"})
        d.insert(0, "variable", y)
        cov_rows.append(d)
    ctx.save_table("q4_coverage", pd.concat(cov_rows, ignore_index=True))

    # --- 3. OLS regression from per-cell sums
    reg = cfg["regression"]
    cat, cont = reg["categorical"], reg["continuous"]
    sums_rows, coef_all = [], []
    for y in reg["targets"]:
        flag = "q4_reg_rate" if y == "interest_rate" else "q4_reg_spread"
        cols = [y] + cont
        cells = raw_moments(ctx, f"q4_reg_cells_{y}", cat, cols, where=[flag], cross=True)
        cells = cells[cells["n"] >= reg.get("min_cell_n", 1)]
        lcells = raw_moments(ctx, f"q4_reg_cells_by_lender_{y}", cat + ["lei"], cols, where=[flag], cross=True)
        lcells = lcells[lcells["n"] >= 1]
        res = ST.ols_from_cells(cells, cat, cont, y, lender_cells=lcells, cluster_col="lei")
        coef = res["coef"]
        coef.insert(0, "target", y)
        ctx.save_table(f"q4_reg_coef_{y}", coef)
        coef_all.append(coef)
        sums_rows.append(res["summary"])
    ctx.save_table("q4_reg_summary", pd.DataFrame(sums_rows))
    ctx.save_table("q4_reg_coef_all", pd.concat(coef_all, ignore_index=True))
    return {"targets": reg["targets"]}
