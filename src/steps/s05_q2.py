"""Step 05 - RQ Q2: how dispersed are lender denial rates; how much difference remains within the same product segment?

One count block (lender x loan_type x purpose x lien) from the engine. Only lenders with >= lender_min_decided decisions are compared
(lenders below the threshold stay in the table, flagged). The TS is only used to attach name/agency labels to lenders and does not enter the results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import schema as S
from ..core import stats as ST
from ._util import FINE, aggregate, raw_sum, read_ts, seg_label, to_counts, top_segments, with_rates

NAME = "s05_q2"
ENGINE_BOUND = True
SECTIONS = ["thresholds", "paths"]


def run(ctx):
    cfg = ctx.cfg
    thr = cfg["thresholds"]
    small, minn = thr["small_cell"], thr["lender_min_decided"]
    cube = to_counts(raw_sum(ctx, "q2_cube_lender_seg", ["lei"] + FINE))
    nat_seg = aggregate(cube, FINE)
    nat_total = aggregate(cube, [])
    nat_rate = nat_total["n_denied"].iloc[0] / nat_total["n_decided"].iloc[0]
    tops = top_segments(nat_seg, thr["top_segments"])

    # --- lender table
    lend = with_rates(aggregate(cube, ["lei"]), small)
    lend["eligible"] = (lend["n_decided"] >= minn).astype(int)
    std = ST.indirect_standardise(cube, "lei", FINE, ref=nat_seg)[["lei", "expected", "oe_ratio", "oe_lo", "oe_hi"]]
    lend = lend.merge(std, on="lei", how="left")
    lend["standardised_rate"] = lend["oe_ratio"] * nat_rate
    lend["oe_above_1"] = (lend["oe_lo"] > 1).astype(int)
    lend["oe_below_1"] = (lend["oe_hi"] < 1).astype(int)
    lend["raw_ci_above_national"] = (lend["denial_ci_lo"] > nat_rate).astype(int)
    lend["raw_ci_below_national"] = (lend["denial_ci_hi"] < nat_rate).astype(int)
    ts = read_ts(cfg)
    if ts is not None:
        lend = lend.merge(ts[["lei", "respondent_name", "agency_code"]], on="lei", how="left")
    lend = lend.sort_values("n_decided", ascending=False).reset_index(drop=True)
    ctx.save_table("q2_lender_table", lend)

    el = lend[lend["eligible"] == 1]
    # --- overall dispersion
    rows = []
    d = ST.dispersion(el["denial_rate"].values, el["n_decided"].values)
    rows.append({"scope": "all products (raw lender rate)", **d})
    d = ST.dispersion(el["standardised_rate"].values, el["n_decided"].values)
    rows.append({"scope": "all products (product-mix standardised rate = O/E x national rate)", **d})
    d = ST.dispersion(el["oe_ratio"].values, el["n_decided"].values)
    rows.append({"scope": "O/E ratio (1.0 = as expected from product mix)", **d})

    # --- dispersion within each of the largest segments
    seg_cube = cube.copy()
    seg_cube["segment"] = seg_label(seg_cube)
    seg_lender_rows = []
    for sg in tops:
        g = seg_cube[seg_cube["segment"] == sg].copy()
        g = with_rates(g, small)
        g["eligible_in_segment"] = (g["n_decided"] >= minn).astype(int)
        seg_lender_rows.append(g)
        ge = g[g["eligible_in_segment"] == 1]
        d = ST.dispersion(ge["denial_rate"].values, ge["n_decided"].values)
        d["national_segment_rate"] = float(g["n_denied"].sum() / g["n_decided"].sum())
        rows.append({"scope": f"segment {sg}", **d})
    disp = pd.DataFrame(rows)
    disp["lenders_total"] = int(len(lend))
    disp["lenders_eligible_overall"] = int(len(el))
    ctx.save_table("q2_dispersion", disp)
    ctx.save_table("q2_segment_lender_rates", pd.concat(seg_lender_rows, ignore_index=True))

    # --- lender vs national, size, concentration
    n_el = len(el)
    summ = {
        "lenders_total": int(len(lend)), "lenders_eligible": int(n_el), "min_decided": minn,
        "share_of_decided_by_eligible": float(el["n_decided"].sum() / lend["n_decided"].sum()),
        "national_denial_rate": nat_rate,
        "eligible_raw_ci_above_national": int(el["raw_ci_above_national"].sum()),
        "eligible_raw_ci_below_national": int(el["raw_ci_below_national"].sum()),
        "eligible_oe_above_1": int(el["oe_above_1"].sum()), "eligible_oe_below_1": int(el["oe_below_1"].sum()),
        "std_raw": float(el["denial_rate"].std(ddof=1)) if n_el > 1 else np.nan,
        "std_standardised": float(el["standardised_rate"].std(ddof=1)) if n_el > 1 else np.nan,
        "iqr_raw": float(disp.loc[disp["scope"] == "all products (raw lender rate)", "iqr"].iloc[0]),
        "iqr_standardised": float(disp.loc[disp["scope"].str.startswith("all products (product-mix"), "iqr"].iloc[0]),
    }
    if n_el > 3:
        summ["spearman_log_size_vs_rate"] = ST.spearman(np.log(el["n_decided"].values), el["denial_rate"].values)[0]
        summ["spearman_log_size_vs_oe"] = ST.spearman(np.log(el["n_decided"].values), el["oe_ratio"].values)[0]
    top10 = lend.nlargest(10, "n_decided")["n_decided"].sum() / lend["n_decided"].sum()
    summ["top10_lenders_share_of_decided"] = float(top10)
    summ["std_reduction_share"] = (1 - summ["std_standardised"] / summ["std_raw"]) if summ["std_raw"] else np.nan
    ctx.save_table("q2_summary", pd.DataFrame([summ]))
    return {"eligible": n_el}
