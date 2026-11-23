"""Bước 06 - RQ Q3: income, loan amount, loan-to-income, DTI (và CLTV, property value) liên hệ thế nào với denial?

Mỗi biến: một khối đếm [band x loan_type x purpose x lien] cho hồ sơ decided. Từ đó:
  - denial rate theo band (Wilson), O/E theo band (đã chuẩn hoá cơ cấu sản phẩm),
  - Cramér's V (chỉ band hợp lệ) tổng thể và trong từng phân khúc lớn nhất,
  - tỷ lệ ghi NA/Exempt/ngoài miền theo outcome (có ghi khác nhau giữa denied và approved không).
Chỉ là liên hệ (association); số lượng lớn làm mọi chi-bình-phương 'có ý nghĩa', nên dùng độ lớn V để đọc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import stats as ST
from ..core.definitions import Q3_VARS
from ._util import FINE, PROBS_DESC, SUMS, aggregate, raw_moments, raw_pct, raw_sum, seg_label, sort_by_order, to_counts, top_segments, with_rates

NAME = "s06_q3"
ENGINE_BOUND = True
SECTIONS = ["thresholds", "bands", "plausibility"]

CONT = [("lti", "v_lti"), ("income", "v_income"), ("loan_amount", "v_loan_amount"), ("cltv", "v_cltv")]


def _v_table(cube, band_col, valid_labels):
    g = cube[cube[band_col].isin(valid_labels)].groupby(band_col)[["n_decided", "n_denied"]].sum()
    tab = np.column_stack([g["n_denied"].values, (g["n_decided"] - g["n_denied"]).values])
    return tab


def run(ctx):
    cfg, defs = ctx.cfg, ctx.defs
    small = cfg["thresholds"]["small_cell"]
    assoc, status_rows, band_all = [], [], []
    tops = None
    for var, (band_col, val_col, label) in Q3_VARS.items():
        cube = raw_sum(ctx, f"q3_cube_{var}", [band_col] + FINE, ["is_decided", "is_denied"])
        cube = to_counts(cube)
        if tops is None:
            tops = top_segments(aggregate(cube, FINE), cfg["thresholds"]["top_segments"])
        order = defs.order_of(band_col)
        valid = defs.valid_labels(band_col)
        # --- bảng theo band
        t = aggregate(cube, [band_col], cols=("n_rows", "n_decided", "n_denied"))
        t = with_rates(t, small, orig=False)
        t["share_of_decided"] = t["n_decided"] / t["n_decided"].sum()
        oe = ST.indirect_standardise(cube, band_col, FINE)[[band_col, "expected", "oe_ratio", "oe_lo", "oe_hi"]]
        t = t.merge(oe, on=band_col, how="left")
        t["is_status_band"] = t[band_col].map(defs.is_status_band).astype(int)
        t = sort_by_order(t, band_col, order)
        t.insert(0, "variable", var)
        t = t.rename(columns={band_col: "band"})
        ctx.save_table(f"q3_band_rates_{var}", t)
        band_all.append(t)

        # --- Cramér's V
        v_all = ST.cramers_v(_v_table(cube, band_col, valid))
        tab_incl = np.column_stack([aggregate(cube, [band_col])["n_denied"].values,
                                    (aggregate(cube, [band_col])["n_decided"] - aggregate(cube, [band_col])["n_denied"]).values])
        v_incl = ST.cramers_v(tab_incl)
        row = {"variable": var, "label": label, "n_decided_valid_bands": v_all["n"], "V": v_all["v"],
               "V_bias_corrected": v_all["v_bias_corrected"], "chi2": v_all["chi2"], "dof": v_all["dof"],
               "V_incl_status_bands": v_incl["v"],
               "share_decided_in_status_bands": float(t.loc[t["is_status_band"] == 1, "n_decided"].sum() / t["n_decided"].sum())}
        vt = t[t["is_status_band"] == 0]
        vt = vt[vt["n_decided"] > 0]
        row["rate_min_valid_band"] = float(vt["denial_rate"].min()) if len(vt) else np.nan
        row["rate_max_valid_band"] = float(vt["denial_rate"].max()) if len(vt) else np.nan
        row["rate_range_pp"] = (row["rate_max_valid_band"] - row["rate_min_valid_band"]) * 100 if len(vt) else np.nan
        row["spearman_band_order_vs_rate"] = ST.spearman(np.arange(len(vt)), vt["denial_rate"].values)[0] if len(vt) >= 3 else np.nan
        row["oe_range_valid_bands"] = float(vt["oe_ratio"].max() - vt["oe_ratio"].min()) if len(vt) else np.nan
        # trong từng phân khúc lớn
        cube["segment"] = seg_label(cube)
        for sg in tops:
            vs = ST.cramers_v(_v_table(cube[cube["segment"] == sg], band_col, valid))
            row[f"V_in_{sg}"] = vs["v"]
        assoc.append(row)

        # --- trạng thái ghi theo outcome
        for _, r in aggregate(cube, [band_col], cols=("n_decided", "n_denied")).iterrows():
            if defs.is_status_band(r[band_col]):
                status_rows.append({"variable": var, "status_band": r[band_col], "n_decided": r["n_decided"],
                                    "n_denied": r["n_denied"], "n_approved": r["n_decided"] - r["n_denied"]})
    ctx.save_table("q3_association", pd.DataFrame(assoc))
    ctx.save_table("q3_band_rates_all", pd.concat(band_all, ignore_index=True))
    st = pd.DataFrame(status_rows)
    if len(st):
        tot = {}
        for var, (band_col, _, _) in Q3_VARS.items():
            b = pd.read_csv(ctx.tab_dir / f"q3_band_rates_{var}.csv")
            tot[var] = (b["n_denied"].sum(), (b["n_decided"] - b["n_denied"]).sum())
        st["share_of_denied"] = [r.n_denied / tot[r.variable][0] for r in st.itertuples()]
        st["share_of_approved"] = [r.n_approved / tot[r.variable][1] for r in st.itertuples()]
        st["gap_denied_minus_approved"] = st["share_of_denied"] - st["share_of_approved"]
    ctx.save_table("q3_status_by_outcome", st)

    # --- quan hệ loan amount ~ income (log-log)
    cols = ["log_loan_amount", "log_income"]
    rel = []
    for scope, extra in (("all applications", "is_app"), ("originated", "is_orig")):
        m = raw_moments(ctx, f"q3_corr_{extra}", [], cols, where=["v_loan_amount", "v_income", extra], cross=True).iloc[0]
        n = m["n"]
        r = ST.pearson_from_moments(n, m["sum__log_loan_amount"], m["sum__log_income"], m["prod__log_loan_amount__log_loan_amount"],
                                    m["prod__log_income__log_income"], m["prod__log_loan_amount__log_income"])
        vx = m["prod__log_income__log_income"] - m["sum__log_income"] ** 2 / n
        cov = m["prod__log_loan_amount__log_income"] - m["sum__log_loan_amount"] * m["sum__log_income"] / n
        rel.append({"scope": scope, "n": int(n), "pearson_r_log_loan_vs_log_income": r,
                    "elasticity_slope_loan_on_income": cov / vx if vx > 0 else np.nan})
    ctx.save_table("q3_loan_income_relation", pd.DataFrame(rel))
    p = raw_pct(ctx, "q3_loan_by_income_band", ["income_band"], "loan_amount", PROBS_DESC, where=["v_loan_amount"])
    ctx.save_table("q3_loan_by_income_band", sort_by_order(p, "income_band", defs.order_of("income_band")))

    # --- biến liên tục theo outcome (decided)
    for v, flag in CONT:
        d = raw_pct(ctx, f"q3_cont_by_outcome_{v}", ["is_denied"], v, PROBS_DESC, where=["is_decided", flag])
        d["outcome"] = np.where(d["is_denied"] == "1", "denied", "approved")
        ctx.save_table(f"q3_cont_by_outcome_{v}", d)
    return {"tops": tops}
