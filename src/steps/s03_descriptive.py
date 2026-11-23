"""Bước 03 - Thống kê mô tả (đề bài 3.3A): phân bố kết quả hồ sơ, loan amount, income, lãi suất, v.v.

Mỗi thống kê ghi rõ mẫu (scope) vì phân mẫu khác nhau: 'valid' = đủ điều kiện hợp lý của biến đó.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import stats as ST
from ._util import PRODUCT, PROBS_DESC, SUMS, raw_moments, raw_pct, raw_sum

NAME = "s03_descriptive"
ENGINE_BOUND = True
SECTIONS = ["plausibility", "bands", "thresholds"]

ACTION_LABEL = {"1": "Loan originated", "2": "Approved, not accepted", "3": "Application denied",
                "4": "Withdrawn by applicant", "5": "File closed incomplete", "6": "Purchased loan",
                "7": "Preapproval denied", "8": "Preapproval approved, not accepted"}

# (biến, mẫu where, mô tả mẫu)
NUMERIC = [
    ("loan_amount", ["v_loan_amount"], "all records with valid loan_amount"),
    ("income", ["v_income"], "all records with valid income"),
    ("property_value", ["v_property_value"], "all records with valid property_value"),
    ("lti", ["v_lti"], "records with valid income and loan_amount"),
    ("cltv", ["v_cltv"], "all records with valid CLTV"),
    ("loan_term", ["v_loan_term"], "all records with valid loan_term"),
    ("interest_rate", ["is_orig", "v_interest_rate"], "originated loans with valid interest_rate"),
    ("rate_spread", ["is_orig", "v_rate_spread"], "originated loans with valid rate_spread"),
    ("total_loan_costs", ["is_orig", "v_total_loan_costs"], "originated loans with valid total_loan_costs"),
]


def run(ctx):
    # --- 1. kết quả hồ sơ (action_taken)
    a = raw_sum(ctx, "ds_action", ["action_taken"], [])
    total = a["n"].sum()
    a["label"] = a["action_taken"].map(ACTION_LABEL).fillna("other / unexpected code")
    a["share_of_all_records"] = a["n"] / total
    dec = a[a["action_taken"].isin(["1", "2", "3"])]["n"].sum()
    a["in_decided_denominator"] = a["action_taken"].isin(["1", "2", "3"]).astype(int)
    ctx.save_table("ds_outcomes", a)

    ap = raw_sum(ctx, "ds_action_by_product", PRODUCT + ["action_taken"], [])
    piv = ap.pivot_table(index=PRODUCT, columns="action_taken", values="n", aggfunc="sum", fill_value=0).reset_index()
    piv.columns = [f"action_{c}" if c not in PRODUCT else c for c in piv.columns]
    ctx.save_table("ds_action_by_product", piv)

    # --- 2. tóm tắt số: moments + percentile
    summ = []
    for f, where, scope in NUMERIC:
        m = raw_moments(ctx, f"ds_mom_{f}", [], [f], where=where, extremes=True)
        p = raw_pct(ctx, f"ds_pct_{f}", [], f, PROBS_DESC, where=where)
        mm = m.iloc[0]
        s = ST.moments_to_summary(mm["n"], mm[f"sum__{f}"], mm[f"prod__{f}__{f}"])
        row = {"variable": f, "scope": scope, "n": int(mm["n"]), "mean": s["mean"], "std": s["std"],
               "min": mm[f"min__{f}"], "max": mm[f"max__{f}"]}
        for c in p.columns:
            if c.startswith("p"):
                row[c] = p[c].iloc[0]
        row["iqr"] = row["p75"] - row["p25"]
        summ.append(row)
    ctx.save_table("ds_numeric_summary", pd.DataFrame(summ))

    # --- 3. phân vị theo sản phẩm
    for f, where in [("loan_amount", ["v_loan_amount"]), ("income", ["v_income"]), ("lti", ["v_lti"]),
                     ("interest_rate", ["is_orig", "v_interest_rate"])]:
        d = raw_pct(ctx, f"ds_pct_by_product_{f}", PRODUCT, f, PROBS_DESC, where=where)
        ctx.save_table(f"ds_pct_by_product_{f}", d)

    # --- 4. histogram (bin log10 cho loan amount & income; bin 0.25 điểm % cho lãi suất)
    h1 = raw_sum(ctx, "ds_hist_loan_amount", ["loan_amount_lbin"], [], where=["v_loan_amount"])
    h1["bin_low"] = 10 ** (h1["loan_amount_lbin"].astype(int) / 10.0)
    h1["bin_high"] = 10 ** ((h1["loan_amount_lbin"].astype(int) + 1) / 10.0)
    ctx.save_table("ds_hist_loan_amount", h1)
    h2 = raw_sum(ctx, "ds_hist_income", ["income_lbin"], [], where=["v_income"])
    h2["bin_low"] = 10 ** (h2["income_lbin"].astype(int) / 10.0)
    h2["bin_high"] = 10 ** ((h2["income_lbin"].astype(int) + 1) / 10.0)
    ctx.save_table("ds_hist_income", h2)
    h3 = raw_sum(ctx, "ds_hist_rate", ["rate_bin"], [], where=["is_orig", "v_interest_rate"])
    h3["bin_low"] = h3["rate_bin"].astype(int) / 4.0
    h3["bin_high"] = (h3["rate_bin"].astype(int) + 1) / 4.0
    ctx.save_table("ds_hist_rate", h3)
    return {"n_records": int(total), "n_decided": int(dec)}
