"""Bước 08 - Kiểm tra độ bền: chạy lại kết quả khi loại
(a) record có ít nhất một giá trị Exempt, (b) lender có ít nhất một record như vậy,
(c) mọi dòng thuộc fingerprint lặp, (d) khoản vay mục đích kinh doanh, (e) tất cả.
Không xoá dữ liệu: mỗi variant chỉ là bộ lọc áp dụng lúc truy vấn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import stats as ST
from ..core.definitions import Q3_VARS
from ._util import FINE, aggregate, raw_pct, raw_sum, to_counts

NAME = "s08_robustness"
ENGINE_BOUND = True
SECTIONS = ["robustness", "thresholds", "bands", "plausibility"]


def _v(cube, band_col, valid):
    g = cube[cube[band_col].isin(valid)].groupby(band_col)[["n_decided", "n_denied"]].sum()
    return ST.cramers_v(np.column_stack([g["n_denied"].values, (g["n_decided"] - g["n_denied"]).values]))["v"]


def run(ctx):
    cfg, defs = ctx.cfg, ctx.defs
    minn = cfg["thresholds"]["lender_min_decided"]
    rows = []
    for vname, items in cfg["robustness"]["variants"].items():
        w = list(items or [])
        cube = to_counts(raw_sum(ctx, f"rb_{vname}_cube_lender", ["lei"] + FINE, where=w))
        tot = aggregate(cube, [])
        nat = tot["n_denied"].iloc[0] / tot["n_decided"].iloc[0]
        lend = aggregate(cube, ["lei"])
        lend["rate"] = lend["n_denied"] / lend["n_decided"]
        el = lend[lend["n_decided"] >= minn]
        d = ST.dispersion(el["rate"].values, el["n_decided"].values)
        row = {"variant": vname, "n_records": int(tot["n_rows"].iloc[0]), "n_decided": int(tot["n_decided"].iloc[0]),
               "national_denial_rate": nat, "lenders_eligible": int(len(el)), "lender_rate_median": d.get("p50"),
               "lender_rate_p5": d.get("p5"), "lender_rate_p95": d.get("p95"), "lender_rate_iqr": d.get("iqr"),
               "lender_rate_std": d.get("std")}
        # tỷ lệ theo loại khoản vay (Q1)
        lt = aggregate(cube, ["loan_type"])
        for r in lt.itertuples():
            row[f"denial_rate_loan_type_{r.loan_type}"] = r.n_denied / r.n_decided if r.n_decided else np.nan
        # Cramér's V (Q3)
        for var in cfg["robustness"]["q3_vars"]:
            band_col = Q3_VARS[var][0]
            c3 = to_counts(raw_sum(ctx, f"rb_{vname}_q3_{var}", [band_col] + FINE, ["is_decided", "is_denied"], where=w))
            row[f"V_{var}"] = _v(c3, band_col, defs.valid_labels(band_col))
        # lãi suất trung vị theo loại khoản vay (Q4)
        p = raw_pct(ctx, f"rb_{vname}_rate_by_type", ["loan_type"], "interest_rate", [0.5],
                    where=w + ["is_orig", "v_interest_rate"])
        for r in p.itertuples():
            row[f"median_interest_rate_loan_type_{r.loan_type}"] = r.p50
        rows.append(row)
    t = pd.DataFrame(rows)
    base = t[t["variant"] == "baseline"]
    if len(base):
        for c in [c for c in t.columns if c not in ("variant",) and t[c].dtype != object]:
            t[f"delta_{c}"] = t[c] - base[c].iloc[0]
    ctx.save_table("robustness_summary", t)
    return {"variants": list(cfg["robustness"]["variants"])}
