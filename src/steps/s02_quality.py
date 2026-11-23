"""Bước 02 - Đánh giá chất lượng dữ liệu: trạng thái theo field, miền phân tích, fingerprint lặp,
lender có giá trị Exempt, ngoại lệ và đối chiếu số dòng theo lender với TS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import schema as S
from ..core import stats as ST
from ._util import PROBS_QA, raw_pct, raw_sum, read_ts, sort_by_order

NAME = "s02_quality"
ENGINE_BOUND = True
SECTIONS = ["paths", "plausibility", "bands", "thresholds"]

NUM_FIELDS = list(S.NUMERIC_RAW) + ["lti"]
STATUS_FIELDS = list(S.NUMERIC_RAW) + ["dti"]
CAT_FIELDS = [c for c in S.CATEGORICAL_RAW if c != "lei"]


def run(ctx):
    cfg = ctx.cfg
    # --- tổng quan cờ
    flags = ["is_app", "is_decided", "is_denied", "is_orig", "is_dup", "has_reported_exempt_value", "is_business",
             "is_open_end", "is_reverse", "is_consumer_cf"] + [f"v_{f}" for f in NUM_FIELDS] + ["v_dti"]
    tot = raw_sum(ctx, "qa_flags_total", [], flags)
    n_total = int(tot["n"].iloc[0])
    rows = [{"flag": f, "n": int(tot[f].iloc[0]), "share_of_records": tot[f].iloc[0] / n_total} for f in flags]
    flag_tab = pd.DataFrame(rows)
    ctx.save_table("qa_flag_summary", flag_tab)

    # --- trạng thái của từng trường số (đếm theo trạng thái)
    st_rows = []
    for f in STATUS_FIELDS:
        d = raw_sum(ctx, f"qa_status_{f}", [f"{f}_status"], [])
        for _, r in d.iterrows():
            st_rows.append({"field": f, "status": r[f"{f}_status"], "n": int(r["n"]), "share": r["n"] / n_total})
    st_tab = pd.DataFrame(st_rows)
    st_tab["_o"] = st_tab["status"].map({s: i for i, s in enumerate(S.STATUS_ORDER)}).fillna(99)
    st_tab = st_tab.sort_values(["field", "_o"]).drop(columns="_o")
    ctx.save_table("qa_numeric_status", st_tab)

    # --- trạng thái theo kết quả hồ sơ: thiếu/miễn có "ghi khác nhau theo outcome" không?
    out_rows = []
    for f in STATUS_FIELDS:
        d = raw_sum(ctx, f"qa_status_action_{f}", [f"{f}_status", "action_taken"], [])
        d = d[d["action_taken"].isin(["1", "2", "3"])].copy()
        d["outcome"] = np.where(d["action_taken"] == "3", "denied", "approved")
        g = d.groupby([f"{f}_status", "outcome"])["n"].sum().unstack(fill_value=0)
        for col in ("approved", "denied"):
            if col not in g:
                g[col] = 0
        tot_o = g[["approved", "denied"]].sum()
        non_value = g.drop(index="VALUE", errors="ignore")[["approved", "denied"]].sum()
        out_rows.append({"field": f,
                         "share_not_VALUE_approved": non_value["approved"] / tot_o["approved"] if tot_o["approved"] else np.nan,
                         "share_not_VALUE_denied": non_value["denied"] / tot_o["denied"] if tot_o["denied"] else np.nan})
    oc = pd.DataFrame(out_rows)
    oc["gap_denied_minus_approved"] = oc["share_not_VALUE_denied"] - oc["share_not_VALUE_approved"]
    oc["recorded_differently_by_outcome"] = (oc["gap_denied_minus_approved"].abs() > 0.05).astype(int)
    ctx.save_table("qa_status_by_outcome", oc)

    # --- mã của các biến phân loại
    code_rows = []
    for c in CAT_FIELDS:
        d = raw_sum(ctx, f"qa_codes_{c}", [c], [])
        for _, r in d.iterrows():
            code_rows.append({"field": c, "code": r[c], "n": int(r["n"]), "share": r["n"] / n_total})
    ctx.save_table("qa_categorical_codes", pd.DataFrame(code_rows))

    # --- fingerprint lặp + record có giá trị Exempt theo hành động
    d = raw_sum(ctx, "qa_dup_exempt_by_action", ["action_taken"], ["is_dup", "has_reported_exempt_value"])
    ctx.save_table("qa_dup_exempt_by_action", d)

    # --- giá trị ngoài miền hợp lý (đã gắn cờ, không xoá)
    imp = []
    for f in list(S.NUMERIC_RAW):
        d = raw_sum(ctx, f"qa_implausible_{f}", [], [f"v_{f}"], where=[(f"{f}_status", "==", "VALUE")])
        n_parsed = int(d["n"].iloc[0])
        imp.append({"field": f, "n_parsed_numeric": n_parsed, "n_in_range": int(d[f"v_{f}"].iloc[0]),
                    "n_out_of_range": n_parsed - int(d[f"v_{f}"].iloc[0]),
                    "plausible_range": str(cfg["plausibility"][f])})
    d = raw_sum(ctx, "qa_implausible_lti", [], ["v_lti"], where=["v_income", "v_loan_amount"])
    imp.append({"field": "lti", "n_parsed_numeric": int(d["n"].iloc[0]), "n_in_range": int(d["v_lti"].iloc[0]),
                "n_out_of_range": int(d["n"].iloc[0]) - int(d["v_lti"].iloc[0]),
                "plausible_range": str(cfg["plausibility"]["lti"])})
    imp_tab = pd.DataFrame(imp)
    ctx.save_table("qa_implausible", imp_tab)

    # --- percentile + ngoại lệ (ngoài hàng rào IQR nhưng vẫn trong miền hợp lý = ngoại lệ thật, giữ)
    out_tab = []
    pct_rows = []
    for f in NUM_FIELDS:
        p = raw_pct(ctx, f"qa_pct_{f}", [], f, PROBS_QA, where=[f"v_{f}"])
        r = p.iloc[0]
        q1, q3 = r["p25"], r["p75"]
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        below = raw_sum(ctx, f"qa_below_{f}", [], [], where=[f"v_{f}", (f, "<", float(lo_f))])
        above = raw_sum(ctx, f"qa_above_{f}", [], [], where=[f"v_{f}", (f, ">", float(hi_f))])
        pct_rows.append({"field": f, **{k: r[k] for k in p.columns}})
        n_valid = int(r["n_valid"])
        out_tab.append({"field": f, "n_valid": n_valid, "p25": q1, "p75": q3, "iqr": iqr, "lower_fence": lo_f,
                        "upper_fence": hi_f, "n_below_fence": int(below["n"].iloc[0]),
                        "n_above_fence": int(above["n"].iloc[0]),
                        "share_beyond_fence": (int(below["n"].iloc[0]) + int(above["n"].iloc[0])) / n_valid if n_valid else np.nan,
                        "rule": "beyond 1.5xIQR but inside analysis domain = outlier, kept; "
                                "outside analysis domain = not usable for this statistic, not asserted to be a reporting error"})
    ctx.save_table("qa_percentiles", pd.DataFrame(pct_rows))
    ctx.save_table("qa_outliers", pd.DataFrame(out_tab))

    # --- đối chiếu theo lender với TS (chỉ để kiểm chứng)
    lc = raw_sum(ctx, "qa_lei_counts", ["lei"], [])
    ts = read_ts(cfg)
    if ts is not None:
        m = lc[["lei", "n"]].merge(ts[["lei", "lar_count", "respondent_name", "agency_code"]], on="lei", how="outer", indicator=True)
        m["n"] = m["n"].fillna(0)
        m["lar_count"] = m["lar_count"].fillna(0)
        m["diff"] = m["n"] - m["lar_count"]
        summ = pd.DataFrame([{
            "lenders_in_LAR": int((lc["n"] > 0).sum()), "institutions_in_TS": int(len(ts)),
            "in_LAR_not_TS": int((m["_merge"] == "left_only").sum()), "in_TS_not_LAR": int((m["_merge"] == "right_only").sum()),
            "lenders_with_count_mismatch": int((m["diff"] != 0).sum()),
            "LAR_rows": int(lc["n"].sum()), "TS_sum_lar_count": int(ts["lar_count"].sum()),
            "totals_match": bool(lc["n"].sum() == ts["lar_count"].sum())}])
        ctx.save_table("qa_ts_reconciliation", summ)
        ctx.save_table("qa_ts_mismatches", m[m["diff"] != 0].drop(columns="_merge"))
    return {"n_total": n_total}
