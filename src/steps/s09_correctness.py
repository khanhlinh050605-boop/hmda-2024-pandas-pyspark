"""Bước 09 - Đối chiếu Pandas với PySpark (đề bài 3.4C).

So từng bảng thô (kết quả của engine) và từng bảng kết quả cuối (tính tiếp bằng code dùng chung):
  - số nguyên (đếm): phải TRÙNG TUYỆT ĐỐI;
  - số thực: np.isclose với rtol/atol trong config (mặc định 1e-9); không nới dung sai để "cho khớp";
  - chuỗi/khoá: phải trùng tuyệt đối, cùng thứ tự dòng.
Mọi chênh lệch đều được liệt kê (correctness_mismatches.csv) để điều tra, không bị che.
"""
from __future__ import annotations

import json
import hashlib
import re

import numpy as np
import pandas as pd

from ..core.runtime import log

NAME = "s09_correctness"
ENGINE_BOUND = False
SECTIONS = ["correctness"]

INT_RE = re.compile(r"^-?\d+$")
INTEGRAL_TOKEN_RE = re.compile(r"^-?\d+(?:\.0+)?$")
COUNT_COLUMN_RE = re.compile(r"^(?:n|count|n_.+|.+_count|.+_counts|rows|records|lenders)$")
KEY_COLUMNS = {
    "lei", "state_code", "loan_type", "loan_purpose", "lien_status", "action_taken", "segment", "product",
    "variable", "band", "status", "status_band", "scope", "variant", "term", "target", "model", "split",
    "income_band", "loan_amount_band", "lti_band", "dti_band", "cltv_band", "property_value_band",
}


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _all_int_strings(s: pd.Series) -> bool:
    v = s[s != ""]
    return bool(len(v) == 0 or v.map(lambda x: bool(INT_RE.fullmatch(str(x)))).all())


def _all_integral_count_tokens(s: pd.Series, column: str) -> bool:
    """Recognise count columns even when a CSV writer appends a harmless .0."""
    if not COUNT_COLUMN_RE.fullmatch(column):
        return False
    v = s[s != ""]
    return bool(len(v) == 0 or v.map(lambda x: bool(INTEGRAL_TOKEN_RE.fullmatch(str(x)))).all())


def _int_token(x: str):
    if x == "":
        return None
    return int(str(x).split(".", 1)[0])


def _catalog_signature(names) -> dict:
    names = sorted(str(n).removesuffix(".csv") for n in names)
    digest = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    return {"count": len(names), "sha256": digest}


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, rtol: float, atol: float, name: str,
                   key_columns: list[str] | set[str] | None = None,
                   column_tolerances: dict[str, dict[str, float]] | None = None):
    mism = []
    if list(a.columns) != list(b.columns):
        return {"agree": False, "note": f"columns differ: {set(a.columns) ^ set(b.columns)}"}, mism
    if len(a) != len(b):
        return {"agree": False, "note": f"row count differs: pandas={len(a)} spark={len(b)}"}, mism
    n_num = n_int = n_str = 0
    max_abs = max_rel = 0.0
    n_bad = 0
    keys = set(key_columns or [])
    column_tolerances = column_tolerances or {}
    for c in a.columns:
        sa, sb = a[c].astype(str), b[c].astype(str)
        if c in keys:
            n_str += len(a)
            bad = sa.to_numpy() != sb.to_numpy()
            diff = rel = np.zeros(len(a))
        elif (_all_int_strings(sa) and _all_int_strings(sb)) or (
                _all_integral_count_tokens(sa, c) and _all_integral_count_tokens(sb, c)):
            n_int += len(a)
            va = np.array([_int_token(x) for x in sa], dtype=object)
            vb = np.array([_int_token(x) for x in sb], dtype=object)
            bad = va != vb
            diff = rel = np.zeros(len(a))
        else:
            try:
                va = pd.to_numeric(sa.replace("", np.nan), errors="raise").to_numpy(float)
                vb = pd.to_numeric(sb.replace("", np.nan), errors="raise").to_numpy(float)
                numeric = True
            except (ValueError, TypeError):
                numeric = False
            if numeric:
                ctol = column_tolerances.get(c, {})
                crtol = float(ctol.get("rtol", rtol))
                catol = float(ctol.get("atol", atol))
                n_num += len(a)
                both_nan = np.isnan(va) & np.isnan(vb)
                one_nan = np.isnan(va) ^ np.isnan(vb)
                bad = one_nan | (~both_nan & ~np.isclose(va, vb, rtol=crtol, atol=catol, equal_nan=False))
                diff = np.abs(va - vb)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(np.abs(va) > 0, diff / np.abs(va), 0.0)
                diff = np.where(both_nan, 0.0, diff)
                rel = np.where(both_nan, 0.0, rel)
            else:
                n_str += len(a)
                bad = sa.to_numpy() != sb.to_numpy()
                diff = rel = np.zeros(len(a))
        if len(diff):
            finite_diff = diff[np.isfinite(diff)]
            finite_rel = rel[np.isfinite(rel)]
            if len(finite_diff):
                max_abs = max(max_abs, float(finite_diff.max()))
            if len(finite_rel):
                max_rel = max(max_rel, float(finite_rel.max()))
        for i in np.where(bad)[0][:25]:
            mism.append({"table": name, "row": int(i), "column": c, "pandas": a[c].iloc[i], "spark": b[c].iloc[i]})
        n_bad += int(np.asarray(bad).sum())
    return {"agree": n_bad == 0, "n_rows": len(a), "n_cols": a.shape[1], "cells_integer_exact": n_int,
            "cells_float_tolerance": n_num, "cells_text_exact": n_str, "max_abs_diff": max_abs,
            "max_rel_diff": max_rel, "n_mismatch_cells": n_bad, "note": ""}, mism


def _num(df, col, row=0):
    try:
        return float(df[col].iloc[row])
    except Exception:  # noqa: BLE001
        return np.nan


def validation_passes(detail: pd.DataFrame, headline: pd.DataFrame) -> bool:
    """A non-empty complete catalog and a non-empty headline set must both pass."""
    return bool(len(detail) and len(headline) and detail["agree"].astype(bool).all() and
                headline["agree"].astype(bool).all())


def run(ctx):
    cfg = ctx.cfg
    tol = cfg["correctness"]
    pd_dir, sp_dir = cfg.engine_out("pandas"), cfg.engine_out("spark")
    rows, mism_all = [], []
    for kind in ("raw", "tables"):
        a_files = {p.name: p for p in (pd_dir / kind).glob("*.csv")}
        b_files = {p.name: p for p in (sp_dir / kind).glob("*.csv")}
        expected_catalog = tol.get("expected_catalog", {}).get(kind)
        if expected_catalog:
            for engine, files in (("pandas", a_files), ("spark", b_files)):
                actual = _catalog_signature(files)
                if actual != expected_catalog:
                    rows.append({"kind": kind, "table": f"<catalog:{engine}>", "agree": False,
                                 "note": f"catalog mismatch: expected {expected_catalog}, actual {actual}"})
        required = {f"{n}.csv" for n in tol.get("required_tables", {}).get(kind, [])}
        for name in sorted(set(a_files) | set(b_files) | required):
            if name not in a_files or name not in b_files:
                if name not in a_files and name not in b_files:
                    note = "required table missing from both engines"
                else:
                    note = "only in " + ("pandas" if name in a_files else "spark")
                rows.append({"kind": kind, "table": name[:-4], "agree": False,
                             "note": note})
                continue
            ov = tol.get("overrides", {}).get(name[:-4], {})
            keys = set(KEY_COLUMNS)
            if kind == "raw":
                metas = []
                for d in (pd_dir, sp_dir):
                    mp = d / kind / f"{name[:-4]}.meta.json"
                    if mp.exists():
                        metas.extend(json.loads(mp.read_text()).get("keys", []))
                keys.update(metas)
            res, mm = compare_tables(_read(a_files[name]), _read(b_files[name]),
                                     ov.get("rtol", tol["rtol"]), ov.get("atol", tol["atol"]), name[:-4], keys,
                                     ov.get("columns", {}))
            rows.append({"kind": kind, "table": name[:-4], "rtol": ov.get("rtol", tol["rtol"]),
                         "atol": ov.get("atol", tol["atol"]),
                         "column_overrides": json.dumps(ov.get("columns", {}), sort_keys=True), **res})
            mism_all += mm
    det = pd.DataFrame(rows)
    if det.empty:
        det = pd.DataFrame([{"kind": "catalog", "table": "<none>", "agree": False,
                             "note": "no output tables found for comparison"}])
    ctx.save_table("correctness_detail", det)
    ctx.save_table("correctness_mismatches", pd.DataFrame(mism_all, columns=["table", "row", "column", "pandas", "spark"]))

    # --- bảng đối chiếu theo mẫu đề bài: đại lượng | Pandas | PySpark | khớp? | giải thích
    def tab(engine, name):
        p = cfg.engine_out(engine) / "tables" / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def meta(engine):
        p = cfg.interim_dir / f"{engine}_info" / "_meta.json"
        return json.loads(p.read_text()) if p.exists() else {}

    items = []

    def add(item_id, q, a, b, explain=""):
        is_int = isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))
        finite = is_int or (np.isfinite(a) and np.isfinite(b))
        agree = bool(finite and ((a == b) if is_int else np.isclose(a, b, rtol=tol["rtol"], atol=tol["atol"])))
        diff = (int(a) - int(b)) if is_int else (float(a) - float(b) if finite else np.nan)
        tolerance = "exact" if is_int else f"rtol={tol['rtol']}, atol={tol['atol']}"
        items.append({"id": item_id, "quantity": q, "pandas": a, "pyspark": b, "difference": diff,
                      "tolerance": tolerance, "conclusion": "PASS" if agree else "FAIL", "agree": agree,
                      "explanation": explain or ("exact integer match" if is_int else f"within {tolerance}")})

    ma, mb = meta("pandas"), meta("spark")
    for k in ("n_rows", "n_dup_rows", "n_excess_dup_rows", "lenders_with_exempt_values"):
        if k in ma and k in mb:
            add(f"master_{k}", f"master: {k}", int(ma[k]), int(mb[k]))
    oa, ob = tab("pandas", "ds_outcomes"), tab("spark", "ds_outcomes")
    if len(oa) and len(ob):
        add("records_after_preparation", "records after preparation (all action_taken)", int(oa["n"].sum()), int(ob["n"].sum()))
        for code, label in (("1", "originated"), ("3", "denied")):
            add(f"action_{code}", f"records with action_taken={code} ({label})", int(oa.loc[oa["action_taken"].astype(str) == code, "n"].sum()),
                int(ob.loc[ob["action_taken"].astype(str) == code, "n"].sum()))
    na, nb = tab("pandas", "q1_national_total"), tab("spark", "q1_national_total")
    if len(na) and len(nb):
        add("decided_applications", "decided applications (action 1-3)", int(na["n_decided"].iloc[0]), int(nb["n_decided"].iloc[0]))
        add("national_denial_rate", "national denial rate among decided", _num(na, "denial_rate"), _num(nb, "denial_rate"))
        add("national_origination_share", "national origination share among applications (1-5)", _num(na, "orig_share"), _num(nb, "orig_share"))
    sa, sb = tab("pandas", "ds_numeric_summary"), tab("spark", "ds_numeric_summary")
    for v in ("loan_amount", "income", "interest_rate"):
        if len(sa) and len(sb) and v in set(sa["variable"]):
            ra, rb = sa[sa["variable"] == v].iloc[0], sb[sb["variable"] == v].iloc[0]
            add(f"{v}_mean", f"{v}: mean", float(ra["mean"]), float(rb["mean"]))
            add(f"{v}_median", f"{v}: median (exact percentile)", float(ra["p50"]), float(rb["p50"]))
    la, lb = tab("pandas", "q2_lender_table"), tab("spark", "q2_lender_table")
    if len(la) and len(lb):
        add("number_of_lenders", "number of lenders", int(len(la)), int(len(lb)))
        add("eligible_lenders", "eligible lenders (>= min decided)", int(la["eligible"].sum()), int(lb["eligible"].sum()))
        add("lender_denial_count", "sum of lender denial counts", int(la["n_denied"].sum()), int(lb["n_denied"].sum()))
    ca, cb = tab("pandas", "q1_state_consistency_summary"), tab("spark", "q1_state_consistency_summary")
    if len(ca) and len(cb):
        add("states_consistent", "states consistent with national product pattern", int(ca["states_consistent"].iloc[0]), int(cb["states_consistent"].iloc[0]))
    aa, ab = tab("pandas", "q3_association"), tab("spark", "q3_association")
    if len(aa) and len(ab):
        for v in aa["variable"]:
            add(f"cramers_v_{v}", f"Cramér's V ({v})", float(aa.loc[aa["variable"] == v, "V_bias_corrected"].iloc[0]),
                float(ab.loc[ab["variable"] == v, "V_bias_corrected"].iloc[0]))
    ra_, rb_ = tab("pandas", "q4_reg_coef_interest_rate"), tab("spark", "q4_reg_coef_interest_rate")
    if len(ra_) and len(rb_):
        for t in ra_["term"].head(3):
            add(f"q4_coef_{t}", f"regression (interest_rate) coefficient: {t}", float(ra_.loc[ra_["term"] == t, "coef"].iloc[0]),
                float(rb_.loc[rb_["term"] == t, "coef"].iloc[0]))
        add("q4_interest_rate_r2", "regression (interest_rate) R-squared", float(tab("pandas", "q4_reg_summary")["r2"].iloc[0]),
            float(tab("spark", "q4_reg_summary")["r2"].iloc[0]))
    present = {x["id"] for x in items}
    for item_id in tol.get("required_headlines", []):
        if item_id not in present:
            items.append({"id": item_id, "quantity": item_id, "pandas": np.nan, "pyspark": np.nan,
                          "difference": np.nan, "tolerance": "required", "conclusion": "MISSING",
                          "agree": False, "explanation": "required headline missing from one or both engines"})
    head = pd.DataFrame(items)
    expected_headlines = tol.get("expected_headline_count")
    if expected_headlines is not None and len(head) != int(expected_headlines):
        head = pd.concat([head, pd.DataFrame([{
            "id": "<headline_catalog>", "quantity": "headline catalog size", "pandas": len(items),
            "pyspark": int(expected_headlines), "difference": len(items) - int(expected_headlines),
            "tolerance": "exact", "conclusion": "FAIL", "agree": False,
            "explanation": f"expected {int(expected_headlines)} headline quantities; found {len(items)}",
        }])], ignore_index=True)
    ctx.save_table("correctness_headline", head)

    n_tab = len(det)
    n_ok = int(det["agree"].sum())
    log.info("[s09] %d/%d bảng khớp (%d ô lệch). Headline: %d/%d đại lượng khớp", n_ok, n_tab,
             int(len(mism_all)), int(head["agree"].sum()) if len(head) else 0, len(head))
    n_head_ok = int(head["agree"].sum()) if len(head) else 0
    all_ok = validation_passes(det, head)
    summary = {"tables_compared": n_tab, "tables_agree": n_ok, "all_agree": all_ok,
               "headline_items": int(len(head)), "headline_agree": n_head_ok}
    (ctx.out / "correctness_summary.json").write_text(json.dumps(summary, indent=2))
    if not all_ok:
        raise RuntimeError(f"Correctness failed: tables {n_ok}/{n_tab}, headlines {n_head_ok}/{len(head)}. "
                           "See correctness_detail.csv, correctness_mismatches.csv and correctness_headline.csv.")
    return summary
