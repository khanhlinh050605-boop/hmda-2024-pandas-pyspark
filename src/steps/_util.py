"""Hàm tiện ích dùng chung cho các bước: gọi engine + lưu bảng thô, đổi tên cột, bảng tỷ lệ."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from ..core import schema as S
from ..core import stats as ST

SUMS = ["is_app", "is_decided", "is_denied", "is_orig"]
FINE = ["loan_type", "loan_purpose", "lien_status"]
PRODUCT = ["loan_type", "loan_purpose"]
NA_STATE = "NA"
RENAME = {"n": "n_rows", "is_app": "n_apps", "is_decided": "n_decided", "is_denied": "n_denied", "is_orig": "n_orig"}

PROBS_DESC = [0.05, 0.25, 0.5, 0.75, 0.95]
PROBS_QA = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


# ------------------------------------------------------------------ gọi engine + lưu bảng thô
def raw_sum(ctx, name, keys, sums=SUMS, where=()):
    df = ctx.engine.group_sum(keys, sums, ctx.where(where))
    ctx.save_raw(name, df, list(keys))
    return df


def raw_moments(ctx, name, keys, cols, where=(), cross=False, extremes=False):
    df = ctx.engine.group_moments(keys, cols, ctx.where(where), cross=cross, extremes=extremes)
    ctx.save_raw(name, df, list(keys))
    return df


def raw_pct(ctx, name, keys, col, probs, where=()):
    df = ctx.engine.group_percentiles(keys, col, probs, ctx.where(where))
    ctx.save_raw(name, df, list(keys))
    return df


# ------------------------------------------------------------------ bảng tỷ lệ
def to_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME)


def with_rates(df: pd.DataFrame, small: int, orig: bool = True) -> pd.DataFrame:
    """Bảng đếm -> thêm denial rate (trên decided) + origination share (trên apps 1-5), kèm Wilson 95%."""
    df = df.copy()
    ST.add_rate_ci(df, "n_denied", "n_decided", prefix="denial_", small=small)
    df = df.rename(columns={"denial_rate": "denial_rate", "denial_small_cell": "small_cell"})
    if orig and "n_orig" in df and "n_apps" in df:
        p, lo, hi = ST.wilson(df["n_orig"].values, df["n_apps"].values)
        df["orig_share"], df["orig_ci_lo"], df["orig_ci_hi"] = p, lo, hi
    return df


def aggregate(df: pd.DataFrame, by: list[str], cols=("n_rows", "n_apps", "n_decided", "n_denied", "n_orig")) -> pd.DataFrame:
    cols = [c for c in cols if c in df]
    if not by:
        return df[cols].sum().to_frame().T
    return df.groupby(by, as_index=False, sort=True)[cols].sum()


def reported_states(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["state_code"] != NA_STATE) & (df["state_code"] != S.MISSING)]


def seg_label(df: pd.DataFrame) -> pd.Series:
    return df["loan_type"].astype(str) + S.SEP + df["loan_purpose"].astype(str) + S.SEP + df["lien_status"].astype(str)


def top_segments(national_seg: pd.DataFrame, k: int) -> list[str]:
    t = national_seg.copy()
    t["segment"] = seg_label(t)
    return t.sort_values(["n_decided", "segment"], ascending=[False, True])["segment"].head(k).tolist()


# ------------------------------------------------------------------ Transmittal Sheet (chỉ đối chiếu / nhãn)
def read_ts(cfg) -> pd.DataFrame | None:
    p = cfg.path("ts_file")
    if not p.exists():
        return None
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            name = [n for n in z.namelist() if not n.endswith("/")][0]
            raw = z.read(name).decode("utf-8", errors="replace")
        ts = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, keep_default_na=False)
    else:
        ts = pd.read_csv(p, sep="|", dtype=str, keep_default_na=False)
    ts.columns = [c.strip().lower() for c in ts.columns]
    ts["lar_count"] = pd.to_numeric(ts["lar_count"], errors="coerce")
    return ts


def sort_by_order(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    rank = {v: i for i, v in enumerate(order)}
    df = df.copy()
    df["_o"] = df[col].map(rank).fillna(len(order))
    return df.sort_values("_o", kind="mergesort").drop(columns="_o").reset_index(drop=True)
