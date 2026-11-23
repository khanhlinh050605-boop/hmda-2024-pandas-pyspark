"""Benchmark workload: by lender -> record count, mean(loan_amount), denial rate (n_denied / n_decided).

Second workload "median_by_state": by state_code -> record count + EXACT median of loan_amount (linear interpolation;
Pandas groupby.median, exact Spark percentile(.,0.5), NOT percentile_approx). The string "NA" is a separate group.

The same computation, written separately for each engine. The sampling key `sample_u` (see pandas_engine.sample_key) makes
the 25% / 50% / 100% samples NESTED and identical in both engines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["lei", "loan_amount", "is_decided", "is_denied"]


def pandas_workload(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("lei", observed=True)
    res = g.agg(n_records=("is_decided", "size"), mean_loan_amount=("loan_amount", "mean"),
                n_denied=("is_denied", "sum"), n_decided=("is_decided", "sum"))
    res["denial_rate"] = res["n_denied"] / res["n_decided"]
    return res.reset_index()


def spark_workload(df):
    from pyspark.sql import functions as F

    res = (df.groupBy("lei")
           .agg(F.count(F.lit(1)).alias("n_records"), F.avg("loan_amount").alias("mean_loan_amount"),
                F.sum("is_denied").alias("n_denied"), F.sum("is_decided").alias("n_decided"))
           .withColumn("denial_rate", F.col("n_denied") / F.col("n_decided")))
    return res.collect()          # action: forces Spark to execute


def spark_rows_to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([r.asDict() for r in rows])


def equivalent(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-9, atol=1e-9) -> dict:
    a = a.assign(lei=a["lei"].astype(str)).sort_values("lei").reset_index(drop=True)
    b = b.assign(lei=b["lei"].astype(str)).sort_values("lei").reset_index(drop=True)
    out = {"same_lenders": bool(len(a) == len(b) and (a["lei"].values == b["lei"].values).all())}
    if not out["same_lenders"]:
        out["counts_equal"] = out["floats_close"] = False
        return out
    out["counts_equal"] = bool(all((a[c].astype("int64").values == b[c].astype("int64").values).all()
                                   for c in ("n_records", "n_denied", "n_decided")))
    out["floats_close"] = bool(all(np.allclose(a[c].astype(float), b[c].astype(float), rtol=rtol, atol=atol, equal_nan=True)
                                   for c in ("mean_loan_amount", "denial_rate")))
    return out


# ------------------------------------------------------------------ workload 2: median loan_amount by state_code
STATE_COLS = ["state_code", "loan_amount"]


def pandas_state_workload(df: pd.DataFrame) -> pd.DataFrame:
    res = (df.groupby("state_code", observed=True, dropna=False)["loan_amount"]
           .agg(n_records="size", median_loan_amount="median").reset_index())
    return res.sort_values("state_code", kind="mergesort").reset_index(drop=True)


def spark_state_workload(df):
    from pyspark.sql import functions as F

    res = (df.groupBy("state_code")
           .agg(F.count(F.lit(1)).alias("n_records"),
                F.expr("percentile(cast(loan_amount as double), 0.5)").alias("median_loan_amount"))
           .orderBy("state_code"))
    return res.collect()


def equivalent_state(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-9, atol=1e-9) -> dict:
    """Equivalence gate: state_code must match exactly as strings, n_records must match exactly as integers,
    |median_a - median_b| <= atol + rtol*|median_b|. The keys `same_lenders`/`counts_equal`/`floats_close` are shared with
    workload 1 (here `same_lenders` = same set of state_code groups)."""
    a = a.sort_values("state_code", kind="mergesort").reset_index(drop=True)
    b = b.sort_values("state_code", kind="mergesort").reset_index(drop=True)
    out = {"same_lenders": bool(len(a) == len(b) and (a["state_code"].astype(str).values == b["state_code"].astype(str).values).all())}
    if not out["same_lenders"]:
        out["counts_equal"] = out["floats_close"] = False
        return out
    out["counts_equal"] = bool((a["n_records"].astype("int64").values == b["n_records"].astype("int64").values).all())
    out["floats_close"] = bool(np.allclose(a["median_loan_amount"].astype(float), b["median_loan_amount"].astype(float),
                                           rtol=rtol, atol=atol, equal_nan=True))
    return out


# ------------------------------------------------------------------ workload registry
WORKLOADS = {
    "lender_groupby": {"cols": COLS, "pandas": pandas_workload, "spark": spark_workload, "equiv": equivalent},
    "median_by_state": {"cols": STATE_COLS, "pandas": pandas_state_workload, "spark": spark_state_workload,
                        "equiv": equivalent_state},
}
