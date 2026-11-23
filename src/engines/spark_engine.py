"""Engine PySpark: prepare + các truy vấn gộp trên master Parquet. Cài đặt độc lập với Pandas.

Dùng select / withColumn / filter / groupBy / agg / join / window (row_number) của Spark DataFrame.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import schema as S
from ..core.definitions import DTI_TEXT_MAP, Defs
from ..core.runtime import log
from .base import Engine, pairs_of, pname
from .pandas_engine import MOD, dup_columns, list_lar_files


def collect_pandas(df) -> pd.DataFrame:
    """Collect small aggregate results without PySpark's legacy distutils gate."""
    return pd.DataFrame([r.asDict(recursive=True) for r in df.collect()], columns=df.columns)


def get_spark(cfg, app: str = "hmda_g5"):
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    sp = cfg["spark"]
    b = (SparkSession.builder.appName(app).master(sp["master"])
         .config("spark.driver.memory", sp["driver_memory"])
         .config("spark.sql.shuffle.partitions", str(sp["shuffle_partitions"]))
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.ansi.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.parquet.compression.codec", "snappy"))
    if sp.get("local_dir"):
        b = b.config("spark.local.dir", str(sp["local_dir"]))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _as_cond(F, where):
    cond = None
    for c, op, v in where:
        col = F.col(f"`{c}`")
        e = {"==": lambda: col == v, "!=": lambda: col != v, "in": lambda: col.isin(list(v)),
             "not_in": lambda: ~col.isin(list(v)), ">=": lambda: col >= v, "<=": lambda: col <= v,
             ">": lambda: col > v, "<": lambda: col < v}[op]()
        cond = e if cond is None else (cond & e)
    return cond


# ------------------------------------------------------------------ biểu thức làm sạch
def num_status(F, name: str, field: str):
    s = F.regexp_replace(F.col(name), r"^\s+|\s+$", "")
    up = F.upper(s)
    status = F.when(s.isNull() | (s == ""), "EMPTY")
    for token, label in S.FIELD_SPECIAL_VALUES[field].items():
        status = status.when(up == token, label)
    return status.when(s.rlike(S.NUM_RE), "VALUE").otherwise("UNPARSABLE"), s


def range_cond(F, col, spec: dict):
    lo, hi = float(spec["lo"]), float(spec["hi"])
    lo_c = (col >= lo) if spec.get("lo_inclusive", False) else (col > lo)
    return lo_c & (col <= hi)


def band_expr(F, val, edges, labels):
    e = F.when(val < float(edges[0]), labels[0])
    for i in range(1, len(edges)):
        e = e.when(val < float(edges[i]), labels[i])
    return e.otherwise(labels[-1])


def band_with_status(F, val, valid, status, edges, labels):
    return (F.when(valid, band_expr(F, val, edges, labels))
            .when(status != "VALUE", F.concat(F.lit("~"), status))
            .otherwise(F.lit("~OUTSIDE_DOMAIN")))


def dti_numeric_band_expr(F, val):
    """Public HMDA DTI bins; unlike other cut points, 60 belongs to 50-60."""
    return (F.when(val < 20.0, "<20")
            .when(val < 30.0, "20-<30")
            .when(val < 36.0, "30-<36")
            .when(val < 43.0, "36-<43")
            .when(val < 50.0, "43-<50")
            .when(val <= 60.0, "50-60")
            .otherwise(">60"))


def build_master(F, df, cfg, defs: Defs, regex: str):
    """df: cột đã chọn, tên chuẩn, kiểu chuỗi, cộng `_file` và `row_hash`."""
    plaus = cfg["plausibility"]
    seed = int(cfg["project"]["seed"]) % MOD
    cols = {}
    d = df
    for c in S.CATEGORICAL_RAW:
        s = F.regexp_replace(F.col(c), r"^\s+|\s+$", "")
        d = d.withColumn(c, F.when(s.isNull() | (s == ""), S.MISSING).otherwise(s))
    num_valid = {}
    for c in S.NUMERIC_RAW:
        st, s = num_status(F, c, c)
        d = d.withColumn(f"{c}_status", st).withColumn(c, F.when(F.col(f"{c}_status") == "VALUE", s.cast("double")))
        d = d.withColumn(f"v_{c}", F.when(F.col(f"{c}_status") == "VALUE", range_cond(F, F.col(c), plaus[c])).otherwise(False))
        num_valid[c] = F.col(f"v_{c}")
    # DTI
    st, s = num_status(F, S.DTI_RAW, "dti")
    t = F.regexp_replace(s, r"[% ]", "")
    is_text = t.isin(list(DTI_TEXT_MAP))
    d = d.withColumn("_dti_raw_status", st).withColumn("_dti_text", is_text).withColumn("_dti_t", t)
    d = d.withColumn("_dti_val", F.when(F.col("_dti_raw_status") == "VALUE", s.cast("double")))
    d = d.withColumn("dti_status", F.when(F.col("_dti_text"), "VALUE").otherwise(F.col("_dti_raw_status")))
    d = d.withColumn("_dti_valid_numeric", (F.col("dti_status") == "VALUE") &
                     range_cond(F, F.col("_dti_val"), plaus["dti"]))
    text_map = F.create_map(*[x for kv in DTI_TEXT_MAP.items() for x in (F.lit(kv[0]), F.lit(kv[1]))])
    d = d.withColumn("dti_band", F.when(F.col("dti_status") != "VALUE", F.concat(F.lit("~"), F.col("dti_status")))
                     .when(F.col("_dti_text"), text_map[F.col("_dti_t")])
                     .when(F.col("_dti_valid_numeric"), dti_numeric_band_expr(F, F.col("_dti_val")))
                     .otherwise(F.lit("~OUTSIDE_DOMAIN")))
    d = d.withColumn("v_dti", F.col("_dti_text") | F.col("_dti_valid_numeric"))
    # LTI
    ok = F.col("v_income") & F.col("v_loan_amount")
    lti_raw = F.when(ok, F.col("loan_amount") / (F.col("income") * F.lit(1000.0)))
    d = d.withColumn("_lti_raw", lti_raw)
    d = d.withColumn("v_lti", ok & range_cond(F, F.col("_lti_raw"), plaus["lti"]))
    d = d.withColumn("lti", F.when(F.col("v_lti"), F.col("_lti_raw")))
    lti_status = (F.when(F.col("income_status") != "VALUE", F.col("income_status"))
                  .when(F.col("loan_amount_status") != "VALUE", F.col("loan_amount_status"))
                  .otherwise(F.lit("VALUE")))
    # band
    d = d.withColumn("income_band", band_with_status(F, F.col("income"), F.col("v_income"), F.col("income_status"),
                                                     defs.edges["income_band"], defs.labels["income_band"]))
    d = d.withColumn("loan_amount_band", band_with_status(F, F.col("loan_amount"), F.col("v_loan_amount"),
                                                          F.col("loan_amount_status"), defs.edges["loan_amount_band"],
                                                          defs.labels["loan_amount_band"]))
    d = d.withColumn("lti_band", band_with_status(F, F.col("lti"), F.col("v_lti"), lti_status,
                                                  defs.edges["lti_band"], defs.labels["lti_band"]))
    d = d.withColumn("cltv_band", band_with_status(F, F.col("cltv"), F.col("v_cltv"), F.col("cltv_status"),
                                                   defs.edges["cltv_band"], defs.labels["cltv_band"]))
    d = d.withColumn("property_value_band", band_with_status(F, F.col("property_value"), F.col("v_property_value"),
                                                             F.col("property_value_status"),
                                                             defs.edges["property_value_band"],
                                                             defs.labels["property_value_band"]))
    # log, năm, bin
    d = d.withColumn("log_loan_amount", F.when(F.col("v_loan_amount"), F.log(F.col("loan_amount"))))
    d = d.withColumn("log_income", F.when(F.col("v_income"), F.log(F.col("income"))))
    d = d.withColumn("log_property_value", F.when(F.col("v_property_value"), F.log(F.col("property_value"))))
    d = d.withColumn("term_years", F.when(F.col("v_loan_term"), F.col("loan_term") / F.lit(12.0)))
    d = d.withColumn("loan_amount_lbin", F.when(F.col("v_loan_amount"), F.floor(F.log10(F.col("loan_amount")) * 10.0))
                     .otherwise(F.lit(-9999)).cast("short"))
    d = d.withColumn("income_lbin", F.when(F.col("v_income"), F.floor(F.log10(F.col("income")) * 10.0))
                     .otherwise(F.lit(-9999)).cast("short"))
    d = d.withColumn("rate_bin", F.when(F.col("v_interest_rate"), F.floor(F.col("interest_rate") * 4.0))
                     .otherwise(F.lit(-9999)).cast("short"))
    sep = S.SEP
    d = d.withColumn("product", F.concat_ws(sep, F.col("loan_type"), F.col("loan_purpose")))
    d = d.withColumn("segment", F.concat_ws(sep, F.col("product"), F.col("lien_status")))
    a = F.col("action_taken")
    d = d.withColumn("is_app", a.isin("1", "2", "3", "4", "5")).withColumn("is_decided", a.isin("1", "2", "3"))
    d = d.withColumn("is_denied", a == "3").withColumn("is_orig", a == "1")
    b, o, r = (F.col(c) for c in ("business_or_commercial_purpose", "open_end_line_of_credit", "reverse_mortgage"))
    d = d.withColumn("is_business", b == "1").withColumn("is_open_end", o == "1").withColumn("is_reverse", r == "1")
    d = d.withColumn("is_consumer_cf", (b == "2") & (o == "2") & (r == "2"))
    ex = (b == "1111") | (o == "1111") | (r == "1111") | (F.col("dti_status") == "EXEMPT")
    for c in ("income", "property_value", "loan_term", "interest_rate", "rate_spread", "total_loan_costs", "cltv"):
        ex = ex | (F.col(f"{c}_status") == "EXEMPT")
    d = d.withColumn("has_reported_exempt_value", ex)
    base = (a == "1") & F.col("is_consumer_cf") & F.col("v_loan_amount") & F.col("v_income") & F.col("v_cltv") \
        & F.col("v_loan_term") & F.col("v_property_value")
    d = d.withColumn("q4_reg_rate", base & F.col("v_interest_rate")).withColumn("q4_reg_spread", base & F.col("v_rate_spread"))
    # Stable local key without a global/window sort.  Benchmarking reads the
    # canonical Pandas master for both engines, so analytical Spark preparation
    # does not need to reproduce Pandas row positions and must not spill a
    # 12.3-million-row Window sort to local disk.
    d = d.withColumn("part_no", F.regexp_extract(F.col("_file"), regex, 1).cast("int"))
    d = d.withColumn("row_idx", F.monotonically_increasing_id().cast("long"))
    i = F.col("part_no").cast("long") * F.lit(10_000_000) + F.col("row_idx").cast("long")
    h = (i * F.lit(48271) + F.lit(seed)) % F.lit(MOD)
    h2 = (h * h + F.lit(1013904223)) % F.lit(MOD)
    h3 = (h2 * h2 + F.lit(12345)) % F.lit(MOD)
    d = d.withColumn("sample_u", h3.cast("double") / F.lit(float(MOD)))
    return d


FLAGS = ["is_app", "is_decided", "is_denied", "is_orig", "is_business", "is_open_end", "is_reverse",
         "is_consumer_cf", "has_reported_exempt_value", "is_dup", "q4_reg_rate", "q4_reg_spread", "v_dti", "v_lti"] + \
        [f"v_{c}" for c in S.NUMERIC_RAW]


class SparkEngine(Engine):
    name = "spark"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.spark = None
        self.df = None

    def _spark(self):
        if self.spark is None:
            self.spark = get_spark(self.cfg)
        return self.spark

    # ---------------------------------------------------------------- prepare
    def prepare(self, ctx) -> dict:
        from pyspark.sql import functions as F

        cfg, defs = self.cfg, ctx.defs
        t0 = time.time()
        spark = self._spark()
        src = list_lar_files(cfg)
        paths = [p.as_posix() for _, p in src]
        raw = spark.read.parquet(*paths)
        colmap = S.resolve_columns(raw.columns)
        dcols = dup_columns(raw.columns)
        S.assert_no_forbidden(dcols)
        inv = {v: k for k, v in colmap.items()}
        sel = [F.col(f"`{c}`").cast("string").alias(inv.get(c, c)) for c in dcols]
        d = raw.select(*sel, F.input_file_name().alias("_file"))
        names = [inv.get(c, c) for c in dcols]
        hash_cols = [F.coalesce(F.col(f"`{c}`"), F.lit("\x01NULL")) for c in names]
        d = d.withColumn("row_hash", F.xxhash64(*hash_cols))
        d = d.withColumnRenamed(S.NUMERIC_RAW["cltv"], "cltv")

        # Materialise only the small repeated-hash table.  Persisting the full
        # enriched frame with DISK_ONLY is much larger than compressed Parquet
        # and can exhaust a single-machine scratch disk before the write.  Two
        # source scans are deliberate here: one for exact duplicate counts and
        # one for the analytical master.
        dup_path = cfg.interim_dir / "_spark_duplicate_hashes"
        if dup_path.exists():
            shutil.rmtree(dup_path)
        (d.groupBy("row_hash").count().filter(F.col("count") > 1)
         .write.mode("overwrite").parquet(dup_path.as_posix()))
        dup_counts = spark.read.parquet(dup_path.as_posix())
        dup_stats = dup_counts.agg(
            F.sum("count").alias("n_dup_rows"),
            F.sum(F.col("count") - 1).alias("n_excess_dup_rows"),
        ).first()
        n_dup = int(dup_stats["n_dup_rows"] or 0)
        excess = int(dup_stats["n_excess_dup_rows"] or 0)
        dup_h = F.broadcast(dup_counts.select("row_hash", F.lit(1).alias("_dup")))

        m = build_master(F, d, cfg, defs, cfg["paths"]["part_regex"])
        m = m.join(dup_h, on="row_hash", how="left").withColumn("is_dup", F.col("_dup").isNotNull())
        keep = (S.CATEGORICAL_RAW + ["product", "segment", "dti_band", "income_band", "loan_amount_band", "lti_band",
                                     "cltv_band", "property_value_band"]
                + [f"{c}_status" for c in S.NUMERIC_RAW] + ["dti_status"]
                + list(S.NUMERIC_RAW) + ["lti", "log_loan_amount", "log_income", "log_property_value", "term_years",
                                         "sample_u", "loan_amount_lbin", "income_lbin", "rate_bin"]
                + FLAGS + ["part_no", "row_idx"])
        keep = list(dict.fromkeys(keep))
        out = m.select(*[(F.col(c).cast("byte") if c in FLAGS else F.col(c)) for c in keep])
        master = ctx.master_dir()
        if master.exists():
            shutil.rmtree(master)
        master.parent.mkdir(parents=True, exist_ok=True)
        # Reduce the 277 input partitions without a full-row shuffle.  The
        # broadcast duplicate join preserves the source partitioning, so a
        # narrow coalesce is sufficient for the requested file count.
        out.coalesce(int(cfg["spark"]["master_files"])).write.mode("overwrite").parquet(master.as_posix())
        # thống kê sau khi ghi
        mm = spark.read.parquet(master.as_posix())
        n_rows = mm.count()
        ex = collect_pandas(mm.filter(F.col("has_reported_exempt_value") == 1).select("lei").distinct()).sort_values("lei")
        ex.to_csv(ctx.info_dir() / "lenders_with_exempt_values.csv", index=False)
        meta = {"engine": "spark", "n_files": len(src), "n_rows": int(n_rows), "n_dup_rows": int(n_dup),
                "n_excess_dup_rows": excess, "dup_columns_n": len(dcols),
                "lenders_with_exempt_values": int(len(ex)),
                "seconds": round(time.time() - t0, 1), "pyspark": spark.version}
        (ctx.info_dir() / "_meta.json").write_text(json.dumps(meta, indent=2))
        shutil.rmtree(dup_path, ignore_errors=True)
        return meta

    # ---------------------------------------------------------------- truy vấn
    def open(self, ctx) -> None:
        d = ctx.master_dir()
        if not d.exists():
            raise FileNotFoundError(f"Chưa có master Spark ở {d}. Chạy bước 01 trước.")
        self.df = self._spark().read.parquet(d.as_posix())
        self._ctx = ctx

    def close(self) -> None:
        if self.spark is not None:
            self.spark.stop()
            self.spark = None

    def master_columns(self) -> list[str]:
        return list(self.df.columns)

    def _filtered(self, where, extra=None):
        from pyspark.sql import functions as F

        df = self.df
        cond = _as_cond(F, where)
        if cond is not None:
            df = df.filter(cond)
        return df

    def _group_sum(self, keys, sums, where):
        from pyspark.sql import functions as F

        df = self._filtered(where)
        aggs = [F.count(F.lit(1)).alias("n")] + [F.sum(F.col(s).cast("long")).alias(s) for s in sums]
        g = df.groupBy(*[F.col(k) for k in keys]).agg(*aggs) if keys else df.agg(*aggs)
        pdf = collect_pandas(g)
        for s in sums:
            pdf[s] = pdf[s].fillna(0).astype("int64")
        return pdf

    def _group_moments(self, keys, cols, where, cross, extremes):
        from pyspark.sql import functions as F

        df = self._filtered(where)
        for c in cols:
            df = df.filter(F.col(c).isNotNull() & ~F.isnan(F.col(c)))
        aggs = [F.count(F.lit(1)).alias("n")]
        for c in cols:
            aggs.append(F.sum(F.col(c)).alias(f"sum__{c}"))
            if extremes:
                aggs += [F.min(F.col(c)).alias(f"min__{c}"), F.max(F.col(c)).alias(f"max__{c}")]
        for a, b in pairs_of(cols, cross):
            aggs.append(F.sum(F.col(a) * F.col(b)).alias(f"prod__{a}__{b}"))
        g = df.groupBy(*[F.col(k) for k in keys]).agg(*aggs) if keys else df.agg(*aggs)
        pdf = collect_pandas(g)
        if len(pdf) and not keys:
            pdf = pdf.fillna(0.0)
        return pdf

    def _group_percentiles(self, keys, col, probs, where):
        from pyspark.sql import functions as F

        df = self._filtered(where).filter(F.col(col).isNotNull() & ~F.isnan(F.col(col)))
        arr = "array(" + ",".join(repr(float(p)) for p in probs) + ")"
        aggs = [F.expr(f"percentile(cast(`{col}` as double), {arr})").alias("_pq"),
                F.count(F.lit(1)).alias("n_valid"), F.min(F.col(col)).alias("min"), F.max(F.col(col)).alias("max")]
        g = df.groupBy(*[F.col(k) for k in keys]).agg(*aggs) if keys else df.agg(*aggs)
        sel = [F.col(k) for k in keys] + [F.col("_pq")[i].alias(pname(p)) for i, p in enumerate(probs)] + \
              [F.col("n_valid"), F.col("min"), F.col("max")]
        return collect_pandas(g.select(*sel))
