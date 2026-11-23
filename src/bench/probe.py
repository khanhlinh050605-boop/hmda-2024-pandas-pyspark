"""Tiến trình con để đo bộ nhớ đỉnh và chi phí khởi tạo trong TIẾN TRÌNH MỚI (không bị nhiễm từ các lần chạy trước).

  python -m src.bench.probe --config config.yaml --engine pandas --fraction 1.0
  python -m src.bench.probe --config config.yaml --engine spark  --fraction 0.5
  python -m src.bench.probe --config config.yaml --mode setup --engine spark
In JSON ra stdout. Bộ nhớ = RSS của tiến trình + mọi tiến trình con (gồm JVM của Spark).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import psutil

from ..core.config import load_config


class PeakRSS:
    def __init__(self, interval=0.05):
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self.interval = interval
        self.t = threading.Thread(target=self._run, daemon=True)

    def _rss(self):
        tot = 0
        try:
            tot += self.proc.memory_info().rss
            for c in self.proc.children(recursive=True):
                try:
                    tot += c.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        return tot

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.peak = max(self.peak, self._rss())
        self._stop.set()
        self.t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--engine", choices=["pandas", "spark"], required=True)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--mode", choices=["memory", "setup"], default="memory")
    ap.add_argument("--workload", default="lender_groupby")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = {"engine": a.engine, "fraction": a.fraction, "mode": a.mode}
    if a.mode == "setup":
        if a.engine == "pandas":
            t = time.perf_counter()
            import pandas  # noqa: F401
            out["import_pandas_s"] = time.perf_counter() - t
        else:
            from ..engines.spark_engine import get_spark
            t = time.perf_counter()
            spark = get_spark(cfg)
            out["spark_session_start_s"] = time.perf_counter() - t
            t = time.perf_counter()
            source = cfg["benchmark"].get("input_master", "pandas")
            n = spark.read.parquet((cfg.interim_dir / f"{source}_master").as_posix()).count()
            out["first_spark_action_s"] = time.perf_counter() - t
            spark.stop()
        print(json.dumps(out))
        return
    from ..core.runtime import Ctx
    from .workload import WORKLOADS
    wl = WORKLOADS[a.workload]
    COLS, pandas_workload, spark_workload = wl["cols"], wl["pandas"], wl["spark"]
    out["workload"] = a.workload
    with PeakRSS() as pk:
        if a.engine == "pandas":
            from ..engines.pandas_engine import PandasEngine
            eng = PandasEngine(cfg)
            eng.open(Ctx(cfg, eng, "pandas"))
            import pandas as pd
            data = pd.DataFrame({c: eng.col(c) for c in COLS})
            u = eng.col("sample_u").to_numpy()
            sub = data[u < a.fraction]
            res = pandas_workload(sub)
            out["rows"] = int(len(sub))
        else:
            from pyspark.sql import functions as F
            from ..engines.spark_engine import get_spark
            spark = get_spark(cfg)
            ctx = Ctx(cfg, None, "spark")
            source = cfg["benchmark"].get("input_master", "pandas")
            df = spark.read.parquet((cfg.interim_dir / f"{source}_master").as_posix())
            sub = df.filter(F.col("sample_u") < a.fraction).select(*COLS).cache()
            out["rows"] = int(sub.count())
            spark_workload(sub)
            spark.stop()
    out["peak_rss_bytes"] = int(pk.peak)
    out["peak_rss_gib"] = pk.peak / 2**30
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
