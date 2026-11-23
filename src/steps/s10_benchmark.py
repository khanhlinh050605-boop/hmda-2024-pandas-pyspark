"""Bước 10 - Benchmark hiệu năng và khả năng mở rộng Pandas vs PySpark (đề bài 3.5).

Workload: theo lender -> số bản ghi, mean(loan_amount), denial rate. Mẫu lồng nhau 25/50/100% (khoá sample_u, giống
hệt ở hai engine). Kịch bản CHÍNH = chỉ tính toán: Pandas đã nạp sẵn, Spark đã cache + count trước khi bấm giờ,
Spark bị buộc thực thi bằng action (collect). Kịch bản PHỤ = đọc + tính (báo cáo riêng).
Thứ tự các ô (engine x tỷ lệ) được xoay có seed trong từng lần lặp; có warm-up toàn cục; báo cáo trung vị/min/max/IQR.
Bộ nhớ đỉnh và chi phí khởi tạo đo ở TIẾN TRÌNH MỚI (src/bench/probe.py).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from ..bench.workload import WORKLOADS, spark_rows_to_frame
from ..core.runtime import Ctx, log

NAME = "s10_benchmark"
ENGINE_BOUND = False
SECTIONS = ["benchmark", "spark", "project"]


def _probe(cfg, *args) -> dict:
    cmd = [sys.executable, "-m", "src.bench.probe", "--config", cfg.source, *args]
    r = subprocess.run(cmd, cwd=str(cfg.root), capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not line:
        raise RuntimeError(f"probe lỗi: {' '.join(args)}\n{r.stderr[-800:]}")
    return json.loads(line[-1])


def _machine(cfg) -> dict:
    import pandas
    import pyarrow
    import pyspark
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
            "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 2), "pandas": pandas.__version__,
            "numpy": np.__version__, "pyarrow": pyarrow.__version__, "pyspark": pyspark.__version__,
            "spark_config": cfg["spark"], "benchmark_config": cfg["benchmark"], "workloads": list(WORKLOADS)}
    try:
        j = subprocess.run(["java", "-version"], capture_output=True, text=True)
        info["java"] = (j.stderr or j.stdout).strip().splitlines()[-3:] if (j.stderr or j.stdout) else ""
    except Exception:  # noqa: BLE001
        info["java"] = "not found"
    return info


TIE = "Tie / inconclusive"


def decide_faster(a, b) -> str:
    """Quy tắc của nhóm: chỉ kết luận engine nhanh hơn khi khoảng min-max của hai engine KHÔNG chồng nhau
    và cả hai ô đều không bị gắn cờ không ổn định (max/min > ngưỡng). Không bao giờ quyết định chỉ bằng trung vị."""
    if bool(a["unstable"]) or bool(b["unstable"]):
        return TIE
    if a["max_s"] < b["min_s"]:
        return "Pandas"
    if b["max_s"] < a["min_s"]:
        return "PySpark"
    return TIE


def _stats(x):
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75])
    return {"median_s": float(np.median(x)), "min_s": float(x.min()), "max_s": float(x.max()), "iqr_s": float(q3 - q1),
            "n_runs": int(x.size), "unstable": bool(x.min() > 0 and x.max() / x.min() > 1.5)}


def _append_correctness_rows(cfg, wl_tables) -> None:
    """Ghi đầu ra workload `median_by_state` của từng engine ra CSV và so sánh bằng chính compare_tables của bước 09
    (rtol = atol = 1e-9, state_code là khoá chuỗi); thêm các dòng kind="benchmark" vào correctness_detail.csv.
    Chạy lại bước 09 sẽ ghi đè file này -> chạy lại bước 10 sau đó."""
    from .s09_correctness import _read, compare_tables

    base = cfg.out_dir / "compare" / "benchmark_tables"
    rows = []
    for (wname, sc, e, f), df in wl_tables.items():
        if wname != "median_by_state":
            continue
        d = base / e
        d.mkdir(parents=True, exist_ok=True)
        df.sort_values("state_code").to_csv(d / f"benchmark_{wname}_{sc}_f{int(f * 100)}.csv", index=False, float_format="%.17g")
    tol = cfg["correctness"]
    for fp in sorted((base / "pandas").glob("benchmark_*.csv")):
        res, _ = compare_tables(_read(fp), _read(base / "spark" / fp.name), tol["rtol"], tol["atol"], fp.stem, {"state_code"}, {})
        rows.append({"kind": "benchmark", "table": fp.stem, "rtol": tol["rtol"], "atol": tol["atol"], "column_overrides": "{}", **res})
    det_path = cfg.out_dir / "compare" / "tables" / "correctness_detail.csv"
    if det_path.exists():
        det = pd.read_csv(det_path, dtype=str, keep_default_na=False)
        det = det[det["kind"] != "benchmark"]
        pd.concat([det, pd.DataFrame(rows).astype(str)], ignore_index=True).to_csv(det_path, index=False)
    bad = [r["table"] for r in rows if not r["agree"]]
    if bad:
        raise RuntimeError(f"Workload output tables differ between engines: {bad}")
    log.info("[s10] %d bảng đầu ra median_by_state khớp Pandas-Spark (đã thêm vào correctness_detail.csv)", len(rows))


def run(ctx):
    cfg = ctx.cfg
    bc = cfg["benchmark"]
    fracs = list(bc["fractions"])
    from pyspark.sql import functions as F

    from ..engines.pandas_engine import PandasEngine
    from ..engines.spark_engine import SparkEngine

    corr_path = cfg.out_dir / "compare" / "correctness_summary.json"
    corr = json.loads(corr_path.read_text()) if corr_path.exists() else {}
    if not corr.get("all_agree"):
        raise RuntimeError("Benchmark requires a fresh passing correctness_summary.json; run step 09 first.")
    manifest_path = cfg.out_dir / "compare" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    recorded_hash = manifest.get("s09_correctness", {}).get("cfg_hash")
    current_hash = cfg.section_hash(["correctness"])
    if recorded_hash != current_hash:
        raise RuntimeError("Benchmark requires correctness from the current code/config fingerprint; "
                           "run step 09 with --force before step 10.")

    pe, se = PandasEngine(cfg), SparkEngine(cfg)
    pctx, sctx = Ctx(cfg, pe, "pandas"), Ctx(cfg, se, "spark")
    pe.open(pctx)
    spark = se._spark()
    source_engine = bc.get("input_master", "pandas")
    benchmark_master = cfg.interim_dir / f"{source_engine}_master"
    sdf = spark.read.parquet(benchmark_master.as_posix())

    timings, equiv, plans = [], [], []
    wl_tables = {}                       # (workload, scenario, engine, fraction) -> bảng đầu ra (rep 0), cho đối chiếu correctness
    n_rows = {}
    seed = int(cfg["project"]["seed"])
    # Mỗi workload chạy trọn giao thức riêng (warm-up, repeats, xoay ô có seed). Workload 1 dùng đúng seed cũ; workload 2 seed + 1,
    # nên chuỗi xoay của workload 1 không đổi.
    for wi, (wname, wl) in enumerate(WORKLOADS.items()):
        wcols, p_fn, s_fn, eq_fn = wl["cols"], wl["pandas"], wl["spark"], wl["equiv"]
        rng = np.random.default_rng(seed + wi)
        log.info("[s10] ===== workload %s =====", wname)

        # ---- nạp sẵn dữ liệu (NGOÀI thời gian đo của kịch bản chính)
        pdata = pe.read_columns(wcols + ["sample_u"])
        pu = pdata.pop("sample_u").to_numpy()
        p_sub = {f: pdata[pu < f] for f in fracs}
        s_sub = {}
        for f in fracs:
            s_sub[f] = sdf.filter(F.col("sample_u") < f).select(*wcols).cache()
            n_rows[f] = int(s_sub[f].count())
            assert n_rows[f] == len(p_sub[f]), f"mẫu {f}: Pandas {len(p_sub[f])} dòng != Spark {n_rows[f]} dòng"

        def run_cell(engine, f):
            t = time.perf_counter()
            res = p_fn(p_sub[f]) if engine == "pandas" else s_fn(s_sub[f])
            return time.perf_counter() - t, res

        cells = [(e, f) for e in ("pandas", "spark") for f in fracs]
        log.info("[s10] warm-up x%d", bc["warmups"])
        for _ in range(bc["warmups"]):
            for c in cells:
                run_cell(*c)
        timings_wl = []
        results = {}
        log.info("[s10] đo %d lần lặp x %d ô", bc["repeats"], len(cells))
        for rep in range(bc["repeats"]):
            order = rng.permutation(len(cells))
            for i in order:
                e, f = cells[i]
                sec, res = run_cell(e, f)
                timings_wl.append({"workload": wname, "engine": e, "fraction": f, "rep": rep, "seconds": sec, "scenario": "compute_only",
                                   "cell_order": int(np.where(order == i)[0][0])})
                if rep == 0:
                    results[(e, f)] = res if e == "pandas" else spark_rows_to_frame(res)
        eq_wl = []
        for f in fracs:
            eq = eq_fn(results[("pandas", f)], results[("spark", f)])
            eq_wl.append({"workload": wname, "scenario": "compute_only", "fraction": f, "rows": n_rows[f],
                          "lenders": len(results[("pandas", f)]), **eq})
            for e in ("pandas", "spark"):
                wl_tables[(wname, "compute_only", e, f)] = results[(e, f)]
        if not all(e["same_lenders"] and e["counts_equal"] and e["floats_close"] for e in eq_wl):
            raise RuntimeError(f"Compute-only benchmark outputs are not equivalent ({wname}); timing comparison aborted.")

        # Read+compute must not inherit Spark DataFrame cache from compute-only.
        for f in fracs:
            s_sub[f].unpersist(blocking=True)
        spark.catalog.clearCache()
        del p_sub, pdata, pu

        # ---- kịch bản phụ: đọc + tính
        if bc.get("read_compute", True):
            read_results = {}

            def read_compute_cell(e, f, capture_plan=False):
                """Đúng đường đi của vòng đo (dùng chung cho warm-up và đo)."""
                if e == "pandas":
                    d = pe.read_columns(wcols + ["sample_u"])
                    uu = d.pop("sample_u").to_numpy()
                    return p_fn(d[uu < f])
                spark.catalog.clearCache()
                df = spark.read.parquet(benchmark_master.as_posix()).select("sample_u", *wcols)
                sub = df.filter(F.col("sample_u") < f).select(*wcols)
                res = s_fn(sub)
                if capture_plan:
                    plans.append({"workload": wname, "fraction": f, "physical_plan": sub._jdf.queryExecution().executedPlan().toString()})
                    if "FileScan parquet" not in plans[-1]["physical_plan"]:
                        raise RuntimeError("Spark read+compute physical plan does not show a Parquet FileScan.")
                return spark_rows_to_frame(res)

            # warm-up riêng cho read_compute (không ghi vào timings, không dùng rng của vòng đo)
            log.info("[s10] warm-up read_compute x%d", bc["warmups"])
            for _ in range(bc["warmups"]):
                for e, f in cells:
                    read_compute_cell(e, f)
            for rep in range(bc["repeats"]):
                order = rng.permutation(len(cells))
                for i in order:
                    e, f = cells[i]
                    t = time.perf_counter()
                    res = read_compute_cell(e, f, capture_plan=(rep == 0 and e == "spark"))
                    timings_wl.append({"workload": wname, "engine": e, "fraction": f, "rep": rep, "seconds": time.perf_counter() - t,
                                       "scenario": "read_compute", "cell_order": int(np.where(order == i)[0][0])})
                    if rep == 0:
                        read_results[(e, f)] = res
            for f in fracs:
                eq = eq_fn(read_results[("pandas", f)], read_results[("spark", f)])
                eq_wl.append({"workload": wname, "scenario": "read_compute", "fraction": f, "rows": n_rows[f],
                              "lenders": len(read_results[("pandas", f)]), **eq})
                for e in ("pandas", "spark"):
                    wl_tables[(wname, "read_compute", e, f)] = read_results[(e, f)]
            if not all(e["same_lenders"] and e["counts_equal"] and e["floats_close"] for e in eq_wl):
                raise RuntimeError(f"Read+compute benchmark outputs are not equivalent ({wname}); timing comparison aborted.")
        timings += timings_wl
        equiv += eq_wl
    (ctx.out / "spark_read_compute_physical_plans.json").write_text(json.dumps(plans, indent=2))
    se.close()

    # ---- đưa bảng đầu ra của workload 2 vào đối chiếu correctness (đầu ra mỗi engine -> compare/benchmark_tables/<engine>/)
    _append_correctness_rows(cfg, wl_tables)

    tm = pd.DataFrame(timings)
    ctx.save_table("benchmark_timings_raw", tm)
    ctx.save_table("benchmark_equivalence", pd.DataFrame(equiv))
    summ = []
    for wname in WORKLOADS:
        for sc in tm["scenario"].unique():
            for e in ("pandas", "spark"):
                for f in fracs:
                    x = tm[(tm.workload == wname) & (tm.scenario == sc) & (tm.engine == e) & (tm.fraction == f)]["seconds"]
                    summ.append({"workload": wname, "scenario": sc, "engine": e, "fraction": f, **_stats(x)})
    summ = pd.DataFrame(summ)
    # cờ không đơn điệu: trung vị phải không giảm khi dữ liệu tăng
    for (wname, sc, e), g in summ.groupby(["workload", "scenario", "engine"]):
        g = g.sort_values("fraction")
        non_mono = (g["median_s"].diff().dropna() < 0).any()
        summ.loc[g.index, "non_monotone"] = bool(non_mono)
    ctx.save_table("benchmark_summary", summ)

    # ---- bảng theo mẫu đề bài
    t3 = []
    for wname in WORKLOADS:
        for sc in summ["scenario"].unique():
            for f in fracs:
                sel = (summ.workload == wname) & (summ.scenario == sc) & (summ.fraction == f)
                a = summ[sel & (summ.engine == "pandas")].iloc[0]
                b = summ[sel & (summ.engine == "spark")].iloc[0]
                t3.append({"workload": wname, "scenario": sc, "fraction": float(f), "data_size": f"{int(f * 100)}%", "records": n_rows[f],
                           "pandas_s": a["median_s"], "pyspark_s": b["median_s"],
                           "faster_method": decide_faster(a, b),
                           "faster_by_median": "Pandas" if a["median_s"] < b["median_s"] else "PySpark",
                           "speedup_pandas_over_spark": a["median_s"] / b["median_s"],
                           "ranges_overlap": bool(not (a["max_s"] < b["min_s"] or b["max_s"] < a["min_s"])),
                           "pandas_unstable": a["unstable"], "spark_unstable": b["unstable"]})
    ctx.save_table("table3_benchmark", pd.DataFrame(t3))

    # ---- bộ nhớ đỉnh + chi phí khởi tạo (tiến trình mới)
    mem = []
    for wname in WORKLOADS:
        for e in ("pandas", "spark"):
            for f in fracs:
                runs = [_probe(cfg, "--engine", e, "--fraction", str(f), "--workload", wname) for _ in range(bc["memory_probe_runs"])]
                peaks = [r["peak_rss_gib"] for r in runs]
                mem.append({"workload": wname, "engine": e, "fraction": f, "peak_rss_gib_median": float(np.median(peaks)),
                            "peak_rss_gib_min": float(min(peaks)), "peak_rss_gib_max": float(max(peaks)), "runs": len(peaks)})
    ctx.save_table("benchmark_memory", pd.DataFrame(mem))
    setup = {}
    for _ in range(3):
        s1 = _probe(cfg, "--engine", "pandas", "--mode", "setup")
        s2 = _probe(cfg, "--engine", "spark", "--mode", "setup")
        for k, v in {**s1, **s2}.items():
            if k.endswith("_s"):
                setup.setdefault(k, []).append(v)
    setup_tab = pd.DataFrame([{"item": k, "median_s": float(np.median(v)), "runs": len(v)} for k, v in setup.items()])
    ctx.save_table("benchmark_setup_cost", setup_tab)
    info = _machine(cfg)
    (ctx.out / "machine_info.json").write_text(json.dumps(info, indent=2, default=str))

    # ---- 5 câu diễn giải bằng số ĐÃ ĐO (đề bài 3.5C)
    (ctx.out / "benchmark_interpretation.md").write_text(_interpret(pd.DataFrame(t3), summ, pd.DataFrame(mem), setup_tab, pd.DataFrame(equiv)))
    log.info("[s10] xong. equivalence=%s", all(e["counts_equal"] and e["floats_close"] for e in equiv))
    return {"equivalent": all(e["counts_equal"] and e["floats_close"] for e in equiv),
            "input_master": source_engine}


def _interpret(t3, summ, mem, setup, equiv) -> str:
    L = ["# Benchmark: interpretation (numbers measured on this machine/workloads only)\n",
         "Workloads: `lender_groupby` (per lender: count, mean loan amount, denial rate) and `median_by_state` "
         "(per state_code: count and exact median loan amount; the literal state \"NA\" is its own group). "
         "Each workload runs the full protocol (untimed warm-up, 7 repeats, seeded rotated cell order).\n"]
    wls = list(t3["workload"].unique())
    for wname in wls:
        tw = t3[t3.workload == wname]
        c = tw[tw.scenario == "compute_only"].sort_values("fraction")
        L.append(f"## 1. [{wname}] Which implementation was faster?\n")
        for r in c.itertuples():
            why = ""
            if r.faster_method == TIE:
                why = (" (min-max ranges overlap)" if r.ranges_overlap else "") + \
                      (" (at least one cell flagged unstable: max/min above the threshold)" if (r.pandas_unstable or r.spark_unstable) else "")
            L.append(f"- {r.data_size} ({r.records:,} records): Pandas {r.pandas_s:.3f} s, PySpark {r.pyspark_s:.3f} s -> "
                     f"faster: **{r.faster_method}**{why}; speedup (T_Pandas / T_Spark) = {r.speedup_pandas_over_spark:.2f}.")
        L.append(f"\n## 2. [{wname}] Did relative performance change as data volume increased?\n")
        sp = c["speedup_pandas_over_spark"].tolist()
        trend = "rose" if sp[-1] > sp[0] else "fell"
        L.append(f"- Speedup went {' -> '.join(f'{x:.2f}' for x in sp)} from {c.data_size.iloc[0]} to {c.data_size.iloc[-1]}; "
                 f"Pandas/Spark ratio {trend} with volume (Pandas time x{c.pandas_s.iloc[-1] / c.pandas_s.iloc[0]:.2f}, "
                 f"PySpark time x{c.pyspark_s.iloc[-1] / c.pyspark_s.iloc[0]:.2f} from smallest to largest).")
        rc = tw[tw.scenario == "read_compute"].sort_values("fraction")
        if len(rc):
            L.append(f"- Read + compute (separate scenario, {wname}): " + "; ".join(
                f"{r.data_size}: Pandas {r.pandas_s:.2f} s vs PySpark {r.pyspark_s:.2f} s -> {r.faster_method}" for r in rc.itertuples()) + ".")
        L.append("")
    L.append("## 3. What overheads affected Spark?\n")
    for r in setup.itertuples():
        L.append(f"- {r.item}: {r.median_s:.3f} s (median of {r.runs} fresh processes).")
    L.append("- Spark catalog/DataFrame caches were cleared before every read+compute cell and the saved physical plans show "
             "a Parquet FileScan. The operating-system file cache was not and cannot be cleared portably.")
    L.append("- In the primary scenario Spark data were cached and counted before timing, so task scheduling and "
             "serialisation overhead remain, but reading and session start are excluded.")
    L.append("\n## 4. Did memory limitations affect either implementation?\n")
    for r in mem.itertuples():
        L.append(f"- [{r.workload}] {r.engine} {int(r.fraction * 100)}%: peak RSS {r.peak_rss_gib_median:.2f} GiB "
                 f"(min {r.peak_rss_gib_min:.2f}, max {r.peak_rss_gib_max:.2f}; local process tree including JVM for Spark).")
    L.append("- Both engines completed all sizes without out-of-memory errors if this file was produced (the step would have failed otherwise).")
    L.append("\n## 5. Under what circumstances would Spark be preferable?\n")
    L.append("- This item is guidance, not a measurement: the results above come from one machine and two aggregation workloads "
             "(a lender-level group-by and an exact median by state). "
             "Spark becomes preferable when data no longer fit in one machine's RAM, when the work can be spread over several "
             "cores/nodes, or when the pipeline repeatedly reads large partitioned files; for data that fit in memory and a single "
             "aggregation, Pandas avoids Spark's start-up and scheduling costs.")
    L.append("\n## Result equivalence (required before comparing speed)\n")
    for r in equiv.itertuples():
        grp = "lenders" if r.workload == "lender_groupby" else "state groups"
        L.append(f"- [{r.workload}] {r.scenario}, {int(r.fraction * 100)}%: {r.lenders} {grp}, counts equal = {r.counts_equal}, "
                 f"floats within 1e-9 = {r.floats_close}.")
    return "\n".join(L) + "\n"
