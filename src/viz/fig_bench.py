"""Hình 5 - Benchmark Pandas vs PySpark, hai workload (lender_groupby, median_by_state)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .common import C_ALT, C_GREY, C_HI, C_MAIN, new_multipanel, save

TIE = "Tie / inconclusive"
WL_LABEL = {"lender_groupby": "lender group-by", "median_by_state": "median by state"}


def _order(d):
    return d.assign(_o=d["data_size"].astype(str).str.rstrip("%").astype(float)).sort_values("_o")


def build(cns, T, out, cfg):
    s = T("benchmark_summary")
    t3 = T("table3_benchmark")
    mem = T("benchmark_memory")
    setup = T("benchmark_setup_cost")
    wls = [w for w in WL_LABEL if w in set(s["workload"])]
    mp = new_multipanel(cns)

    # a, b: thời gian compute-only, một panel cho mỗi workload
    last_ax = None
    for lab, w in zip("ab", wls):
        ax = mp.panel(lab, 42, 42)
        c = s[(s["scenario"] == "compute_only") & (s["workload"] == w)]
        for e, col in (("pandas", C_MAIN), ("spark", C_ALT)):
            g = c[c["engine"] == e].sort_values("fraction")
            ax.errorbar(g["fraction"] * 100, g["median_s"], yerr=[g["median_s"] - g["min_s"], g["max_s"] - g["median_s"]],
                        color=col, marker="o", ms=3, lw=0.8, capsize=2, label="Pandas" if e == "pandas" else "PySpark")
        ax.set_xlabel("Data size (% of records)")
        ax.set_ylabel("Compute time, median (s)")
        ax.set_title(WL_LABEL[w], fontsize=5.5)
        if lab == "a":
            ax.legend(frameon=False, fontsize=5)

    # c: speedup, cột nhóm theo workload; hatched = Tie / inconclusive
    ax = mp.panel("c", 42, 42, margin_right=0)
    a = t3[t3["scenario"] == "compute_only"]
    w_bar, gap = 0.8, 0.8
    xs, labels, cols = [], [], []
    pos = 0.0
    for w in wls:
        g = _order(a[a["workload"] == w])
        for r in g.itertuples():
            tie = r.faster_method == TIE
            col = C_MAIN if w == wls[0] else C_ALT
            b = ax.bar(pos, r.speedup_pandas_over_spark, color=col, width=w_bar)[0]
            if tie:
                b.set_facecolor("white"); b.set_edgecolor(col); b.set_hatch("////"); b.set_linewidth(0.6)
            xs.append(pos); labels.append(r.data_size)
            pos += 1
        pos += gap
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_xlabel("Data size, by workload (" + " | ".join(WL_LABEL[w] for w in wls) + ")", fontsize=5.5)
    ax.set_ylabel("Speedup = T_Pandas / T_Spark")
    ax.legend(handles=[Patch(facecolor=C_MAIN, label=WL_LABEL[wls[0]]), Patch(facecolor=C_ALT, label=WL_LABEL[wls[-1]]),
                       Patch(facecolor="white", edgecolor="black", hatch="////", label=TIE)],
              frameon=False, fontsize=5, loc="upper left")

    # d: bộ nhớ đỉnh, nhóm theo workload x tỷ lệ
    ax = mp.panel("d", 42, 42)
    w_ = 0.38
    xs, labels = [], []
    pos = 0.0
    for w in wls:
        for f in sorted(mem["fraction"].unique()):
            for i, (e, col) in enumerate((("pandas", C_MAIN), ("spark", C_ALT))):
                v = mem[(mem["workload"] == w) & (mem["engine"] == e) & (mem["fraction"] == f)]["peak_rss_gib_median"]
                ax.bar(pos + (i - 0.5) * w_, float(v.iloc[0]), width=w_, color=col, label=("Pandas" if e == "pandas" else "PySpark") if (w == wls[0] and f == mem["fraction"].min()) else None)
            xs.append(pos); labels.append(f"{int(f * 100)}%")
            pos += 1
        pos += 0.6
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_xlabel("Data size, by workload (" + " | ".join(WL_LABEL[w] for w in wls) + ")", fontsize=5.5)
    ax.set_ylabel("Peak memory (GiB, process tree)")
    ax.legend(frameon=False, fontsize=5)

    # e: chi phí khởi tạo
    ax = mp.panel("e", 42, 42)
    ax.barh(range(len(setup)), setup["median_s"], color=C_GREY, height=0.6)
    ax.set_yticks(range(len(setup)))
    ax.set_yticklabels(setup["item"].str.replace(r"_s$", "", regex=True).str.replace("_", " "), fontsize=5)
    ax.set_xlabel("Set-up cost, fresh process (s)")

    # f: đọc + tính, hai workload; marker rỗng = Tie / inconclusive
    ax = mp.panel("f", 42, 42, margin_right=0)
    rcs = s[s["scenario"] == "read_compute"]
    for w, ls in zip(wls, ("-", "--")):
        rc = _order(t3[(t3["scenario"] == "read_compute") & (t3["workload"] == w)])
        tie_rc = (rc["faster_method"] == TIE).to_numpy()
        for e, col, lab, eng in (("pandas_s", C_MAIN, "Pandas", "pandas"), ("pyspark_s", C_ALT, "PySpark", "spark")):
            g = rcs[(rcs["engine"] == eng) & (rcs["workload"] == w)].sort_values("fraction")
            ax.errorbar(range(len(rc)), g["median_s"], yerr=[g["median_s"] - g["min_s"], g["max_s"] - g["median_s"]],
                        color=col, marker="o" if w == wls[0] else "s", ms=3, lw=0.8, ls=ls, capsize=2,
                        label=f"{lab}, {WL_LABEL[w]}")
            ax.plot(np.arange(len(rc))[tie_rc], rc[e].to_numpy()[tie_rc], "o" if w == wls[0] else "s", mfc="white", mec=col, ms=4, zorder=4)
    ax.plot([], [], "o", mfc="white", mec="black", ms=4, ls="", label=TIE)
    ax.set_xticks(range(len(rc)))
    ax.set_xticklabels(rc["data_size"])
    ax.set_xlabel("Data size")
    ax.set_ylabel("Read + compute, median (s)")
    ax.legend(frameon=False, fontsize=4.5)
    save(cns, out / "fig5_engine_benchmark", cfg["figures"]["formats"], cfg["figures"]["dpi"])

    obs = []
    for w in wls:
        last = _order(t3[(t3["scenario"] == "compute_only") & (t3["workload"] == w)]).iloc[-1]
        obs.append(f"{WL_LABEL[w]} at {last['data_size']}: Pandas {last['pandas_s']:.3f} s, PySpark {last['pyspark_s']:.3f} s, faster = {last['faster_method']}")
    n100 = int(_order(t3).iloc[-1]["records"])
    return {"title": "Relative speed of Pandas and PySpark depends on data size, workload and on what is timed",
            "panels": "a-b, compute-only median time (bars = min-max over repeats) for the lender group-by and the exact median by state. "
                      "c, speedup T_Pandas / T_Spark per workload (above 1 = Spark faster; hatched bars = Tie / inconclusive). "
                      "d, peak memory in fresh processes. e, start-up costs. f, read + compute scenario for both workloads (hollow markers = Tie / inconclusive).",
            "stats": "Median of repeated runs after warm-up; seeded rotated cell order; Spark cached and counted before timing in a-c. "
                     "A cell is Tie / inconclusive when the min-max ranges overlap or either cell is unstable (max/min > 1.5). "
                     "Valid for these two workloads and this machine only.",
            "n_unit": f"n = {n100:,} records at 100%; unit = application record; two aggregation tasks.",
            "observed": "; ".join(obs) + "."}
