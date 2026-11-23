"""Bước 12 - Hình và bảng cho báo cáo.

Hình: cnsplots (Week 9 sec. 2-4): multipanel 183 mm, Okabe-Ito, PDF vector + PNG, panel a/b/c, mỗi hình có tiêu đề/trục/đơn vị
và chú thích 4 phần (claim | panels | statistics | n & unit) trong figure_captions.md. Câu 'Observed' sinh tự động từ số liệu.
Bảng: ba đường kẻ (booktabs), đơn vị ở tiêu đề cột, cùng số chữ số thập phân; xuất CSV + LaTeX (tiêu đề đặt TRÊN bảng).
Số liệu vẽ lấy từ engine cấu hình ở figures.engine_for_figures (hai engine đã được chứng minh trùng khớp ở bước 09).
"""
from __future__ import annotations

import importlib
import traceback

import numpy as np
import pandas as pd

from ..core.runtime import log
from ..viz.common import LOAN_PURPOSE, LOAN_TYPE, import_cns

NAME = "s12_report_assets"
ENGINE_BOUND = False
SECTIONS = ["figures", "thresholds"]

FIGS = [("fig0_data_quality", "fig_data"), ("fig1_q1_products_states", "fig_q1"), ("fig2_q2_lenders", "fig_q2"),
        ("fig3_q3_finance_bands", "fig_q3"), ("fig4_q4_pricing", "fig_q4"), ("fig5_engine_benchmark", "fig_bench"),
        ("fig6_predictive", "fig_pred")]


# ------------------------------------------------------------------ bảng ba đường kẻ
def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)


def _tex(s: str) -> str:
    return s.replace("\\", " ").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("~", "")


def three_line(df: pd.DataFrame, caption: str, label: str, note: str = "", fit: bool = False) -> str:
    cols = list(df.columns)
    out = [r"\begin{table}[t]", r"\centering", r"\caption{" + _tex(caption) + "}", r"\label{tab:" + label + "}",
           r"\small", r"\resizebox{\textwidth}{!}{%" if fit else r"\relax", r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
           " & ".join(_tex(str(c)) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join(_tex(_fmt(r[c])) for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}" + ("}" if fit else "")]
    out = [ln for ln in out if ln != r"\relax"]      # bảng không `fit` giữ nguyên đầu ra cũ
    if note:
        out.append(r"\par\footnotesize " + _tex(note))
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def run(ctx):
    cfg = ctx.cfg
    eng = cfg["figures"]["engine_for_figures"]
    eng_t = cfg.engine_out(eng) / "tables"
    cmp_t = cfg.out_dir / "compare" / "tables"
    out = cfg.out_dir / "figures"
    tdir = cfg.out_dir / "tables_report"
    out.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)

    def T(name):
        for d in (eng_t, cmp_t):
            p = d / f"{name}.csv"
            if p.exists():
                df = pd.read_csv(p, keep_default_na=False, na_values=[""], dtype={"loan_type": str, "loan_purpose": str,
                                 "lien_status": str, "state_code": str, "is_denied": str, "lei": str})
                return df
        raise FileNotFoundError(f"Thiếu bảng {name}.csv (chạy bước tương ứng trước)")

    # ---------------- bảng
    specs = []

    def spec(name, caption, builder, note="", fit=False):
        try:
            df = builder()
            note = note() if callable(note) else note
            (tdir / f"{name}.csv").write_text(df.to_csv(index=False))
            (tdir / f"{name}.tex").write_text(three_line(df, caption, name, note, fit))
            specs.append(name)
        except FileNotFoundError as e:
            log.info("[s12] bỏ qua %s: %s", name, e)

    def t_overview():
        ov = pd.read_csv(cfg.out_dir / "compare" / "overview.csv")
        return ov[ov["item"].isin(["n_files", "total_size_gib", "n_rows", "n_columns", "ts_institutions", "ts_matches_lar_rows"])].rename(columns={"item": "Item", "value": "Value"})

    def t_q1():
        p = T("q1_product_rates")
        p = p[p["n_decided"] > 0].copy()
        return pd.DataFrame({"Loan type": p["loan_type"].map(LOAN_TYPE), "Purpose": p["loan_purpose"].map(LOAN_PURPOSE),
                             "Decided (n)": p["n_decided"].astype(int), "Denial rate": p["denial_rate"],
                             "95% CI low": p["denial_ci_lo"], "95% CI high": p["denial_ci_hi"], "Origination share": p["orig_share"]})

    def t_bench():
        b = T("table3_benchmark")
        b = b[b["scenario"] == "compute_only"]
        return pd.DataFrame({"Workload": b["workload"], "Data size": b["data_size"], "Pandas (s)": b["pandas_s"], "PySpark (s)": b["pyspark_s"],
                             "Faster method": b["faster_method"], "Speedup (T_Pandas/T_Spark)": b["speedup_pandas_over_spark"]})

    def t_q2():
        d = T("q2_dispersion")
        return d[["scope", "n", "p5", "p25", "p50", "p75", "p95", "iqr", "std"]].rename(columns={"scope": "Scope", "n": "Lenders", "p50": "Median", "iqr": "IQR", "std": "SD"})

    def t_q3():
        a = T("q3_association")
        return a[["variable", "V", "V_bias_corrected", "rate_min_valid_band", "rate_max_valid_band", "rate_range_pp", "share_decided_in_status_bands"]].rename(
            columns={"variable": "Variable", "V": "Cramer's V", "V_bias_corrected": "V (bias-corr.)", "rate_min_valid_band": "Min band rate",
                     "rate_max_valid_band": "Max band rate", "rate_range_pp": "Range (pp)", "share_decided_in_status_bands": "Share in status bands"})

    def t_q4():
        c = T("q4_reg_coef_interest_rate")
        return c[["term", "coef", "se", "ci_lo", "ci_hi", "p_value", "se_cluster", "t_cluster", "ci_lo_cluster", "ci_hi_cluster"]].rename(
            columns={"term": "Term", "coef": "Coefficient (pp)", "se": "SE", "ci_lo": "95% CI low", "ci_hi": "95% CI high", "p_value": "P",
                     "se_cluster": "Cluster SE", "t_cluster": "Cluster t", "ci_lo_cluster": "Cluster 95% CI low",
                     "ci_hi_cluster": "Cluster 95% CI high"})

    def n_q4():
        g = int(T("q4_reg_summary").set_index("y").loc["interest_rate", "G"])
        return ("SE, CI and P are classical (homoskedastic); Cluster columns are lender-clustered (CR1, G = " + f"{g:,}" +
                " lenders; t = coefficient / cluster SE; 95% CI = coefficient +/- 1.96 cluster SE). Coefficients are identical. "
                "Reference level is the most frequent level of each categorical variable.")

    def t_corr():
        h = T("correctness_headline")
        wanted = ["decided_applications", "national_denial_rate", "interest_rate_median",
                  "q4_coef_loan_type[2]"]
        h = h.set_index("id").reindex(wanted).reset_index()
        return h[["quantity", "pandas", "pyspark", "difference", "tolerance", "conclusion"]].rename(
            columns={"quantity": "Quantity", "pandas": "Pandas", "pyspark": "PySpark", "difference": "Difference",
                     "tolerance": "Tolerance", "conclusion": "Conclusion"})

    def t_pred():
        m = T("pred_metrics")
        return m[["model", "test_roc_auc", "test_pr_auc", "test_log_loss", "test_brier"]].rename(columns={"model": "Model", "test_roc_auc": "ROC-AUC", "test_pr_auc": "Average precision (AP)", "test_log_loss": "Log-loss", "test_brier": "Brier"})

    def t_rob():
        r = T("robustness_summary")
        keep = [c for c in r.columns if not c.startswith("delta_")][:9]
        return r[keep]

    spec("table1_data_overview", "Data overview and reconciliation with the Transmittal Sheet", t_overview)
    spec("table2_q1_product_rates", "Denial rate among decided applications by loan type and purpose, 2024", t_q1,
         "Denominator: applications with action taken 1, 2 or 3. Wilson 95% interval. Origination share = originated / applications 1-5.")
    spec("table3_benchmark", "Pandas versus PySpark: two aggregation workloads (compute-only, median of repeats)", t_bench,
         "Workloads: lender_groupby = per-lender count, mean loan amount and denial rate; median_by_state = per-state count and exact median loan amount. "
         "Speedup = T_Pandas / T_Spark; Tie / inconclusive when min-max ranges overlap or either cell is unstable (max/min > 1.5). "
         "Results apply to these workloads and this machine only.")
    spec("table4_q2_lender_dispersion", "Dispersion of lender denial rates (lenders with enough decided applications)", t_q2,
         "Exact percentiles (linear interpolation); SD with ddof = 1.")
    spec("table5_q3_association", "Association between applicant-finance bands and denial", t_q3,
         "Cramer's V on valid bands (denied versus approved). Association only.")
    spec("table6_q4_interest_rate_regression", "OLS association of loan characteristics with interest rate (originated loans)", t_q4, n_q4, fit=True)
    spec("table7_correctness", "Pandas and PySpark outputs compared", t_corr, "Counts exact; floats rtol = atol = 1e-9.")
    spec("table8_predictive", "Predictive extension: held-out-lender performance", t_pred, "Lender-grouped split.")
    spec("table9_robustness", "Robustness of headline results to exclusions", t_rob)

    # ---------------- hình
    cns = import_cns()
    caps, fails = [], []
    for stem, mod in FIGS:
        try:
            m = importlib.import_module(f"src.viz.{mod}")
            info = m.build(cns, T, out, cfg)
            caps.append((stem, info))
            log.info("[s12] đã vẽ %s", stem)
        except FileNotFoundError as e:
            log.info("[s12] bỏ qua %s: %s", stem, e)
        except Exception as e:   # noqa: BLE001
            traceback.print_exc()
            fails.append((stem, str(e)))
    lines = ["# Figure captions (structure: claim | panels | statistics | n & unit)\n",
             "The 'Observed' line is generated from the result tables. Write the final claim and interpretation yourself after reading the figure.\n"]
    for i, (stem, info) in enumerate(caps):
        lines += [f"## {stem}\n", f"**Fig. {i} |** {info['title']}.", f"- Panels: {info['panels']}", f"- Statistics: {info['stats']}",
                  f"- n and unit: {info['n_unit']}"]
        if info.get("observed"):
            lines.append(f"- Observed: {info['observed']}")
        lines.append("")
    (out / "figure_captions.md").write_text("\n".join(lines))
    if fails:
        raise RuntimeError(f"Vẽ lỗi: {fails}")
    return {"figures": [s for s, _ in caps], "tables": specs}
