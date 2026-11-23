#!/usr/bin/env python
"""Render verified full-data CSV results into the editable LaTeX report.

This script deliberately fails when a required output is absent.  It never falls
back to values copied from the supplied PDF or from the synthetic fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "pandas" / "tables"
CMP = ROOT / "outputs" / "compare"
REPORT = ROOT / "report"


def table(name: str, base: Path = TAB) -> pd.DataFrame:
    path = base / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required verified output is missing: {path}")
    return pd.read_csv(path)


def row(df: pd.DataFrame, col: str, value) -> pd.Series:
    hit = df[df[col].astype(str) == str(value)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one row where {col}={value!r}; found {len(hit)}")
    return hit.iloc[0]


def comma(x) -> str:
    return f"{int(round(float(x))):,}"


def num(x, digits=3) -> str:
    return f"{float(x):.{digits}f}"


def pct(x, digits=1) -> str:
    return f"{100 * float(x):.{digits}f}"


def sci_tex(x, digits=2) -> str:
    v = float(x)
    if v == 0:
        return "0"
    mantissa, exponent = f"{v:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def tex(s) -> str:
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("<", r"$<$").replace(">", r"$>$"))


def main() -> None:
    national = table("q1_national_total").iloc[0]
    outcomes = table("ds_numeric_summary")
    flags = table("qa_flag_summary")
    states = table("q1_state_consistency_summary").iloc[0]
    state_na = table("q1_state_na_profile").iloc[0]
    state_oe = table("q1_state_oe")
    state_product = table("q1_state_product_rates")
    q2 = table("q2_summary").iloc[0]
    lender = table("q2_lender_table")
    disp = table("q2_dispersion")
    raw = row(disp, "scope", "all products (raw lender rate)")
    std = row(disp, "scope", "all products (product-mix standardised rate = O/E x national rate)")
    assoc = table("q3_association")
    q4 = row(table("q4_reg_summary"), "y", "interest_rate")
    q4_spread = row(table("q4_reg_summary"), "y", "rate_spread")
    q4coef = row(table("q4_reg_coef_interest_rate"), "term", "loan_type[2]")
    q4seg_ir = table("q4_summary_interest_rate_by_segment")
    q4seg_sp = table("q4_summary_rate_spread_by_segment")
    robustness = table("robustness_summary")
    prepare_meta = json.loads((ROOT / "data" / "interim" / "pandas_info" / "_meta.json").read_text())

    corr = json.loads((CMP / "correctness_summary.json").read_text())
    if not corr.get("all_agree"):
        raise RuntimeError("Refusing to build report macros: full-data correctness did not pass")
    bench = table("table3_benchmark", CMP / "tables")
    if "workload" in bench.columns:          # macro hiện có mô tả workload 1 (lender_groupby)
        bench = bench[bench["workload"] == "lender_groupby"]
    bcomp = bench[(bench["scenario"] == "compute_only") & (bench["fraction"] == 1.0)].iloc[0]
    bread = bench[(bench["scenario"] == "read_compute") & (bench["fraction"] == 1.0)].iloc[0]
    pred = table("pred_metrics", CMP / "tables")
    gb = row(pred, "model", "Gradient boosting")
    gb0 = row(pred, "model", "Gradient boosting, no lender profiles")
    pred_summary = json.loads((CMP / "pred_summary.json").read_text())
    pred_population = table("pred_population", CMP / "tables")
    pred_eligible = pred_population[pred_population["step"].str.startswith("no reported")].iloc[0]
    detail = table("correctness_detail", CMP / "tables")
    machine = json.loads((CMP / "machine_info.json").read_text())

    def segment(df, lt, lp, lien):
        hit = df[(df["loan_type"].astype(str) == str(lt)) &
                 (df["loan_purpose"].astype(str) == str(lp)) &
                 (df["lien_status"].astype(str) == str(lien))]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one segment {lt}|{lp}|{lien}; found {len(hit)}")
        return hit.iloc[0]

    def state_product_row(state, lt="1", lp="1"):
        hit = state_product[(state_product["state_code"].astype(str) == state) &
                            (state_product["loan_type"].astype(str) == lt) &
                            (state_product["loan_purpose"].astype(str) == lp)]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one state-product row for {state}|{lt}|{lp}")
        return hit.iloc[0]

    ca_cp, tx_cp = state_product_row("CA"), state_product_row("TX")
    eligible_lender = lender[lender["eligible"] == 1]
    conv_purchase = row(disp, "scope", "segment 1|1|1")
    ir_1, ir_2 = segment(q4seg_ir, 1, 1, 1), segment(q4seg_ir, 1, 1, 2)
    sp_1, sp_2 = segment(q4seg_sp, 1, 1, 1), segment(q4seg_sp, 1, 1, 2)

    med_rate = row(outcomes, "variable", "interest_rate")["p50"]
    n_exempt = row(flags, "flag", "has_reported_exempt_value")["n"]
    vals = {
        "NRecords": comma(national["n_rows"]),
        "NApplications": comma(national["n_apps"]),
        "NDecided": comma(national["n_decided"]),
        "NOriginations": comma(national["n_orig"]),
        "NDenied": comma(national["n_denied"]),
        "NationalDenialPct": pct(national["denial_rate"], 1),
        "NationalOriginationPct": pct(national["orig_share"], 1),
        "CAConvPurchaseOrigPct": pct(ca_cp["orig_share"], 1),
        "TXConvPurchaseOrigPct": pct(tx_cp["orig_share"], 1),
        "EligibleLenderOrigMedianPct": pct(eligible_lender["orig_share"].median(), 1),
        "EligibleLenderOrigIQRpp": pct(eligible_lender["orig_share"].quantile(.75) - eligible_lender["orig_share"].quantile(.25), 1),
        "MedianInterest": num(med_rate, 3),
        "DuplicateGroupRows": comma(prepare_meta["n_dup_rows"]),
        "DuplicateExcessRows": comma(prepare_meta["n_excess_dup_rows"]),
        "ReportedExemptRecords": comma(n_exempt),
        "ReportedExemptLenders": comma(prepare_meta["lenders_with_exempt_values"]),
        "StatesReported": comma(states["states_reported"]),
        "StatesEvaluated": comma(states["states_evaluated"]),
        "StatesConsistent": comma(states["states_consistent"]),
        "MedianStateRho": num(states["median_rho"], 2),
        "MissingStateRecords": comma(state_na["n_rows"]),
        "MissingStateDecided": comma(state_na["n_decided"]),
        "LouisianaOE": num(row(state_oe, "state_code", "LA")["oe_ratio"], 3),
        "EligibleLenders": comma(q2["lenders_eligible"]),
        "RawIQRpp": pct(raw["iqr"], 1),
        "StdIQRpp": pct(std["iqr"], 1),
        "RawSDpp": pct(raw["std"], 1),
        "StdSDpp": pct(std["std"], 1),
        "ConvPurchaseFirstLienIQRpp": pct(conv_purchase["iqr"], 1),
        "ConvPurchaseFirstLienLenders": comma(conv_purchase["n"]),
        "DtiV": num(row(assoc, "variable", "dti")["V_bias_corrected"], 3),
        "LtiV": num(row(assoc, "variable", "lti")["V_bias_corrected"], 3),
        "IncomeV": num(row(assoc, "variable", "income")["V_bias_corrected"], 3),
        "LoanAmountV": num(row(assoc, "variable", "loan_amount")["V_bias_corrected"], 3),
        "QFourN": comma(q4["n"]),
        "QFourAdjRtwo": num(q4["adj_r2"], 3),
        "QFourRMSE": num(q4["rmse"], 3),
        "QFourLoanTypeTwoCoef": num(q4coef["coef"], 3),
        "QFourReferenceLevels": tex(str(q4["reference_levels"]).replace("_", " ")),
        "QFourSpreadN": comma(q4_spread["n"]),
        "QFourSpreadAdjRtwo": num(q4_spread["adj_r2"], 3),
        "QFourSpreadRMSE": num(q4_spread["rmse"], 3),
        "ConvPurchaseFirstLienRate": num(ir_1["p50"], 3),
        "ConvPurchaseSubLienRate": num(ir_2["p50"], 3),
        "ConvPurchaseFirstLienSpread": num(sp_1["p50"], 3),
        "ConvPurchaseSubLienSpread": num(sp_2["p50"], 3),
        "RobustDenialMinPct": pct(robustness["national_denial_rate"].min(), 2),
        "RobustDenialMaxPct": pct(robustness["national_denial_rate"].max(), 2),
        "RobustCombinedLenderIQRpp": pct(row(robustness, "variant", "all_combined")["lender_rate_iqr"], 2),
        "TablesAgree": comma(corr["tables_agree"]),
        "TablesCompared": comma(corr["tables_compared"]),
        "HeadlinesAgree": comma(corr["headline_agree"]),
        "HeadlinesCompared": comma(corr["headline_items"]),
        "RawTablesAgree": comma((detail["kind"] == "raw").sum()),
        "FinalTablesAgree": comma((detail["kind"] == "tables").sum()),
        "PandasComputeFull": num(bcomp["pandas_s"], 3),
        "SparkComputeFull": num(bcomp["pyspark_s"], 3),
        "PandasReadFull": num(bread["pandas_s"], 3),
        "SparkReadFull": num(bread["pyspark_s"], 3),
        "ComputeSpeedupFull": num(bcomp["speedup_pandas_over_spark"], 2),
        "ReadSpeedupFull": num(bread["speedup_pandas_over_spark"], 2),
        "MachineCPUs": comma(machine["cpu_logical"]),
        "MachineRAM": num(machine["ram_gib"], 2),
        "PredRows": comma(pred_summary["rows_used"]),
        "PredAUC": num(gb["test_roc_auc"], 3),
        "PredAucLo": num(gb["test_roc_auc_ci_lo"], 3),
        "PredAucHi": num(gb["test_roc_auc_ci_hi"], 3),
        "PredAP": num(gb["test_pr_auc"], 3),
        "PredBrier": num(gb["test_brier"], 3),
        "PredNoProfileAUC": num(gb0["test_roc_auc"], 3),
        "PredEligiblePopulation": comma(pred_eligible["n"]),
    }
    lines = ["% Generated only from verified full-data outputs; do not edit by hand."]
    lines.extend(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in vals.items())
    (REPORT / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    h = table("correctness_headline", CMP / "tables").set_index("id")
    wanted = [
        ("decided_applications", "Decided applications"),
        ("national_denial_rate", "National denial rate"),
        ("interest_rate_median", "Median interest rate"),
        ("q4_coef_loan_type[2]", "Q4 coefficient: loan type 2"),
    ]
    out = []
    for key, label in wanted:
        r = h.loc[key]
        is_count = key == "decided_applications"
        a = comma(r["pandas"]) if is_count else num(r["pandas"], 9)
        b = comma(r["pyspark"]) if is_count else num(r["pyspark"], 9)
        d = comma(r["difference"]) if is_count else sci_tex(r["difference"])
        tolerance = "exact" if is_count else r"$10^{-9}$"
        out.append(f"{tex(label)} & {a} & {b} & {d} & {tolerance} & {tex(r['conclusion'])} \\\\")
    # Keep the final booktabs rule inside the generated input.  TeX otherwise
    # sees the end-of-input token between the last row and \bottomrule.
    out.append(r"\bottomrule")
    (REPORT / "correctness_rows.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
