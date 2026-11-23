"""Figure 4 - Q4: interest rate / rate spread of originated loans and the regression coefficients."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .common import C_ALT, C_MAIN, boxes_from_percentiles, new_multipanel, save


def build(cns, T, out, cfg):
    ir = T("q4_summary_interest_rate_by_segment")
    sp = T("q4_summary_rate_spread_by_segment")
    coef_ir = T("q4_reg_coef_interest_rate")
    coef_sp = T("q4_reg_coef_rate_spread")
    rs = T("q4_reg_summary")
    selected = [
        ("1", "1", "1", "Conv purchase, 1st"), ("1", "1", "2", "Conv purchase, sub."),
        ("1", "31", "1", "Conv refinance, 1st"), ("1", "31", "2", "Conv refinance, sub."),
        ("1", "32", "1", "Conv cash-out, 1st"), ("1", "32", "2", "Conv cash-out, sub."),
        ("2", "1", "1", "FHA purchase, 1st"), ("3", "1", "1", "VA purchase, 1st"),
    ]

    def selected_rows(d):
        d = d.copy()
        for c in ("loan_type", "loan_purpose", "lien_status"):
            d[c] = d[c].astype(str)
        rows = []
        for lt, lp, li, label in selected:
            r = d[(d.loan_type == lt) & (d.loan_purpose == lp) & (d.lien_status == li)]
            if len(r):
                rr = r.iloc[0].copy()
                rr["label"] = label
                rows.append(rr)
        return pd.DataFrame(rows).reset_index(drop=True)

    ir, sp = selected_rows(ir), selected_rows(sp)
    mp = new_multipanel(cns)
    ax = mp.panel("a", 62, 45, margin_bottom=18)
    boxes_from_percentiles(ax, ir, ir["label"], positions=range(len(ir)), horizontal=True)
    ax.set_yticklabels(ir["label"], fontsize=5)
    ax.set_xlabel("Interest rate (%)")
    ax.set_title(f"n={int(ir['n_valid'].sum()):,}", fontsize=5.5)
    ax = mp.panel("b", 62, 45, margin_bottom=18, margin_right=0)
    boxes_from_percentiles(ax, sp, sp["label"], positions=range(len(sp)), horizontal=True, color=C_ALT)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Rate spread (percentage points)")
    ax.set_title(f"n={int(sp['n_valid'].sum()):,}", fontsize=5.5)

    def coefficient_panel(ax, coef, target, n):
        wanted = ["loan_type[2]", "loan_type[3]", "loan_type[4]", "log_loan_amount", "log_income", "cltv", "term_years", "log_property_value"]
        d = coef[coef["term"].isin(wanted)].copy()
        label = {"loan_type[2]": "FHA (vs conventional)", "loan_type[3]": "VA (vs conventional)",
                 "loan_type[4]": "RHS/FSA (vs conventional)", "log_loan_amount": "log loan amount",
                 "log_income": "log income", "cltv": "CLTV", "term_years": "term (years)",
                 "log_property_value": "log property value"}
        d["label"] = d["term"].map(label)
        cns.forestplot(data=d, label="label", estimate="coef", lower="ci_lo_cluster", upper="ci_hi_cluster", reference=0, ax=ax,
                       xlabel="OLS coefficient (pp per unit)", add_pvalue=False)
        ax.tick_params(labelsize=5)
        lo, hi = float(d["ci_lo_cluster"].min()), float(d["ci_hi_cluster"].max())
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{target}; complete-case n={int(n):,}", fontsize=5.5)

    rir = rs[rs["y"] == "interest_rate"].iloc[0]
    rsp = rs[rs["y"] == "rate_spread"].iloc[0]
    ax = mp.panel("c", 62, 46, margin_bottom=6)
    coefficient_panel(ax, coef_ir, "interest rate", rir["n"])
    ax = mp.panel("d", 62, 46, margin_bottom=6, margin_right=0)
    coefficient_panel(ax, coef_sp, "rate spread", rsp["n"])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    save(cns, out / "fig4_q4_pricing", cfg["figures"]["formats"], cfg["figures"]["dpi"])
    return {"title": "Pricing differs by product, purpose and lien; observable loan characteristics explain only part of the variation",
            "panels": "a-b, interest rate and rate spread in selected high-volume product-purpose-lien segments. c-d, matched OLS coefficient panels for interest rate and rate spread with 95% CI, lender-clustered (CR1).",
            "stats": f"Boxes = IQR, whiskers = 5th-95th percentile (exact). OLS from cell sums; panels c-d show 95% CI, lender-clustered (CR1; G = {int(rs.set_index("y").loc["interest_rate", "G"]):,} lenders for interest rate, {int(rs.set_index("y").loc["rate_spread", "G"]):,} for rate spread); R-squared = {rir['r2']:.2f} for interest rate and {rsp['r2']:.2f} for rate spread. "
                     "Association only: unobserved credit quality and market timing are not in the data.",
            "n_unit": f"Panel a n = {int(ir['n_valid'].sum()):,}; panel b n = {int(sp['n_valid'].sum()):,}; panel c n = {int(rir['n']):,}; panel d n = {int(rsp['n']):,}. Unit = originated LAR record.",
            "observed": f"Regression explains {rir['r2']:.1%} of interest-rate variance and {rsp['r2']:.1%} of rate-spread variance."}
