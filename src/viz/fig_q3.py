"""Figure 3 - Q3: denial rate by band of income, loan amount, LTI, DTI, CLTV + Cramér's V."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import C_ALT, C_GREY, C_HI, C_MAIN, new_multipanel, pct_axis, save

VARS = [("income", "Income (USD thousand)"), ("loan_amount", "Loan amount (USD)"),
        ("lti", "Loan-to-income ratio (x)"), ("dti", "Debt-to-income ratio (%)")]


def build(cns, T, out, cfg):
    assoc = T("q3_association")
    mp = new_multipanel(cns)
    letters = "abcd"
    k = 0
    for var, xl in VARS:
        t = T(f"q3_band_rates_{var}")
        t = t[t["n_decided"] > 0].reset_index(drop=True)
        # The top row needs enough physical space for rotated tick labels and
        # its x-axis title; otherwise those elements collide with the titles
        # of the lower row when the figure is embedded at report width.
        row_bottom = 22 if k < 2 else 6
        ax = mp.panel(letters[k], 72, 39, margin_bottom=row_bottom,
                      margin_right=0 if k % 2 == 1 else None)
        k += 1
        x = np.arange(len(t))
        st = t["is_status_band"].to_numpy() == 1
        small = (t["n_decided"].to_numpy() < cfg["thresholds"]["small_cell"])
        ok = ~small   # small-n groups: hollow marker only, no CI
        ax.vlines(x[ok], t["denial_ci_lo"][ok], t["denial_ci_hi"][ok], color=C_MAIN, lw=0.8)
        ax.plot(x[~st], t["denial_rate"][~st], "-", color=C_MAIN, lw=0.8)
        valid_pt = ~st & ok
        ax.plot(x[valid_pt], t["denial_rate"][valid_pt], "o", color=C_MAIN, ms=3)
        hollow = st | small
        ax.plot(x[hollow], t["denial_rate"][hollow], "o", mfc="white", mec=C_ALT, ms=3.5)
        ax.set_xticks(x)
        labels = [str(b).replace("~OUTSIDE_DOMAIN", "Out-of-domain").replace("~", "") for b in t["band"]]
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=5)
        ax.set_ylabel("Denial rate")
        pct_axis(ax, "y")
        ar = assoc[assoc["variable"] == var].iloc[0]
        ax.set_title(f"{xl}\nvalid n={int(ar['n_decided_valid_bands']):,}; corrected V={ar['V_bias_corrected']:.3f}",
                     fontsize=5.5)
    save(cns, out / "fig3_q3_finance_bands", cfg["figures"]["formats"], cfg["figures"]["dpi"])
    top = assoc.sort_values("V_bias_corrected", ascending=False).iloc[0]
    return {"title": "Denial rates move with applicant-finance bands, and recording of the bands differs by outcome",
            "panels": "a-d, denial rate among decided applications by the four proposal-core financial bands: income, loan amount, loan-to-income and DTI. "
                      "Filled points are valid bands; hollow points are NA, Exempt or outside-analysis-domain statuses. Each panel states its valid n and corrected V.",
            "stats": "Wilson 95% intervals; bias-corrected Cramér's V on valid bands (2 x k table: denied vs approved); with millions of records every chi-square is "
                     "'significant', so V (effect size) is the quantity to read. Association only.",
            "n_unit": "Panel-specific valid-band n is printed above each panel; unit = decided application record. CLTV and property-value results remain in the accompanying table.",
            "observed": f"Largest association: {top['variable']} (bias-corrected V = {top['V_bias_corrected']:.3f}); denial rate range across its valid bands = {top['rate_range_pp']:.1f} percentage points."}
