"""Figure 0 - Data: status of the numeric fields, distribution of loan amount and income (log scale)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import C_ALT, C_GREY, C_HI, C_MAIN, new_multipanel, save


def build(cns, T, out, cfg):
    st = T("qa_numeric_status")
    h1, h2 = T("ds_hist_loan_amount"), T("ds_hist_income")
    mp = new_multipanel(cns)
    # Keep all three panels on one row at report width; the previous wrap left
    # half a page blank and made the figure look larger than its information.
    ax = mp.panel("a", 46, 43)
    fields = ["loan_amount", "income", "property_value", "cltv", "loan_term", "interest_rate", "rate_spread", "total_loan_costs", "dti"]
    p = st.pivot_table(index="field", columns="status", values="share", aggfunc="sum", fill_value=0).reindex(fields).dropna(how="all")
    nonval = 1 - p["VALUE"] if "VALUE" in p else p.sum(axis=1)
    order = nonval.sort_values().index
    left = np.zeros(len(order))
    colors = {"NA": C_ALT, "EXEMPT": C_HI, "EMPTY": C_GREY, "UNPARSABLE": "black"}
    for s in [c for c in p.columns if c != "VALUE"]:
        v = p.loc[order, s].to_numpy()
        ax.barh(range(len(order)), v, left=left, color=colors.get(s, C_GREY), label=s, height=0.7)
        left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5.5)
    ax.set_xlabel("Share of records not a plain number (fraction)")
    ax.legend(fontsize=5, frameon=False, loc="lower right")
    ax = mp.panel("b", 32, 43)
    ax.bar(np.log10((h1["bin_low"] * h1["bin_high"]) ** 0.5), h1["n"] / 1e3, width=0.1, color=C_MAIN)
    ax.set_xlabel("Loan amount (log10 USD)")
    ax.set_ylabel("Records (thousand)")
    ax = mp.panel("c", 32, 43, margin_right=0)
    ax.bar(np.log10((h2["bin_low"] * h2["bin_high"]) ** 0.5), h2["n"] / 1e3, width=0.1, color=C_ALT)
    ax.set_xlabel("Income (log10 USD thousand)")
    ax.set_ylabel("Records (thousand)")
    save(cns, out / "fig0_data_quality", cfg["figures"]["formats"], cfg["figures"]["dpi"])
    top = order[-1]
    return {"title": "Special codes are common in some numeric fields; amounts and incomes are right-skewed",
            "panels": "a, share of LAR records whose value is NA, Exempt, empty or unparsable by field. b, loan amount distribution. c, income distribution.",
            "stats": "Counts only; log10 bins of width 0.1. Values outside the positive analysis domain are flagged, not called data errors or removed.",
            "n_unit": f"n = {int(h1['n'].sum()):,} LAR records with valid loan amount; unit = LAR record.",
            "observed": f"Field with the largest non-numeric share: {top} ({nonval[top]:.1%})."}
