"""Figure 1 - Q1: denial rate by loan type x purpose; state x loan type; consistency across states."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import C_ALT, C_GREY, C_HI, C_LO, C_MAIN, errorbar_h, new_multipanel, pct_axis, product_label, save


def build(cns, T, out, cfg):
    prod = T("q1_product_rates")
    prod = prod[prod["n_decided"] > 0]
    stt = T("q1_state_type_rates")
    cons = T("q1_state_consistency")
    summ = T("q1_state_consistency_summary").iloc[0]
    oe = T("q1_state_oe")
    mp = new_multipanel(cns)
    # Display the twelve largest product cells and both proposal outcomes on
    # the same axis.  All 24 cells remain in the output table and statistics.
    shown = prod.nlargest(12, "n_decided").sort_values("denial_rate").reset_index(drop=True)
    ax = mp.panel("a", 72, 55, margin_bottom=18)
    lab = [product_label(a, b) for a, b in zip(shown["loan_type"], shown["loan_purpose"])]
    y = np.arange(len(shown))
    ax.hlines(y + 0.12, shown["denial_ci_lo"], shown["denial_ci_hi"], color=C_MAIN, lw=0.8)
    ax.plot(shown["denial_rate"], y + 0.12, "o", color=C_MAIN, ms=3, label="Denial / decided")
    ax.hlines(y - 0.12, shown["orig_ci_lo"], shown["orig_ci_hi"], color=C_ALT, lw=0.8)
    ax.plot(shown["orig_share"], y - 0.12, "s", color=C_ALT, ms=2.8, label="Originated / applications")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=5)
    ax.set_xlabel("Rate (own denominator)")
    pct_axis(ax, "x")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=4.8, loc="lower right", labelspacing=0.2, handletextpad=0.3)
    # b: heatmap state x loan type
    top_states = (stt.groupby("state_code")["n_decided"].sum().sort_values(ascending=False).head(10).index)
    m = stt[stt["state_code"].isin(top_states)].pivot_table(index="state_code", columns="loan_type", values="denial_rate")
    m = m.reindex(top_states)
    m.columns = [{"1": "Conv", "2": "FHA", "3": "VA", "4": "RHS"}.get(str(c), str(c)) for c in m.columns]
    ax = mp.panel("b", 44, 55, margin_bottom=18, margin_right=2)
    im = ax.imshow(m.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, fontsize=5)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index, fontsize=5)
    ax.set_xlabel("Loan type")
    ax.set_ylabel("State (10 largest by decided volume)")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=4, color="white" if v < np.nanmedian(m.values) else "black")
    # c: Spearman
    ax = mp.panel("c", 42, 50, margin_bottom=12, margin_right=0)
    ev = cons[cons["evaluated"].astype(bool)]
    ax.hist(ev["spearman_rho"].dropna(), bins=np.linspace(-1, 1, 21), color=C_MAIN, edgecolor="white", lw=0.4)
    ax.axvline(summ["rho_threshold"], color=C_HI, ls="--", lw=0.8)
    ax.set_xlabel("Rank corr. (rho) vs national")
    ax.set_ylabel("Number of states")
    # d: O/E extremes
    ax = mp.panel("d", 76, 50, margin_right=0)
    o = oe[oe["n_decided"] >= cfg["thresholds"]["small_cell"] * 10].sort_values("oe_ratio")
    sel = pd.concat([o.head(6), o.tail(6)]).drop_duplicates("state_code")
    colr = np.where(sel["oe_lo"] > 1, C_HI, np.where(sel["oe_hi"] < 1, C_LO, C_GREY))
    ax.hlines(range(len(sel)), sel["oe_lo"], sel["oe_hi"], colors=colr, lw=0.8)
    ax.scatter(sel["oe_ratio"], range(len(sel)), c=colr, s=9, zorder=3)
    ax.axvline(1, color="black", lw=0.5, ls=":")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(sel["state_code"], fontsize=5)
    ax.set_xlabel("State denial O/E (product-mix standardised)")
    save(cns, out / "fig1_q1_products_states", cfg["figures"]["formats"], cfg["figures"]["dpi"])
    hi, lo = prod.iloc[-1], prod.iloc[0]
    return {"title": "Denial rates differ by loan product and are broadly consistent across states",
            "panels": "a, denial rate and origination share for the 12 largest product cells (different denominators). "
                      "b, state x loan type denial rate (cell label = percent denied). c, Spearman correlation between each state's product-level rates and the national pattern "
                      "(dashed = threshold). d, state observed/expected denial extremes after product-mix standardisation. "
                      "State origination-share results are reported in the accompanying table rather than adding another crowded panel.",
            "stats": "Wilson 95% intervals (a); Spearman on product cells meeting the configured size threshold (c), interpreted as agreement in product ranking, "
                      "not equality of state rates; Garwood 95% intervals for O/E (d). Differences are associations, not causal effects.",
            "n_unit": f"n = {int(prod['n_decided'].sum()):,} decided applications for denial and {int(prod['n_apps'].sum()):,} applications for origination share; "
                      "unit = application record; state NA (location not provided) excluded from b-d.",
            "observed": None}   # drop the Observed line (which names a 13-record cell) from the caption
