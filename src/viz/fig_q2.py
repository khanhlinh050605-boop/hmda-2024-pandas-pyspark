"""Figure 2 - Q2: dispersion of denial rates between lenders and within each segment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import C_ALT, C_GREY, C_HI, C_LO, C_MAIN, boxes_from_percentiles, new_multipanel, pct_axis, product_label, save


def build(cns, T, out, cfg):
    L = T("q2_lender_table")
    el = L[L["eligible"] == 1]
    disp = T("q2_dispersion")
    summ = T("q2_summary").iloc[0]
    nat = summ["national_denial_rate"]
    mp = new_multipanel(cns)
    ax = mp.panel("a", 42, 42)
    ax.hist(el["denial_rate"], bins=25, color=C_MAIN, edgecolor="white", lw=0.4)
    ax.axvline(nat, color=C_HI, ls="--", lw=0.8)
    ax.set_xlabel("Lender denial rate among decided")
    ax.set_ylabel("Number of lenders")
    pct_axis(ax, "x")
    ax = mp.panel("b", 42, 42)
    col = np.where(el["oe_lo"] > 1, C_HI, np.where(el["oe_hi"] < 1, C_LO, C_GREY))
    ax.scatter(np.log10(el["n_decided"]), el["oe_ratio"], c=col, s=5, lw=0)
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Decided applications (log10)")
    ax.set_ylabel("Observed / expected denials")
    ax = mp.panel("c", 42, 42, margin_right=0)
    ax.scatter(el["denial_rate"], el["oe_ratio"], c=C_MAIN, s=5, lw=0)
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Raw lender denial rate")
    ax.set_ylabel("Observed / expected denials")
    ax.set_xlim(0, 1)
    pct_axis(ax, "x")
    ax = mp.panel("d", 140, 44, margin_right=0)
    seg = disp[disp["scope"].str.startswith("segment") | disp["scope"].str.startswith("all products (raw")].reset_index(drop=True)
    labels = []
    for s in seg["scope"]:
        if s.startswith("segment"):
            lt, lp, lien = s.replace("segment ", "").split("|")
            labels.append(f"{product_label(lt, lp)}\nlien {lien}")
        else:
            labels.append("All products\n(raw)")
    boxes_from_percentiles(ax, seg, labels, positions=range(len(seg)))
    ax.axhline(nat, color=C_HI, ls="--", lw=0.8)
    ax.set_ylabel("Lender denial rate (p5-p95, IQR box)")
    pct_axis(ax, "y")
    ax.tick_params(axis="x", labelsize=5)
    save(cns, out / "fig2_q2_lenders", cfg["figures"]["formats"], cfg["figures"]["dpi"])
    return {"title": "Lender denial rates are widely dispersed, and much of the spread persists within common product segments",
            "panels": "a, distribution of lender denial rates (eligible lenders; dashed = national rate). b, observed/expected denials versus lender "
                      "size (red/blue = interval excludes 1). c, observed/expected denials versus raw rate (dotted = O/E 1; raw rate shown 0-100%). d, dispersion overall and "
                      "within the largest segments.",
            "stats": f"Lenders need at least {int(summ['min_decided'])} decided applications; Garwood intervals for O/E; whiskers = 5th-95th percentile. "
                     "Standardisation reduces product-mix differences but does not remove confounding (no credit score, income controls or lender policy).",
            "n_unit": f"n = {int(summ['lenders_eligible'])} eligible of {int(summ['lenders_total'])} lenders; unit = lender.",
            "observed": f"Lender rate p5-p95: {disp.iloc[0]['p5']:.1%} to {disp.iloc[0]['p95']:.1%} (median {disp.iloc[0]['p50']:.1%}); "
                        f"IQR changes from {summ['iqr_raw'] * 100:.1f} to {summ['iqr_standardised'] * 100:.1f} pp, while standard deviation changes "
                        f"from {summ['std_raw']:.3f} to {summ['std_standardised']:.3f}; the two dispersion measures are reported separately."}
