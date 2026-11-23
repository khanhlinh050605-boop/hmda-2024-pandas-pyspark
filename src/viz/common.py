"""Plotting utilities using cnsplots (Week 9, sec. 2-4): sizes in mm, Okabe-Ito palette, panels a/b/c,
saved as vector PDF + PNG. cnsplots is imported only when plotting (needs Python 3.12)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OKABE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
         "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
C_MAIN, C_ALT, C_HI, C_LO, C_GREY = OKABE["blue"], OKABE["orange"], OKABE["vermillion"], OKABE["sky"], "#888888"

LOAN_TYPE = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "RHS/FSA"}
LOAN_PURPOSE = {"1": "Purchase", "2": "Improvement", "31": "Refinance", "32": "Cash-out refi", "4": "Other", "5": "N/A"}
LT_SHORT = {"1": "Conv", "2": "FHA", "3": "VA", "4": "RHS"}


def product_label(lt, lp) -> str:
    return f"{LT_SHORT.get(str(lt), lt)} | {LOAN_PURPOSE.get(str(lp), lp)}"


def import_cns():
    try:
        import cnsplots as cns
    except Exception as e:   # noqa: BLE001
        raise RuntimeError("cnsplots 0.8.0 and Python 3.12 are required (3.13 cannot install it). See the README.") from e
    cns.settings.pvalue_format = "full"
    return cns


def new_multipanel(cns, width_mm: float = 183):
    return cns.multipanel(max_width=width_mm, unit="mm")


def save(cns, stem: Path, formats=("png", "pdf"), dpi: int = 200):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        cns.savefig(str(stem.with_suffix(f".{f}")), dpi=dpi) if f == "png" else cns.savefig(str(stem.with_suffix(f".{f}")))


def pct_axis(ax, which="y"):
    from matplotlib.ticker import PercentFormatter

    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


def errorbar_h(ax, y, x, lo, hi, color=C_MAIN, hollow=None, size=3.5):
    """Points + horizontal CI. hollow: boolean array marking small cells (n < threshold) with hollow markers."""
    y, x, lo, hi = map(np.asarray, (y, x, lo, hi))
    ax.hlines(y, lo, hi, color=color, lw=0.8)
    if hollow is None:
        ax.plot(x, y, "o", color=color, ms=size)
    else:
        hollow = np.asarray(hollow, bool)
        ax.plot(x[~hollow], y[~hollow], "o", color=color, ms=size)
        ax.plot(x[hollow], y[hollow], "o", mfc="white", mec=color, ms=size)


def boxes_from_percentiles(ax, df: pd.DataFrame, labels, positions=None, color=C_MAIN, horizontal=False):
    """Draw boxes from precomputed percentiles (p5, p25, p50, p75, p95): whiskers = p5-p95."""
    stats = [{"label": l, "whislo": r["p5"], "q1": r["p25"], "med": r["p50"], "q3": r["p75"], "whishi": r["p95"], "fliers": []}
             for l, (_, r) in zip(labels, df.iterrows())]
    ax.bxp(stats, positions=positions, vert=not horizontal, showfliers=False, patch_artist=True, widths=0.6,
           boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, lw=0.6),
           medianprops=dict(color="black", lw=0.8), whiskerprops=dict(color=color, lw=0.6), capprops=dict(color=color, lw=0.6))
