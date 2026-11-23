"""Hình 6 - Predictive extension: ROC, calibration, chỉ số, tầm quan trọng biến."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import C_ALT, C_GREY, C_HI, C_MAIN, new_multipanel, save


def build(cns, T, out, cfg):
    pred = T("pred_test_sample")
    met = T("pred_metrics")
    imp = T("pred_importance")
    models = [c for c in pred.columns if c != "y_true"]
    display = {
        "LR product-only": "LR product",
        "LR full, no lender profiles": "LR full - profiles",
        "LR full": "LR full",
        "Gradient boosting, no lender profiles": "GB - profiles",
        "Gradient boosting": "GB",
    }
    plot_pred = pred.rename(columns=display)
    plot_models = [display.get(m, m) for m in models]
    mp = new_multipanel(cns)
    ax = mp.panel("a", 42, 42)
    cns.rocplot(plot_pred, "y_true", plot_models)
    ax.legend(loc="lower right", frameon=False, fontsize=5.5, labelspacing=0.2,
              handlelength=1.3, handletextpad=0.4, borderaxespad=0.2)
    ax = mp.panel("b", 42, 42)
    cns.calibrationplot(plot_pred, "y_true", plot_models, n_bins=10, strategy="quantile")
    ax.legend(loc="upper left", frameon=False, fontsize=5.5, labelspacing=0.2,
              handlelength=1.3, handletextpad=0.4, borderaxespad=0.2)
    ax = mp.panel("c", 42, 42, margin_right=0)
    m = met[met["model"].isin(models)].reset_index(drop=True)
    ax.barh(range(len(m)), m["test_roc_auc"], color=C_MAIN, height=0.5)
    if "test_roc_auc_ci_lo" in m:
        ax.hlines(range(len(m)), m["test_roc_auc_ci_lo"], m["test_roc_auc_ci_hi"], color="black", lw=0.8)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m["model"].map(lambda x: display.get(x, x)), fontsize=5)
    lo = float(m["test_roc_auc_ci_lo"].min()) if "test_roc_auc_ci_lo" in m else float(m["test_roc_auc"].min())
    ax.set_xlim(min(0.45, lo - 0.02), 1.0)
    base = met[met["model"].str.startswith("Training-prior")]
    b_auc = float(base["test_roc_auc"].iloc[0]) if len(base) else 0.5
    ax.axvline(b_auc, color=C_HI, ls="--", lw=0.8, label=f"Training prior (AUC {b_auc:.1f})")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.3, handletextpad=0.4)
    ax.set_xlabel("Test ROC-AUC (95% lender-bootstrap interval)")
    ax = mp.panel("d", 90, 42)
    i = imp.head(10).iloc[::-1]
    ax.barh(range(len(i)), i["auc_drop_mean"], xerr=i["auc_drop_std"], color=C_ALT, height=0.6, error_kw=dict(lw=0.6))
    ax.set_yticks(range(len(i)))
    ax.set_yticklabels(i["feature"], fontsize=5)
    ax.set_xlabel("Drop in test ROC-AUC when feature is shuffled (gradient boosting)")
    save(cns, out / "fig6_predictive", cfg["figures"]["formats"], cfg["figures"]["dpi"])
    best = met[met["model"].isin(models)].sort_values("test_roc_auc", ascending=False).iloc[0]
    return {"title": "Application characteristics support retrospective denial-risk ranking for lenders not seen in training",
            "panels": "a, ROC curves on held-out lenders. b, calibration (deciles of predicted probability). c, test ROC-AUC. d, permutation importance.",
            "stats": "Lender-grouped train/validation/test split; ROC-AUC with lender-bootstrap interval; calibration by deciles; Brier score in panel b. "
                     "Analytical risk ranking, not a decision rule; no demographic inputs; inputs recorded differently by outcome removed.",
            "n_unit": f"n = {len(pred):,} test applications (sample of the test lenders); unit = decided application.",
            "observed": f"Best test ROC-AUC: {best['model']} = {best['test_roc_auc']:.3f}; base rate = {best.get('base_rate_test', np.nan):.3f}."}
