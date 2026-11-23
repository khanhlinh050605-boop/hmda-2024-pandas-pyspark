"""Step 11 - Predictive extension (proposal section 8; brief section 4).

TARGET AND INTENDED USE
  Estimate the probability of denial (action_taken = 3) among DECIDED applications (1,2,3), excluding exempt records,
  consumer purpose, closed-end loans, not reverse mortgages. Used for an analytical RISK RANKING
  and to measure the dispersion between lenders that remains after accounting for application mix.
  It is NOT a lending decision rule and NOT a causal claim.
INPUT VARIABLES (retrospective analysis; not assumed to be all available at intake): loan type, purpose, lien, occupancy, construction, number of units,
  DTI band, log loan amount, log income, CLTV, term, log property value, state, lender profile (size, product mix).
NOT USED: age/sex/race/ethnicity (never read), and post-decision fields (denial_reason, interest_rate,
  rate_spread, loan costs, purchaser_type, hoepa_status). Variables whose "recording differs by outcome" are dropped (leakage check).
EVALUATION: train/val/test split BY LENDER (measures generalisation to unseen lenders);
  ROC-AUC, average precision (AP), log-loss, Brier, calibration; LR is compared with gradient boosting and a product-only baseline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..core import schema as S
from ..core import stats as ST
from ..core.runtime import Ctx, log

NAME = "s11_predictive"
ENGINE_BOUND = False
SECTIONS = ["predictive", "thresholds", "project"]

LOAD = ["lei", "state_code", "loan_type", "loan_purpose", "lien_status", "occupancy_type", "construction_method",
        "total_units", "dti_band", "is_decided", "is_denied", "has_reported_exempt_value", "is_consumer_cf",
        "log_loan_amount", "log_income", "cltv", "term_years", "log_property_value", "sample_u"]


def _missing_share(df: pd.DataFrame, feat: str, is_cont: bool) -> pd.Series:
    if is_cont:
        return df[feat].isna()
    s = df[feat].astype(str)
    return s.str.startswith("~") | (s == S.MISSING)


def _lender_split(lei: pd.Series, seed: int, test: float, val: float, min_n: int) -> dict[str, str]:
    cnt = lei.value_counts()
    cnt = cnt[cnt >= min_n]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cnt))
    ids, sizes = cnt.index.to_numpy()[order], cnt.to_numpy()[order]
    tot = sizes.sum()
    cum = np.cumsum(sizes) / tot
    out = {}
    for l, c in zip(ids, cum):
        out[l] = "test" if c <= test else ("val" if c <= test + val else "train")
    return out


def _bootstrap_auc(y, p, groups, n_boot, seed):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx_by = [np.where(codes == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by[i] for i in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(roc_auc_score(y[ii], p[ii]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def run(ctx):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    from ..engines.pandas_engine import PandasEngine

    cfg = ctx.cfg
    pc = cfg["predictive"]
    seed = int(cfg["project"]["seed"])
    eng = PandasEngine(cfg)
    pctx = Ctx(cfg, eng, "pandas")
    eng.open(pctx)
    S.assert_no_forbidden(LOAD)
    df = pd.DataFrame({c: eng.col(c) for c in LOAD})
    ex_lei = set(pctx.lists.get("lenders_with_exempt_values", []))
    pop = ((df["is_decided"] == 1) & (df["has_reported_exempt_value"] == 0) & (df["is_consumer_cf"] == 1)
           & ~df["lei"].astype(str).isin(ex_lei))
    pop_n = int(pop.sum())
    df = df[pop].copy()
    for c in ("lei", "state_code", "loan_type", "loan_purpose", "lien_status", "occupancy_type", "construction_method",
              "total_units", "dti_band"):
        df[c] = df[c].astype(str)
    df["y"] = df["is_denied"].astype(int)
    pd.DataFrame([{"step": "all records", "n": len(eng.col("lei"))},
                  {"step": "decided (1,2,3)", "n": int((eng.col("is_decided") == 1).sum())},
                  {"step": "no reported Exempt value; lender has no such record; consumer, closed-end, forward", "n": pop_n}]
                 ).to_csv(ctx.tab_dir / "pred_population.csv", index=False)

    # ---- split by lender before any outcome-conditioned feature audit
    split = _lender_split(df["lei"], seed, pc["test_size"], pc["val_size"], pc["min_lender_decided"])
    df["split"] = df["lei"].map(split)
    df = df[df["split"].notna()].reset_index(drop=True)

    # ---- retrospective lender profiles (no outcomes).  They use the full-year
    # application mix of each lender inside its split and are therefore not
    # claimed to be available at an individual application's intake time.
    g = df.groupby("lei")
    prof = pd.DataFrame({"lender_log_apps": np.log(g.size())})
    for lt in ("1", "2"):
        prof[f"lender_share_loan_type_{lt}"] = g["loan_type"].apply(lambda s, lt=lt: (s == lt).mean())
    for lp in ("1", "31", "32", "4"):
        prof[f"lender_share_purpose_{lp}"] = g["loan_purpose"].apply(lambda s, lp=lp: (s == lp).mean())
    prof = prof[[c for c in pc["lender_profile"] if c in prof.columns]]
    df = df.join(prof, on="lei")

    # ---- deterministic row sample after split and profile construction
    frac = min(1.0, pc["sample_rows"] / max(len(df), 1))
    df = df[df["sample_u"] < frac].reset_index(drop=True)

    cat = [c for c in pc["categorical"]]
    base_cont = [c for c in pc["continuous"]]

    # ---- train-only audit: test outcomes never influence feature retention
    train_audit = df[df["split"] == "train"]
    rows = []
    for f in cat + base_cont:
        ms = _missing_share(train_audit, f, f in base_cont)
        den, app = ms[train_audit["y"] == 1].mean(), ms[train_audit["y"] == 0].mean()
        gap = float(den - app)
        rows.append({"feature": f, "share_missing_or_status_denied": den, "share_missing_or_status_approved": app,
                     "gap": gap, "overall_missing": float(ms.mean()),
                     "flag_recorded_differently_by_outcome": int(abs(gap) > pc["leakage_gap"]),
                     "flag_nearly_empty": int(ms.mean() > 0.95)})
    audit = pd.DataFrame(rows)
    audit["dropped"] = ((audit["flag_recorded_differently_by_outcome"] == 1) | (audit["flag_nearly_empty"] == 1)).astype(int)
    ctx.save_table("pred_leakage_audit", audit)
    drop = set(audit.loc[audit["dropped"] == 1, "feature"])
    cat = [c for c in cat if c not in drop]
    base_cont = [c for c in base_cont if c not in drop]
    profile_cols = list(prof.columns)
    cont = base_cont + profile_cols
    log.info("[s11] leakage audit: dropped %s", sorted(drop))

    sp = df.groupby("split").agg(rows=("y", "size"), lenders=("lei", "nunique"), denial_rate=("y", "mean")).reset_index()
    ctx.save_table("pred_split_summary", sp)
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) == 0 or tr["y"].nunique() < 2 or te["y"].nunique() < 2:
        raise RuntimeError("The lender-level split produced an empty or single-class set: lower min_lender_decided or check the data.")
    feats = cat + cont
    no_profile_feats = cat + base_cont
    Xtr, Xva, Xte = tr[feats], va[feats], te[feats]
    ytr, yva, yte = tr["y"].values, va["y"].values, te["y"].values

    def prep_onehot(cats, conts):
        parts = []
        if cats:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats))
        if conts:
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                           ("sc", StandardScaler())]), conts))
        return ColumnTransformer(parts, sparse_threshold=0.3)

    def prep_ordinal(cats, conts):
        parts = []
        if cats:
            parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                encoded_missing_value=-1), cats))
        if conts:
            parts.append(("num", "passthrough", conts))
        return ColumnTransformer(parts)

    models, model_features, sel_rows = {}, {}, []

    def fit_eval(name, make, grid, Xa, Xb, features):
        best = None
        for params in grid:
            m = make(**params).fit(Xa, ytr)
            pv = m.predict_proba(Xb)[:, 1]
            ll = log_loss(yva, pv, labels=[0, 1])
            sel_rows.append({"model": name, "params": json.dumps(params), "val_log_loss": ll,
                             "val_roc_auc": roc_auc_score(yva, pv)})
            if best is None or ll < best[0]:
                best = (ll, params, m)
        models[name] = best[2]
        model_features[name] = list(features)
        return best

    prod_cols = [c for c in ("loan_type", "loan_purpose", "lien_status") if c in cat]
    fit_eval("LR product-only", lambda C: Pipeline([("p", prep_onehot(prod_cols, [])),
                                                    ("m", LogisticRegression(C=C, max_iter=400))]),
             [{"C": 1.0}], Xtr[prod_cols], Xva[prod_cols], prod_cols)
    fit_eval("LR full, no lender profiles",
             lambda C: Pipeline([("p", prep_onehot(cat, base_cont)), ("m", LogisticRegression(C=C, max_iter=400))]),
             [{"C": c} for c in (0.1, 1.0, 10.0)], Xtr[no_profile_feats], Xva[no_profile_feats], no_profile_feats)
    fit_eval("LR full", lambda C: Pipeline([("p", prep_onehot(cat, cont)), ("m", LogisticRegression(C=C, max_iter=400))]),
             [{"C": c} for c in (0.1, 1.0, 10.0)], Xtr, Xva, feats)
    cat_mask_no_profile = [True] * len(cat) + [False] * len(base_cont)
    fit_eval("Gradient boosting, no lender profiles", lambda lr, it, leaves: Pipeline([
        ("p", prep_ordinal(cat, base_cont)),
        ("m", HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, random_state=seed,
                                             categorical_features=cat_mask_no_profile if cat else None))]),
             [{"lr": 0.1, "it": 100, "leaves": 31}, {"lr": 0.1, "it": 200, "leaves": 31},
              {"lr": 0.05, "it": 300, "leaves": 63}],
             Xtr[no_profile_feats], Xva[no_profile_feats], no_profile_feats)
    cat_mask = [True] * len(cat) + [False] * len(cont)
    fit_eval("Gradient boosting", lambda lr, it, leaves: Pipeline([
        ("p", prep_ordinal(cat, cont)),
        ("m", HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, random_state=seed,
                                             categorical_features=cat_mask if cat else None))]),
             [{"lr": 0.1, "it": 100, "leaves": 31}, {"lr": 0.1, "it": 200, "leaves": 31}, {"lr": 0.05, "it": 300, "leaves": 63}],
             Xtr, Xva, feats)
    ctx.save_table("pred_model_selection", pd.DataFrame(sel_rows))

    # ---- evaluation on TEST (unseen lenders)
    prior = float(ytr.mean())
    out, calib, roc_pts, preds = [], [], [], {"y_true": yte}
    base_ll = log_loss(yte, np.full(len(yte), prior), labels=[0, 1])
    out.append({"model": "Training-prior (no features)", "test_log_loss": base_ll, "test_roc_auc": 0.5,
                "test_pr_auc": float(yte.mean()), "test_brier": brier_score_loss(yte, np.full(len(yte), prior))})
    for name, m in models.items():
        sub = model_features[name]
        p = m.predict_proba(te[sub])[:, 1]
        preds[name] = p
        lo, hi = _bootstrap_auc(yte, p, te["lei"].values, int(pc.get("bootstrap", 100)), seed)
        out.append({"model": name, "test_log_loss": log_loss(yte, p, labels=[0, 1]), "test_roc_auc": roc_auc_score(yte, p),
                    "test_roc_auc_ci_lo": lo, "test_roc_auc_ci_hi": hi, "test_pr_auc": average_precision_score(yte, p),
                    "test_brier": brier_score_loss(yte, p), "base_rate_test": float(yte.mean())})
        q = pd.qcut(pd.Series(p), 10, duplicates="drop")
        cb = pd.DataFrame({"p": p, "y": yte, "bin": q}).groupby("bin", observed=True).agg(
            mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
        cb.insert(0, "model", name)
        calib.append(cb)
        fpr, tpr, _ = roc_curve(yte, p)
        k = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        roc_pts.append(pd.DataFrame({"model": name, "fpr": fpr[k], "tpr": tpr[k]}))
    ctx.save_table("pred_metrics", pd.DataFrame(out))
    ctx.save_table("pred_calibration", pd.concat(calib, ignore_index=True))
    ctx.save_table("pred_roc_points", pd.concat(roc_pts, ignore_index=True))
    n_s = min(len(yte), 150_000)
    ii = np.random.default_rng(seed).choice(len(yte), n_s, replace=False)
    pd.DataFrame({k: v[ii] for k, v in preds.items()}).to_csv(ctx.tab_dir / "pred_test_sample.csv", index=False)

    # ---- ablation: is the model exploiting whether a field was recorded?  On the same complete test set (no missing
    # CLTV / property value / income / DTI band), compare the model trained on all rows with a model retrained only on
    # complete rows.  If AUC drops sharply, part of the predictive power comes from missingness rather than from the values.
    from sklearn.base import clone

    def _complete(d):
        ok = d[base_cont].notna().all(axis=1)
        if "dti_band" in cat:
            ok &= ~d["dti_band"].astype(str).str.startswith("~")
        return ok.to_numpy()

    ctr, cte = _complete(tr), _complete(te)
    abl = []
    if ctr.sum() > 1000 and cte.sum() > 1000 and len(np.unique(yte[cte])) == 2 and len(np.unique(ytr[ctr])) == 2:
        for name in ("Gradient boosting, no lender profiles", "LR full, no lender profiles"):
            sub = model_features[name]
            full_p = models[name].predict_proba(te.loc[cte, sub])[:, 1]
            refit = clone(models[name]).fit(tr.loc[ctr, sub], ytr[ctr])
            re_p = refit.predict_proba(te.loc[cte, sub])[:, 1]
            abl.append({"model": name, "complete_test_rows": int(cte.sum()), "complete_train_rows": int(ctr.sum()),
                        "share_of_test_complete": float(cte.mean()), "base_rate_complete_test": float(yte[cte].mean()),
                        "auc_trained_on_all_rows": roc_auc_score(yte[cte], full_p),
                        "auc_trained_on_complete_rows": roc_auc_score(yte[cte], re_p),
                        "auc_all_test_rows": roc_auc_score(yte, models[name].predict_proba(te[sub])[:, 1])})
    ctx.save_table("pred_ablation_complete_case", pd.DataFrame(abl, columns=[
        "model", "complete_test_rows", "complete_train_rows", "share_of_test_complete", "base_rate_complete_test",
        "auc_trained_on_all_rows", "auc_trained_on_complete_rows", "auc_all_test_rows"]))

    # ---- permutation importance of the gradient-boosting model
    gb = models["Gradient boosting"]
    ns = min(len(te), 40_000)
    ti = np.random.default_rng(seed + 1).choice(len(te), ns, replace=False)
    pi = permutation_importance(gb, Xte[feats].iloc[ti], yte[ti], scoring="roc_auc", n_repeats=3, random_state=seed)
    ctx.save_table("pred_importance", pd.DataFrame({"feature": feats, "auc_drop_mean": pi.importances_mean,
                                                    "auc_drop_std": pi.importances_std}).sort_values("auc_drop_mean", ascending=False))

    # ---- unusual lenders after accounting for application mix: O/E from predicted probabilities (test lenders only)
    pg = preds["Gradient boosting"]
    lt = pd.DataFrame({"lei": te["lei"].values, "y": yte, "p": pg}).groupby("lei").agg(
        n_decided=("y", "size"), observed=("y", "sum"), expected=("p", "sum")).reset_index()
    lt["oe_ratio"] = lt["observed"] / lt["expected"]
    lo, hi = ST.garwood(lt["observed"].values)
    lt["oe_lo"], lt["oe_hi"] = lo / lt["expected"], hi / lt["expected"]
    lt["unusual_high"] = (lt["oe_lo"] > 1).astype(int)
    lt["unusual_low"] = (lt["oe_hi"] < 1).astype(int)
    lt["eligible"] = (lt["n_decided"] >= cfg["thresholds"]["lender_min_decided"]).astype(int)
    ctx.save_table("pred_lender_oe", lt.sort_values("oe_ratio", ascending=False))

    m = pd.DataFrame(out)
    summary = {"population_n": pop_n, "rows_used": int(len(df)), "features_kept": feats, "features_dropped_leakage": sorted(drop),
               "profile_features": profile_cols, "profile_timing": "retrospective full-year lender mix within each split",
               "split_rows": {r.split: int(r.rows) for r in sp.itertuples()},
               "test_roc_auc": {r.model: r.test_roc_auc for r in m.itertuples()}}
    (ctx.out / "pred_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (ctx.out / "pred_notes.md").write_text(_notes(pc, summary))
    eng.close()
    return summary


def _notes(pc, summary) -> str:
    return f"""# Predictive extension: target, inputs, evaluation and limits

**Target and intended use.** Probability that a decided application is denied (action_taken = 3 versus 1 or 2), restricted to
non-exempt, consumer-purpose, closed-end, forward-mortgage applications. Intended use: analytical risk ranking and measuring how
much lender-level variation remains after application mix. It is not a lending decision rule and not a causal claim.

**Inputs and timing.** {', '.join(summary['features_kept'])}.
The lender-profile features use the lender's full-year application mix and size within its split, never outcomes. Because that
full-year profile is not necessarily available when an individual application arrives, the stated use is retrospective risk ranking,
not intake-time scoring. The no-profile ablations and product-only logistic model are reported separately.

**Excluded by design.** Applicant age, sex, race and ethnicity are never read. Post-decision fields (denial reasons, interest rate,
rate spread, loan costs, purchaser type, HOEPA status) are excluded. Inputs recorded differently by outcome (gap above
{pc['leakage_gap']:.0%} in missing/status share between denied and approved) or nearly empty are dropped:
{', '.join(summary['features_dropped_leakage']) or 'none'}.

**Evaluation.** Lenders are assigned to train / validation / test before the train-only missingness audit. Test labels therefore do
not affect feature retention or model selection. Model selection uses validation log-loss; reported metrics are on the test lenders:
ROC-AUC (with lender-bootstrap 95% interval),
average precision (AP), log-loss, Brier score and a decile calibration table. Models: logistic regression and histogram gradient boosting,
plus no-profile ablations, a product-only logistic baseline and the training-prior baseline.

**Limitations.** No credit score and no application date are available; HMDA public data are one year; loan amount is rounded; DTI
is often banded; CLTV/property value/DTI availability may reflect the underwriting process rather than intake-time information; observed
associations are not causal. Because the lender-level split removes lender identity, differences between
lenders show up as residual O/E, which describes unexplained variation, not misconduct. A high AUC would not justify use on individual
borrowers; a low AUC would reflect missing variables, not absence of risk differences.
"""
