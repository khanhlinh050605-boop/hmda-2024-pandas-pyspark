"""Bước 04 - RQ Q1: denial rate theo loan type x purpose và state; mô hình có nhất quán giữa các state không?

Một "khối" đếm duy nhất (state x loan_type x purpose x lien) từ engine; mọi bảng còn lại là cộng gộp của khối này.
State "NA" (không cung cấp vị trí): giữ ở phân tích quốc gia, loại khỏi phân tích theo state (có hồ sơ riêng).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import stats as ST
from ._util import FINE, PRODUCT, NA_STATE, aggregate, raw_sum, reported_states, seg_label, to_counts, top_segments, with_rates

NAME = "s04_q1"
ENGINE_BOUND = True
SECTIONS = ["thresholds"]


def run(ctx):
    thr = ctx.cfg["thresholds"]
    small = thr["small_cell"]
    cube = to_counts(raw_sum(ctx, "q1_cube_state_seg", ["state_code"] + FINE))
    nat_seg = aggregate(cube, FINE)

    # --- quốc gia theo sản phẩm / phân khúc / loại / mục đích
    prod = with_rates(aggregate(cube, PRODUCT), small)
    prod["share_of_decided"] = prod["n_decided"] / prod["n_decided"].sum()
    ctx.save_table("q1_product_rates", prod)
    seg = with_rates(nat_seg, small)
    seg["segment"] = seg_label(seg)
    seg["share_of_decided"] = seg["n_decided"] / seg["n_decided"].sum()
    seg = seg.sort_values("n_decided", ascending=False)
    ctx.save_table("q1_segment_rates", seg)
    tops = top_segments(nat_seg, thr["top_segments"])
    ctx.save_table("q1_top_segments", pd.DataFrame({"rank": range(1, len(tops) + 1), "segment": tops}))
    ctx.save_table("q1_loan_type_rates", with_rates(aggregate(cube, ["loan_type"]), small))
    ctx.save_table("q1_loan_purpose_rates", with_rates(aggregate(cube, ["loan_purpose"]), small))
    nat_all = aggregate(cube, [])
    ctx.save_table("q1_national_total", with_rates(nat_all, small))

    # --- theo state (chỉ state có báo cáo)
    rep = reported_states(cube)
    st_prod = with_rates(aggregate(rep, ["state_code"] + PRODUCT), small)
    ctx.save_table("q1_state_product_rates", st_prod)
    ctx.save_table("q1_state_type_rates", with_rates(aggregate(rep, ["state_code", "loan_type"]), small))
    st_tot = with_rates(aggregate(rep, ["state_code"]), small)

    # --- nhất quán: Spearman giữa tỷ lệ state và quốc gia trên các ô tin cậy (n_decided >= small)
    nat_p = prod.set_index(PRODUCT)["denial_rate"]
    rows = []
    for st, g in st_prod.groupby("state_code"):
        g = g[g["n_decided"] >= small]
        x = nat_p.reindex(pd.MultiIndex.from_frame(g[PRODUCT])).to_numpy()
        rho, n = ST.spearman(x, g["denial_rate"].to_numpy())
        rows.append({"state_code": st, "n_reliable_products": n, "spearman_rho": rho})
    cons = pd.DataFrame(rows)
    cons["evaluated"] = (cons["n_reliable_products"] >= 3) & cons["spearman_rho"].notna()
    cons["consistent"] = (cons["evaluated"] & (cons["spearman_rho"] >= thr["state_rho"])).astype(int)
    ctx.save_table("q1_state_consistency", cons)
    ev = cons[cons["evaluated"]]
    k, n = int(ev["consistent"].sum()), int(len(ev))
    p, lo, hi = ST.wilson([k], [n])
    ctx.save_table("q1_state_consistency_summary", pd.DataFrame([{
        "states_reported": int(cons["state_code"].nunique()), "states_evaluated": n, "states_consistent": k,
        "share_consistent": p[0], "wilson_lo": lo[0], "wilson_hi": hi[0], "rho_threshold": thr["state_rho"],
        "median_rho": float(ev["spearman_rho"].median()) if n else np.nan,
        "min_rho": float(ev["spearman_rho"].min()) if n else np.nan}]))

    # --- phân tán giữa state trong từng sản phẩm (ô tin cậy)
    spread = []
    for key, g in st_prod[st_prod["n_decided"] >= small].groupby(PRODUCT):
        d = ST.dispersion(g["denial_rate"].values)
        spread.append({"loan_type": key[0], "loan_purpose": key[1], "national_rate": nat_p.get(key, np.nan), **d})
    ctx.save_table("q1_state_spread_by_product", pd.DataFrame(spread))

    # --- chuẩn hoá gián tiếp theo state (kiểm soát cơ cấu sản phẩm)
    seg_cols = FINE
    std = ST.indirect_standardise(rep, "state_code", seg_cols, ref=nat_seg)
    std = std.merge(st_tot[["state_code", "n_apps", "n_orig", "denial_rate", "denial_ci_lo", "denial_ci_hi", "small_cell"]],
                    on="state_code", how="left")
    nat_rate = nat_all["n_denied"].iloc[0] / nat_all["n_decided"].iloc[0]
    std["standardised_rate"] = std["oe_ratio"] * nat_rate
    std["rank_raw_rate"] = std["denial_rate"].rank(ascending=False, method="min")
    std["rank_standardised"] = std["oe_ratio"].rank(ascending=False, method="min")
    ctx.save_table("q1_state_oe", std.sort_values("oe_ratio", ascending=False))

    # --- hồ sơ state "NA"
    na = cube[cube["state_code"] == NA_STATE]
    if len(na):
        t = with_rates(aggregate(na, []), small)
        t.insert(0, "state_code", NA_STATE)
        ctx.save_table("q1_state_na_profile", t)
        ctx.save_table("q1_state_na_by_product", with_rates(aggregate(na, PRODUCT), small))
    return {"states": int(cons["state_code"].nunique()), "tops": tops}
