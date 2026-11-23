"""Sinh dữ liệu GIẢ LẬP cùng schema HMDA LAR (chỉ để TEST code, KHÔNG dùng để phân tích).

  python scripts/make_synthetic.py --out data/synthetic --rows 120000 --parts 6 --lenders 120

Tạo: <out>/lar/part-00000.parquet..., <out>/ts.zip (pipe), và 1 file *.lnk để kiểm tra việc bỏ qua file lạ.
Mọi cột là chuỗi như dữ liệu thật; có trạng thái NA/Exempt theo field, trùng lặp, lender có giá trị Exempt,
và cả các cột nhân khẩu học (để kiểm tra code KHÔNG đọc chúng).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXTRA_COLS = (
    ["activity_year", "derived_msa-md", "county_code", "census_tract", "conforming_loan_limit",
     "derived_loan_product_type", "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
     "purchaser_type", "preapproval", "loan_to_value_ratio", "hoepa_status", "total_points_and_fees",
     "origination_charges", "discount_points", "lender_credits", "prepayment_penalty_term", "intro_rate_period",
     "negative_amortization", "interest_only_payment", "balloon_payment", "other_nonamortizing_features",
     "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
     "multifamily_affordable_units", "applicant_credit_score_type", "co-applicant_credit_score_type",
     "applicant_ethnicity_observed", "co-applicant_ethnicity_observed", "applicant_race_observed",
     "co-applicant_race_observed", "applicant_sex", "co-applicant_sex", "applicant_sex_observed",
     "co-applicant_sex_observed", "applicant_age", "co-applicant_age", "applicant_age_above_62",
     "co-applicant_age_above_62", "submission_of_application", "initially_payable_to_institution",
     "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
     "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_one_to_four_family_homes",
     "tract_median_age_of_housing_units"]
    + [f"applicant_ethnicity-{i}" for i in range(1, 6)] + [f"co-applicant_ethnicity-{i}" for i in range(1, 6)]
    + [f"applicant_race-{i}" for i in range(1, 6)] + [f"co-applicant_race-{i}" for i in range(1, 6)]
    + [f"aus-{i}" for i in range(1, 6)] + [f"denial_reason-{i}" for i in range(1, 5)]
)
STATES = ["CA", "TX", "FL", "NY", "IL", "OH", "PA", "GA", "NC", "WA", "NA", "PR", "AZ", "CO"]


def make(rows: int, parts: int, lenders: int, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    leis = np.array([f"LEI{str(i).zfill(17)}" for i in range(lenders)])
    lender_p = rng.dirichlet(np.ones(lenders) * 0.4)
    lender_eff = rng.normal(0, 0.5, lenders)
    lenders_with_exempt_values = set(rng.choice(lenders, size=max(2, lenders // 20), replace=False).tolist())
    n = rows
    li = rng.choice(lenders, size=n, p=lender_p)
    loan_type = rng.choice(["1", "2", "3", "4"], size=n, p=[0.7, 0.15, 0.1, 0.05])
    loan_purpose = rng.choice(["1", "2", "31", "32", "4", "5"], size=n, p=[0.45, 0.1, 0.15, 0.15, 0.1, 0.05])
    lien = rng.choice(["1", "2"], size=n, p=[0.93, 0.07])
    occ = rng.choice(["1", "2", "3"], size=n, p=[0.88, 0.08, 0.04])
    cm = rng.choice(["1", "2"], size=n, p=[0.97, 0.03])
    units = rng.choice(["1", "2", "3", "4", "5-24"], size=n, p=[0.93, 0.04, 0.01, 0.01, 0.01])
    open_end = rng.choice(["1", "2"], size=n, p=[0.1, 0.9])
    biz = rng.choice(["1", "2"], size=n, p=[0.04, 0.96])
    rev = rng.choice(["1", "2"], size=n, p=[0.01, 0.99])
    state = rng.choice(STATES, size=n, p=np.array([0.2, 0.15, 0.1, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.03, 0.01, 0.06, 0.06]))
    income = np.exp(rng.normal(4.4, 0.6, n))               # nghìn USD
    loan_amt = np.round(np.exp(rng.normal(12.3, 0.6, n)) / 10000) * 10000 + 5000
    prop = np.round(loan_amt / rng.uniform(0.5, 1.0, n) / 5000) * 5000
    cltv = np.round(rng.normal(78, 14, n), 2)
    term = rng.choice([360, 240, 180, 120, 300], size=n, p=[0.8, 0.05, 0.1, 0.03, 0.02])
    dti = rng.choice(["<20%", "20%-<30%", "30%-<36%", "36", "38", "41", "44", "47", "50%-60%", ">60%", "NA"],
                     size=n, p=[0.08, 0.15, 0.17, 0.05, 0.06, 0.08, 0.08, 0.06, 0.1, 0.02, 0.15])
    lti = loan_amt / (income * 1000)
    base = -1.6 + 0.9 * (loan_type != "1") + 0.4 * (loan_purpose == "4") - 0.3 * (loan_purpose == "1") \
        + 0.8 * np.log(np.clip(lti, 0.2, 10)) - 0.015 * (income - 80) / 10 + lender_eff[li] + 0.02 * (cltv - 80)
    p_den = 1 / (1 + np.exp(-base))
    r = rng.random(n)
    act = np.where(r < 0.04, "6", np.where(r < 0.1, "4", np.where(r < 0.14, "5", np.where(r < 0.16, "7", "x"))))
    dec = act == "x"
    den = rng.random(n) < p_den
    act = np.where(dec, np.where(den, "3", np.where(rng.random(n) < 0.05, "2", "1")), act)
    rate = np.round(5.5 + 0.8 * (loan_type != "1") + 0.4 * (lien == "2") + 0.003 * (cltv - 80) + rng.normal(0, 0.5, n), 3)
    spread = np.round(rate - 6.1 + rng.normal(0, 0.2, n), 3)
    costs = np.round(np.abs(rng.normal(6000, 2500, n)), 2)
    df = pd.DataFrame({
        "activity_year": "2024", "lei": leis[li], "state_code": state, "action_taken": act,
        "loan_type": loan_type, "loan_purpose": loan_purpose, "lien_status": lien, "occupancy_type": occ,
        "construction_method": cm, "total_units": units, "open-end_line_of_credit": open_end,
        "business_or_commercial_purpose": biz, "reverse_mortgage": rev,
        "loan_amount": loan_amt.astype(int).astype(str), "income": np.round(income).astype(int).astype(str),
        "property_value": prop.astype(int).astype(str), "loan_term": term.astype(str),
        "combined_loan_to_value_ratio": cltv.astype(str), "debt_to_income_ratio": dti,
        "interest_rate": rate.astype(str), "rate_spread": spread.astype(str), "total_loan_costs": costs.astype(str),
    })
    orig = df["action_taken"] != "1"
    for c in ("interest_rate", "rate_spread", "total_loan_costs"):
        df.loc[orig, c] = "NA"
    miss = rng.random(n)
    df.loc[miss < 0.04, "income"] = "NA"
    df.loc[(miss > 0.04) & (miss < 0.05), "income"] = "-5"          # không hợp lệ
    df.loc[(miss > 0.05) & (miss < 0.052), "loan_amount"] = "0"     # không hợp lệ
    df.loc[(miss > 0.06) & (miss < 0.07), "property_value"] = "NA"
    df.loc[(miss > 0.07) & (miss < 0.0705), "income"] = "9999999"   # cực đoan
    # Regression fixture: two spellings of the same valid income.
    df.loc[0, "income"] = "1111"
    df.loc[1, "income"] = "1111.0"
    # Lenders with at least one field reported Exempt (not a legal lender-exemption classification).
    ex_mask = np.isin(li, list(lenders_with_exempt_values))
    for c in ("interest_rate", "rate_spread", "total_loan_costs", "loan_term", "property_value",
              "combined_loan_to_value_ratio", "debt_to_income_ratio"):
        df.loc[ex_mask, c] = "Exempt"
    df.loc[ex_mask, "open-end_line_of_credit"] = "1111"
    df.loc[ex_mask, "business_or_commercial_purpose"] = "1111"
    # cột phụ (nhân khẩu học v.v.) để thử hàng rào
    for c in EXTRA_COLS:
        if c not in df.columns:
            if "census_tract" == c:
                df[c] = rng.integers(0, 3000, n).astype(str)
            elif "county_code" == c:
                df[c] = rng.integers(1000, 1100, n).astype(str)
            elif c.startswith("applicant_age") or c.startswith("co-applicant_age"):
                df[c] = rng.choice(["25-34", "35-44", "45-54", "55-64", "8888"], n)
            elif "race" in c or "ethnicity" in c or "sex" in c:
                df[c] = rng.choice(["1", "2", "3", "5", "6"], n)
            else:
                df[c] = rng.choice(["1", "2", "NA", "1111"], n)
    # trùng lặp chính xác: sao chép 0.5% dòng
    dup_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return np.array_split(df, parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--lenders", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    lar = out / "lar"
    lar.mkdir(parents=True, exist_ok=True)
    chunks = make(a.rows, a.parts, a.lenders, a.seed)
    all_df = []
    for i, ch in enumerate(chunks):
        t = pa.Table.from_pandas(ch.reset_index(drop=True).astype(str), preserve_index=False)
        pq.write_table(t, lar / f"part-{i:05d}.parquet")
        all_df.append(ch)
    (lar / "part-00000.parquet - Shortcut.lnk").write_text("not a parquet file")
    full = pd.concat(all_df)
    ts = full.groupby("lei").size().rename("lar_count").reset_index()
    ts["activity_year"] = "2024"
    ts["calendar_quarter"] = "4"
    ts["tax_id"] = "00-0000000"
    ts["agency_code"] = np.random.default_rng(1).choice(["1", "2", "3", "5", "7", "9"], len(ts))
    ts["respondent_name"] = "BANK " + ts["lei"].str[-4:]
    ts["respondent_state"] = "NY"
    ts["respondent_city"] = "NEW YORK"
    ts["respondent_zip_code"] = "10001"
    cols = ["activity_year", "calendar_quarter", "lei", "tax_id", "agency_code", "respondent_name",
            "respondent_state", "respondent_city", "respondent_zip_code", "lar_count"]
    with zipfile.ZipFile(out / "ts.zip", "w") as z:
        z.writestr("2024_public_ts_one_year_pipe.txt", ts[cols].to_csv(sep="|", index=False))
    print(f"OK: {len(full)} dòng, {len(ts)} lender -> {out}")


if __name__ == "__main__":
    main()
