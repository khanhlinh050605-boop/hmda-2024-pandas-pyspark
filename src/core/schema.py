"""Schema: cột đọc từ LAR, mã đặc biệt, và hàng rào chặn biến nhân khẩu học / sau quyết định.

QUY TẮC (đã chốt): code KHÔNG đọc applicant_*, co_applicant_*, derived_ethnicity/race/sex,
tract_minority_*, và các trường sau quyết định (denial_reason_*, hoepa_status, purchaser_type...).
Mọi nơi đọc cột từ nguồn đều đi qua `resolve_columns`, nên hàng rào này luôn được áp dụng.
"""
from __future__ import annotations

import re

MISSING = "__MISSING__"
SEP = "|"

# --- cột đọc từ LAR (tên chuẩn) -------------------------------------------------
CATEGORICAL_RAW = [
    "lei", "action_taken", "state_code", "loan_type", "loan_purpose", "lien_status",
    "occupancy_type", "construction_method", "total_units",
    "open_end_line_of_credit", "business_or_commercial_purpose", "reverse_mortgage",
]
# Quantitative fields.  Special values are field-specific; a numeric-looking
# token is never treated as a status merely because another HMDA field uses it
# as a code.  In particular, income="1111" is USD 1.111 million (income is
# reported in thousands), not an exemption code.
NUMERIC_RAW = {
    "loan_amount": "loan_amount",
    "income": "income",
    "property_value": "property_value",
    "loan_term": "loan_term",
    "interest_rate": "interest_rate",
    "rate_spread": "rate_spread",
    "total_loan_costs": "total_loan_costs",
    "cltv": "combined_loan_to_value_ratio",   # tên chuẩn -> tên gốc trong LAR
}
DTI_RAW = "debt_to_income_ratio"              # dạng band chuỗi hoặc số

# Allowed non-numeric tokens in the public 2024 LAR/FIG.  Loan amount is always
# numeric.  Income permits NA but not Exempt.  The remaining reported numeric
# fields may contain NA and/or Exempt.  1111/8888/9999 are numeric values for
# these fields and are subsequently assessed against the analysis domain.
FIELD_SPECIAL_VALUES = {
    "loan_amount": {},
    "income": {"NA": "NA"},
    "property_value": {"NA": "NA", "EXEMPT": "EXEMPT"},
    "loan_term": {"NA": "NA", "EXEMPT": "EXEMPT"},
    "interest_rate": {"NA": "NA", "EXEMPT": "EXEMPT"},
    "rate_spread": {"NA": "NA", "EXEMPT": "EXEMPT"},
    "total_loan_costs": {"NA": "NA", "EXEMPT": "EXEMPT"},
    "cltv": {"NA": "NA", "EXEMPT": "EXEMPT"},
    "dti": {"NA": "NA", "EXEMPT": "EXEMPT"},
}

READ_COLUMNS = CATEGORICAL_RAW + list(NUMERIC_RAW.values()) + [DTI_RAW]

# --- hàng rào ---------------------------------------------------------------------
FORBIDDEN_RE = re.compile(
    r"^(applicant_|co_applicant_|derived_(ethnicity|race|sex)|denial_reason|tract_minority"
    r"|hoepa_status|purchaser_type|aus_|applicant$)"
)


def norm_name(name: str) -> str:
    """'open-end_line_of_credit' và 'open_end_line_of_credit' cùng một tên chuẩn."""
    return name.strip().lower().replace("-", "_")


def assert_no_forbidden(columns) -> None:
    bad = [c for c in columns if FORBIDDEN_RE.search(norm_name(c))]
    if bad:
        raise RuntimeError(f"Cột bị cấm (nhân khẩu học / sau quyết định) bị yêu cầu đọc: {bad}")


def resolve_columns(available) -> dict[str, str]:
    """Trả về {tên_chuẩn: tên_thực_trong_file} cho đúng các cột cần đọc. Thiếu cột -> báo lỗi rõ."""
    assert_no_forbidden(READ_COLUMNS)
    lut = {norm_name(c): c for c in available}
    out, missing = {}, []
    for want in READ_COLUMNS:
        if norm_name(want) in lut:
            out[want] = lut[norm_name(want)]
        else:
            missing.append(want)
    if missing:
        raise KeyError(f"LAR thiếu cột bắt buộc: {missing}")
    assert_no_forbidden(out.values())
    return out


# --- trạng thái của giá trị số ------------------------------------------------------
STATUS_VALUE = "VALUE"
STATUS_ORDER = ["VALUE", "NA", "EXEMPT", "EMPTY", "UNPARSABLE"]
NUM_RE = r"^-?[0-9]+(\.[0-9]+)?$"
