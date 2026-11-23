"""Schema: columns read from the LAR, special codes, and the guard that blocks demographic / post-decision variables.

RULE (final): the code NEVER reads applicant_*, co_applicant_*, derived_ethnicity/race/sex,
tract_minority_*, or post-decision fields (denial_reason_*, hoepa_status, purchaser_type...).
Every place that reads source columns goes through `resolve_columns`, so this guard is always applied.
"""
from __future__ import annotations

import re

MISSING = "__MISSING__"
SEP = "|"

# --- columns read from the LAR (canonical names) -------------------------------------------------
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
    "cltv": "combined_loan_to_value_ratio",   # canonical name -> original name in the LAR
}
DTI_RAW = "debt_to_income_ratio"              # string band or number

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

# --- guard ---------------------------------------------------------------------
FORBIDDEN_RE = re.compile(
    r"^(applicant_|co_applicant_|derived_(ethnicity|race|sex)|denial_reason|tract_minority"
    r"|hoepa_status|purchaser_type|aus_|applicant$)"
)


def norm_name(name: str) -> str:
    """'open-end_line_of_credit' and 'open_end_line_of_credit' map to the same canonical name."""
    return name.strip().lower().replace("-", "_")


def assert_no_forbidden(columns) -> None:
    bad = [c for c in columns if FORBIDDEN_RE.search(norm_name(c))]
    if bad:
        raise RuntimeError(f"Forbidden column (demographic / post-decision) requested: {bad}")


def resolve_columns(available) -> dict[str, str]:
    """Return {canonical_name: actual_name_in_file} for exactly the columns to read. A missing column raises a clear error."""
    assert_no_forbidden(READ_COLUMNS)
    lut = {norm_name(c): c for c in available}
    out, missing = {}, []
    for want in READ_COLUMNS:
        if norm_name(want) in lut:
            out[want] = lut[norm_name(want)]
        else:
            missing.append(want)
    if missing:
        raise KeyError(f"LAR is missing required columns: {missing}")
    assert_no_forbidden(out.values())
    return out


# --- status of numeric values ------------------------------------------------------
STATUS_VALUE = "VALUE"
STATUS_ORDER = ["VALUE", "NA", "EXEMPT", "EMPTY", "UNPARSABLE"]
NUM_RE = r"^-?[0-9]+(\.[0-9]+)?$"
