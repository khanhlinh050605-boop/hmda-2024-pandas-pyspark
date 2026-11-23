"""Định nghĩa dùng chung cho cả hai engine: band, nhãn, miền hợp lý, quy tắc where.

Mọi định nghĩa chỉ nằm ở đây (và config.yaml) để Pandas và Spark không thể lệch nhau.
"""
from __future__ import annotations

from .schema import MISSING

STATUS_BANDS = ["~NA", "~EXEMPT", "~EMPTY", "~UNPARSABLE", "~OUTSIDE_DOMAIN"]

# DTI dạng band chữ trong HMDA (sau khi bỏ '%' và khoảng trắng) -> nhãn chuẩn
DTI_TEXT_MAP = {"<20": "<20", "20-<30": "20-<30", "30-<36": "30-<36", "50-60": "50-60", ">60": ">60"}


def dti_band_numpy(values):
    """Map numeric DTI percentages; 50-60 is closed at 60 per the HMDA bins."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    return np.select(
        [v < 20, v < 30, v < 36, v < 43, v < 50, v <= 60],
        ["<20", "20-<30", "30-<36", "36-<43", "43-<50", "50-60"],
        default=">60",
    ).astype(object)

# Biến Q3 -> (cột band trong master, cột giá trị, tên hiển thị)
Q3_VARS = {
    "income":         ("income_band", "income", "Income (USD thousand)"),
    "loan_amount":    ("loan_amount_band", "loan_amount", "Loan amount (USD)"),
    "lti":            ("lti_band", "lti", "Loan-to-income ratio (x)"),
    "dti":            ("dti_band", None, "Debt-to-income ratio (%)"),
    "cltv":           ("cltv_band", "cltv", "Combined LTV (%)"),
    "property_value": ("property_value_band", "property_value", "Property value (USD)"),
}


def _fmt(v: float, unit: str) -> str:
    if unit == "usd":
        if v >= 1_000_000 and v % 1_000_000 == 0:
            return f"{int(v // 1_000_000)}M"
        if v >= 1000 and v % 1000 == 0:
            return f"{int(v // 1000)}k"
    return f"{v:g}"


def make_labels(edges, unit: str = "plain") -> list[str]:
    e = [_fmt(x, unit) for x in edges]
    labels = [f"<{e[0]}"]
    for a, b in zip(e[:-1], e[1:]):
        labels.append(f"{a}-<{b}")
    labels.append(f">={e[-1]}")
    return labels


class Defs:
    """Đọc config và dựng toàn bộ nhãn, thứ tự band."""

    def __init__(self, cfg):
        self.cfg = cfg
        b = cfg["bands"]
        self.edges = {
            "income_band": list(b["income_k"]),
            "loan_amount_band": list(b["loan_amount"]),
            "lti_band": list(b["lti"]),
            "cltv_band": list(b["cltv"]),
            "property_value_band": list(b["property_value"]),
            "dti_numeric": list(b["dti_numeric"]),
        }
        units = {"loan_amount_band": "usd", "property_value_band": "usd"}
        self.labels = {k: make_labels(v, units.get(k, "plain")) for k, v in self.edges.items()
                       if k != "dti_numeric"}
        self.dti_labels = ["<20", "20-<30", "30-<36", "36-<43", "43-<50", "50-60", ">60"]
        self.labels["dti_band"] = self.dti_labels
        self.order = {k: v + STATUS_BANDS for k, v in self.labels.items()}
        self.plaus = cfg["plausibility"]
        self.thr = cfg["thresholds"]

    def is_status_band(self, label: str) -> bool:
        return str(label).startswith("~")

    def order_of(self, band_col: str) -> list[str]:
        return self.order[band_col]

    def valid_labels(self, band_col: str) -> list[str]:
        return self.labels[band_col]


def in_range_spec(spec: dict) -> tuple[float, float, bool]:
    return float(spec["lo"]), float(spec["hi"]), bool(spec.get("lo_inclusive", False))


def normalize_where(where) -> list[tuple]:
    """where: danh sách phần tử là tên cột cờ (==1) hoặc (cột, toán_tử, giá_trị)."""
    out = []
    for w in where or []:
        if isinstance(w, str):
            out.append((w, "==", 1))
        else:
            c, op, v = w
            out.append((c, op, v))
    return out


def resolve_where(where, ctx_lists: dict) -> list[tuple]:
    """Thay '@tên' bằng danh sách thật (vd. @lenders_with_exempt_values)."""
    res = []
    for c, op, v in normalize_where(where):
        if isinstance(v, str) and v.startswith("@"):
            v = list(ctx_lists[v[1:]])
        res.append((c, op, v))
    return res


__all__ = ["Defs", "Q3_VARS", "STATUS_BANDS", "DTI_TEXT_MAP", "dti_band_numpy", "MISSING", "make_labels",
           "in_range_spec", "normalize_where", "resolve_where"]
