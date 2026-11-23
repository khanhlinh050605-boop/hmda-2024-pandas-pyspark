"""Print a breakdown of values OUTSIDE the plausible domain in the raw LAR, to decide whether the domain wrongly excludes valid values.

Run:  python scripts/check_domain_values.py            (reads config.yaml, data/raw/2024_one_year_parquet; another config path can be passed)
Writes nothing to outputs; only prints to the screen (and scripts/domain_values_report.txt).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402

from src.core import schema as S  # noqa: E402
from src.core.config import load_config  # noqa: E402

FIELDS = ["income", "interest_rate", "property_value", "loan_term", "rate_spread", "cltv", "loan_amount"]


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml")
    d = cfg.path("lar_dir")
    files = sorted(d.glob("*.parquet"))
    plaus = cfg["plausibility"]
    stats = {f: Counter() for f in FIELDS}
    top = {f: Counter() for f in FIELDS}
    for p in files:
        names = pq.ParquetFile(p).schema_arrow.names
        cm = S.resolve_columns(names)
        cols = [cm[S.NUMERIC_RAW[f]] for f in FIELDS]
        t = pq.read_table(p, columns=cols).to_pandas()
        for f in FIELDS:
            s = t[cm[S.NUMERIC_RAW[f]]].astype(str).str.strip()
            ok = s.str.match(S.NUM_RE)
            v = s[ok].astype(float)
            spec = plaus[f]
            lo, hi, lo_inc = float(spec["lo"]), float(spec["hi"]), bool(spec.get("lo_inclusive", False))
            below = (v < lo) | ((v == lo) & (not lo_inc))
            above = v > hi
            stats[f]["numeric"] += int(len(v))
            stats[f]["== 0"] += int((v == 0).sum())
            stats[f]["< 0"] += int((v < 0).sum())
            stats[f]["out_below_domain"] += int(below.sum())
            stats[f]["out_above_domain"] += int(above.sum())
            top[f].update(s[ok][below | above].value_counts().head(30).to_dict())
    lines = []
    for f in FIELDS:
        lines.append(f"== {f}: domain {plaus[f]}")
        for k, v in stats[f].items():
            lines.append(f"   {k:>18}: {v:,}")
        lines.append("   most common out-of-domain values: " + ", ".join(f"{k} (x{c:,})" for k, c in top[f].most_common(8)))
    out = "\n".join(lines)
    print(out)
    (ROOT / "scripts" / "domain_values_report.txt").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
