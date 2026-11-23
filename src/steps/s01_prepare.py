"""Step 01 - Data preparation (data engineering): each engine reads the raw LAR and builds its OWN Parquet master.

The cleaning rules (see cleaning_rules.md) are identical in both engines:
  - read every column as a string; cast to numbers explicitly;
  - recognise NA / Exempt field by field; 1111 / 8888 / 9999 are special
    codes only on fields where the 2024 documentation defines them as such (for example 1111
    on the three categorical flags), not as a global rule;
  - values outside the plausible domain are KEPT and flagged v_<field>=0 (not deleted);
  - missing values are not zero-filled; derived ratios are computed only when eligible and are null otherwise.
"""
from __future__ import annotations

import json

NAME = "s01_prepare"
ENGINE_BOUND = True
SECTIONS = ["paths", "plausibility", "bands", "project", "spark"]


def run(ctx):
    meta = ctx.engine.prepare(ctx)
    (ctx.out / "prepare_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
