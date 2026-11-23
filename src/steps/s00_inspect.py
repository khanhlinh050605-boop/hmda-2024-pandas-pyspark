"""Step 00 - Inspect the data source (brief 2.3): format, size, number of rows/columns, schema,
which columns are read / not read, and the reconciliation of totals with the Transmittal Sheet (TS).
The content of demographic columns is never read: only Parquet metadata (column names, types, row counts).
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from ..core import schema as S
from ..core.runtime import log
from ..engines.pandas_engine import dup_columns, list_lar_files
from ._util import read_ts

NAME = "s00_inspect"
ENGINE_BOUND = False
SECTIONS = ["paths"]


def run(ctx):
    cfg = ctx.cfg
    files = list_lar_files(cfg)
    sizes = [p.stat().st_size for _, p in files]
    metas = [pq.ParquetFile(p).metadata for _, p in files]
    n_rows = int(sum(m.num_rows for m in metas))
    schema = pq.ParquetFile(files[0][1]).schema_arrow
    cols = schema.names
    colmap = S.resolve_columns(cols)
    dcols = set(dup_columns(cols))
    used = set(colmap.values())
    rows = []
    for f in schema:
        if f.name in used:
            role = "analysis input"
        elif f.name in dcols:
            role = "duplicate fingerprint only (never analysed)"
        else:
            role = "NEVER READ (demographic or post-decision)"
        rows.append({"column": f.name, "parquet_type": str(f.type), "role": role})
    pd.DataFrame(rows).to_csv(ctx.out / "schema_columns.csv", index=False)

    ts = read_ts(cfg)
    ts_total = int(ts["lar_count"].sum()) if ts is not None else None
    ov = {
        "source": "HMDA 2024 One-Year National Loan-Level Dataset (FFIEC/CFPB), group's Parquet conversion",
        "file_format": "Parquet (snappy), one file per part",
        "n_files": len(files),
        "total_size_gib": round(sum(sizes) / 2**30, 3),
        "n_rows": n_rows,
        "n_columns": len(cols),
        "columns_used_as_input": len(used),
        "columns_in_duplicate_fingerprint": len(dcols),
        "columns_never_read": len(cols) - len(dcols),
        "row_groups_total": int(sum(m.num_row_groups for m in metas)),
        "ts_institutions": int(len(ts)) if ts is not None else None,
        "ts_sum_lar_count": ts_total,
        "ts_matches_lar_rows": (ts_total == n_rows) if ts_total is not None else None,
        "unit_of_observation": "one application/loan record (not one funded loan)",
        "partitioning_note": ("Source and master stored as many Parquet part files (no Hive partition by state): "
                              "lender-level and product-level queries are not state-keyed, state skew (CA/TX/FL) "
                              "would give uneven partitions, and column pruning already limits I/O."),
    }
    pd.DataFrame([{"item": k, "value": v} for k, v in ov.items()]).to_csv(ctx.out / "overview.csv", index=False)
    log.info("[s00] %s rows, %s columns, %s files; TS match=%s", n_rows, len(cols), len(files), ov["ts_matches_lar_rows"])
    return ov
