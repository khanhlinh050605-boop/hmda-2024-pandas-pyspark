#!/usr/bin/env python
"""Stream the official 2024 One-Year pipe file from ZIP into portable Parquet parts.

The 4.6 GB text member is never extracted.  All public columns are retained as
strings so the two engines apply the project's cleaning rules themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

LAR_URL = "https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_lar_one_year_pipe.zip"
TS_URL = "https://files.ffiec.cfpb.gov/static-data/one-year/2024/2024_public_ts_one_year_pipe.zip"
FREEZE_DATE = "2026-06-02"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data_member(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if not n.startswith("__MACOSX/") and not n.endswith("/")]
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one data member, found {names}")
    return names[0]


def convert(lar_zip: Path, out_dir: Path, block_mib: int) -> tuple[str, int, int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    rows = parts = 0
    with zipfile.ZipFile(lar_zip) as zf:
        member = data_member(zf)
        with zf.open(member) as fh:
            header = fh.readline().decode("utf-8-sig").rstrip("\r\n").split("|")
        types = {c: pa.string() for c in header}
        with zf.open(member) as fh:
            reader = csv.open_csv(
                fh,
                read_options=csv.ReadOptions(block_size=block_mib * 1024 * 1024, use_threads=True),
                parse_options=csv.ParseOptions(delimiter="|"),
                convert_options=csv.ConvertOptions(column_types=types, strings_can_be_null=False),
            )
            for batch in reader:
                pq.write_table(pa.Table.from_batches([batch]), out_dir / f"part-{parts:05d}.parquet",
                               compression="snappy", use_dictionary=True)
                rows += batch.num_rows
                parts += 1
                if parts % 20 == 0:
                    print(f"wrote {parts} parts, {rows:,} rows", flush=True)
    return member, rows, parts, header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lar-zip", type=Path, required=True)
    ap.add_argument("--ts-zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/raw/2024_one_year_parquet"))
    ap.add_argument("--block-mib", type=int, default=16)
    args = ap.parse_args()
    lar_zip, ts_zip, out = args.lar_zip.resolve(), args.ts_zip.resolve(), args.out.resolve()
    member, rows, parts, columns = convert(lar_zip, out, args.block_mib)
    ts_target = out.parent / "2024_public_ts_one_year_pipe.zip"
    shutil.copy2(ts_zip, ts_target)
    try:
        directory = out.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        directory = str(out)
    manifest = {
        "release": "HMDA 2024 One-Year National Loan-Level Dataset",
        "freeze_date": FREEZE_DATE,
        "lar_url": LAR_URL,
        "ts_url": TS_URL,
        "lar_zip": {"name": lar_zip.name, "bytes": lar_zip.stat().st_size, "sha256": sha256(lar_zip), "member": member},
        "ts_zip": {"name": ts_target.name, "bytes": ts_target.stat().st_size, "sha256": sha256(ts_target)},
        "parquet": {"directory": directory, "parts": parts, "rows": rows, "columns": len(columns),
                    "column_names": columns},
    }
    (out.parent / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "parts": parts, "columns": len(columns)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
