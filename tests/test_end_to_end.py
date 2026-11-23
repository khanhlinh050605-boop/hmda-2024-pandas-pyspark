"""Chạy nhỏ trên dữ liệu giả lập: hai engine phải khớp. Cần scripts/make_synthetic.py đã chạy (hoặc tự tạo)."""
import json, subprocess, sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parents[1]


def test_pipeline_engines_agree():
    # Rebuild the deterministic fixture so stale synthetic files cannot make
    # the test pass against an older duplicate definition.
    subprocess.run([sys.executable, "scripts/make_synthetic.py"], cwd=ROOT, check=True)
    r = subprocess.run([sys.executable, "run.py", "--config", "config.test.yaml", "--steps", "00-09", "--engine", "both", "--force"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for engine in ("pandas", "spark"):
        meta = json.loads((ROOT / f"data/interim_test/{engine}_info/_meta.json").read_text())
        assert meta["n_dup_rows"] == 1200      # both rows in each of 600 repeated pairs
        assert meta["n_excess_dup_rows"] == 600
        master = ROOT / f"data/interim_test/{engine}_master"
        n_flagged = sum(pq.ParquetFile(p).read(columns=["is_dup"])["is_dup"].to_numpy().sum()
                        for p in master.glob("*.parquet"))
        assert int(n_flagged) == meta["n_dup_rows"]
