#!/usr/bin/env python
"""Điều phối pipeline. Mỗi bước là một file riêng trong src/steps/, chạy lại độc lập.

Ví dụ:
  python run.py --steps all --engine both            # 00..12 (11 = predictive, 12 = hình/bảng)
  python run.py --steps 06,07 --engine pandas --force  # sửa Q3/Q4 xong chỉ chạy lại 2 bước đó
  python run.py --steps 11                             # chỉ chạy predictive (proposal mục 8)
  python run.py --list
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback

from src.core.config import load_config
from src.core.runtime import Ctx, Manifest, log, setup_logging

STEPS = {
    "00": "s00_inspect", "01": "s01_prepare", "02": "s02_quality", "03": "s03_descriptive",
    "04": "s04_q1", "05": "s05_q2", "06": "s06_q3", "07": "s07_q4", "08": "s08_robustness",
    "09": "s09_correctness", "10": "s10_benchmark", "11": "s11_predictive", "12": "s12_report_assets",
}
DEFAULT = list(STEPS)


def parse_steps(arg: str) -> list[str]:
    if arg == "all":
        return DEFAULT
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out += [k for k in STEPS if int(a) <= int(k) <= int(b)]
        elif tok:
            out.append(tok.zfill(2))
    bad = [k for k in out if k not in STEPS]
    if bad:
        raise SystemExit(f"Bước không tồn tại: {bad}. Dùng --list.")
    return out


def make_engine(name, cfg):
    if name == "pandas":
        from src.engines.pandas_engine import PandasEngine
        return PandasEngine(cfg)
    from src.engines.spark_engine import SparkEngine
    return SparkEngine(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="all")
    ap.add_argument("--engine", choices=["pandas", "spark", "both"], default="both")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="chạy lại dù manifest báo đã xong")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    if a.list:
        for k, v in STEPS.items():
            m = importlib.import_module(f"src.steps.{v}")
            print(f"{k}  {v:20s} {'[theo engine]' if m.ENGINE_BOUND else '[dùng chung]'}  {(m.__doc__ or '').strip().splitlines()[0]}")
        return 0
    cfg = load_config(a.config)
    steps = parse_steps(a.steps)
    engines = ["pandas", "spark"] if a.engine == "both" else [a.engine]
    failures = []
    cache: dict = {}
    for sid in steps:
        mod = importlib.import_module(f"src.steps.{STEPS[sid]}")
        h = cfg.section_hash(mod.SECTIONS)
        targets = engines if mod.ENGINE_BOUND else ["shared"]
        for t in targets:
            cfg_name = t
            if t == "shared":
                (cfg.out_dir / "compare").mkdir(parents=True, exist_ok=True)
            man = Manifest((cfg.out_dir / "compare" if t == "shared" else cfg.engine_out(t)) / "manifest.json")
            if not a.force and man.is_done(mod.NAME, h):
                log.info("[%s/%s] đã xong (config không đổi) - bỏ qua. Dùng --force để chạy lại.", sid, t)
                continue
            log.info("=== bước %s %s [%s] ===", sid, mod.NAME, t)
            t0 = time.time()
            try:
                if mod.ENGINE_BOUND:
                    if t not in cache:
                        cache[t] = make_engine(t, cfg)
                    eng = cache[t]
                    ctx = Ctx(cfg, eng, t)
                    if sid != "01":
                        eng.open(ctx)
                else:
                    ctx = Ctx(cfg, None, "shared")
                mod.run(ctx)
                man.mark(mod.NAME, h, time.time() - t0)
                log.info("[%s/%s] xong trong %.1fs", sid, t, time.time() - t0)
            except Exception as e:   # noqa: BLE001
                traceback.print_exc()
                man.mark(mod.NAME, h, time.time() - t0, status="failed", note=str(e)[:200])
                failures.append((sid, t, str(e)))
                log.info("[%s/%s] LỖI: %s", sid, t, e)
                if sid in ("01",):
                    break
            finally:
                if mod.ENGINE_BOUND and t == "pandas" and t in cache:
                    cache[t].drop_cache()
    for e in cache.values():
        e.close()
    if failures:
        log.info("CÁC BƯỚC LỖI: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
