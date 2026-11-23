"""Run context (Ctx), writing of raw/result tables, manifests for re-running single steps, logging."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg
from .definitions import Defs, resolve_where

log = logging.getLogger("hmda")


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, stream=sys.stdout,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", force=True)


class Ctx:
    """Every step receives a Ctx: cfg + engine (or None) + where to write results."""

    def __init__(self, cfg: Cfg, engine=None, name: str = "shared"):
        self.cfg = cfg
        self.defs = Defs(cfg)
        self.engine = engine
        self.name = name
        self.out = cfg.engine_out(name) if name != "shared" else cfg.out_dir / "compare"
        self.raw_dir = self.out / "raw"
        self.tab_dir = self.out / "tables"
        for d in (self.out, self.raw_dir, self.tab_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.lists: dict[str, list] = {}
        self._load_lists()

    # ---- auxiliary lists created by the prepare step
    def master_dir(self) -> Path:
        return self.cfg.interim_dir / f"{self.name}_master"

    def info_dir(self) -> Path:
        d = self.cfg.interim_dir / f"{self.name}_info"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_lists(self) -> None:
        f = self.info_dir() / "lenders_with_exempt_values.csv" if self.name != "shared" else None
        if f is not None and f.exists():
            self.lists["lenders_with_exempt_values"] = pd.read_csv(f, dtype=str, keep_default_na=False)["lei"].tolist()
        else:
            self.lists.setdefault("lenders_with_exempt_values", [])

    def where(self, items) -> list[tuple]:
        return resolve_where(items, self.lists)

    # ---- write results
    def save_raw(self, name: str, df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """Raw table from an engine: compared between Pandas and Spark in the correctness step."""
        path = self.raw_dir / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.17g")
        (self.raw_dir / f"{name}.meta.json").write_text(json.dumps({"keys": keys, "rows": int(len(df))}))
        return df

    def save_table(self, name: str, df: pd.DataFrame, index: bool = False) -> pd.DataFrame:
        """Final result table (with rates, CIs...). Also compared."""
        path = self.tab_dir / f"{name}.csv"
        df.to_csv(path, index=index, float_format="%.17g")
        return df

    def read_raw(self, name: str) -> pd.DataFrame:
        meta = json.loads((self.raw_dir / f"{name}.meta.json").read_text())
        return pd.read_csv(self.raw_dir / f"{name}.csv", dtype={k: str for k in meta["keys"]},
                           keep_default_na=False, na_values=[""])


class Manifest:
    """Record which step finished with the config hash; re-run only when the config changes or with --force."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def is_done(self, step: str, cfg_hash: str) -> bool:
        s = self.data.get(step)
        return bool(s and s.get("status") == "done" and s.get("cfg_hash") == cfg_hash)

    def mark(self, step: str, cfg_hash: str, seconds: float, status: str = "done", note: str = "") -> None:
        self.data[step] = {"status": status, "cfg_hash": cfg_hash, "seconds": round(seconds, 2),
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note}
        self.path.write_text(json.dumps(self.data, indent=2))


def fmt_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Cast keys to string and sort stably so both engines produce tables in the same order."""
    df = df.copy()
    for k in keys:
        df[k] = df[k].astype(str)
    if keys:
        df = df.sort_values(keys, kind="mergesort")
    return df.reset_index(drop=True)


def clean_numeric(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    for c in df.columns:
        if c in keys:
            continue
        if pd.api.types.is_bool_dtype(df[c].dtype):
            df[c] = df[c].astype("int64")
        elif pd.api.types.is_integer_dtype(df[c].dtype):
            df[c] = df[c].astype("int64")
        elif pd.api.types.is_float_dtype(df[c].dtype):
            df[c] = df[c].astype("float64")
    return df
