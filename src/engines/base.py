"""Common interface of the two engines. Every analysis step calls only these 4 functions:

  prepare()           read the raw LAR -> Parquet master (independently in each engine)
  group_sum()         group by key, count rows + sum the flag/integer columns
  group_moments()     group by key, n + sums + cross-product sums (for mean/std/Pearson/regression)
  group_percentiles() group by key, EXACT percentiles (linear interpolation)

Engines return small (aggregated) DataFrames. All later statistics run on the small tables with shared code.
"""
from __future__ import annotations

import itertools

import pandas as pd

from ..core.runtime import clean_numeric, fmt_keys


def pname(p: float) -> str:
    return "p" + format(p * 100, "g").replace(".", "_")


def pairs_of(cols: list[str], cross: bool) -> list[tuple[str, str]]:
    if not cross:
        return [(c, c) for c in cols]
    return [(a, b) for a, b in itertools.combinations_with_replacement(cols, 2)]


class Engine:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    # ---- to be implemented in the subclass
    def prepare(self, ctx) -> dict:
        raise NotImplementedError

    def open(self, ctx) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def _group_sum(self, keys, sums, where):
        raise NotImplementedError

    def _group_moments(self, keys, cols, where, cross, extremes):
        raise NotImplementedError

    def _group_percentiles(self, keys, col, probs, where):
        raise NotImplementedError

    def master_columns(self) -> list[str]:
        raise NotImplementedError

    # ---- public API (normalised output)
    def group_sum(self, keys, sums=(), where=()) -> pd.DataFrame:
        keys, sums = list(keys), list(sums)
        df = self._group_sum(keys, sums, list(where))
        return clean_numeric(fmt_keys(df, keys), keys)

    def group_moments(self, keys, cols, where=(), cross=False, extremes=False) -> pd.DataFrame:
        keys, cols = list(keys), list(cols)
        df = self._group_moments(keys, cols, list(where), cross, extremes)
        return clean_numeric(fmt_keys(df, keys), keys)

    def group_percentiles(self, keys, col, probs, where=()) -> pd.DataFrame:
        keys = list(keys)
        df = self._group_percentiles(keys, col, list(probs), list(where))
        df = df[keys + ["n_valid", "min", "max"] + [pname(p) for p in probs]]   # same column order in every engine
        return clean_numeric(fmt_keys(df, keys), keys)
