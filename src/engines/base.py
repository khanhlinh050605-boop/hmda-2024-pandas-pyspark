"""Giao diện chung của hai engine. Mọi bước phân tích chỉ gọi 4 hàm này:

  prepare()           đọc LAR thô -> master Parquet (độc lập trong từng engine)
  group_sum()         nhóm theo khoá, đếm dòng + cộng các cột cờ/số nguyên
  group_moments()     nhóm theo khoá, n + tổng + tổng tích chéo (cho mean/std/Pearson/hồi quy)
  group_percentiles() nhóm theo khoá, percentile CHÍNH XÁC (nội suy tuyến tính)

Engine trả DataFrame nhỏ (đã gộp). Toàn bộ thống kê tiếp theo chạy trên bảng nhỏ bằng code dùng chung.
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

    # ---- cần cài đặt trong lớp con
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

    # ---- API công khai (chuẩn hoá đầu ra)
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
        df = df[keys + ["n_valid", "min", "max"] + [pname(p) for p in probs]]   # cùng thứ tự cột ở mọi engine
        return clean_numeric(fmt_keys(df, keys), keys)
