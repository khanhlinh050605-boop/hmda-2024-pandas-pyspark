"""Engine Pandas: prepare (đọc từng file, làm sạch, tạo biến) + các truy vấn gộp trên master Parquet.

Cài đặt độc lập với Spark: chỉ dùng pandas/numpy/pyarrow. Master được đọc theo cột, cache từng cột.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..core import schema as S
from ..core.definitions import DTI_TEXT_MAP, Defs, dti_band_numpy
from ..core.runtime import log
from .base import Engine, pairs_of, pname

MOD = 2147483647  # 2^31 - 1: số nguyên tố của bộ băm khoá lấy mẫu (xem sample_key)


# =============================================================== khoá lấy mẫu (dùng chung ý tưởng với Spark)
def sample_key(part_no: np.ndarray, row_idx: np.ndarray, seed: int) -> np.ndarray:
    """u in [0,1) xác định theo (part_no,row_idx): cùng công thức số nguyên với Spark => mẫu lồng nhau giống hệt."""
    i = part_no.astype(np.int64) * 10_000_000 + row_idx.astype(np.int64)
    h = (i * 48271 + (seed % MOD)) % MOD
    h2 = (h * h + 1013904223) % MOD
    h3 = (h2 * h2 + 12345) % MOD
    return h3.astype(np.float64) / MOD


# =============================================================== phân tích chuỗi -> số + trạng thái
def parse_numeric(ser: pd.Series, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one quantitative HMDA field using only that field's special tokens."""
    s = ser.str.strip()
    up = s.str.upper()
    is_val = s.str.match(S.NUM_RE).to_numpy(bool)
    specials = S.FIELD_SPECIAL_VALUES[field]
    conds = [(s == "").to_numpy(bool)]
    labels = ["EMPTY"]
    for token, label in specials.items():
        conds.append((up == token).to_numpy(bool))
        labels.append(label)
    conds.append(is_val)
    labels.append("VALUE")
    status = np.select(conds, labels, default="UNPARSABLE").astype(object)
    vals = pd.to_numeric(s.where(pd.Series(status == "VALUE", index=s.index)), errors="coerce").to_numpy(np.float64)
    return vals, status


def in_range(vals: np.ndarray, spec: dict) -> np.ndarray:
    lo, hi = float(spec["lo"]), float(spec["hi"])
    lo_ok = vals >= lo if spec.get("lo_inclusive", False) else vals > lo
    with np.errstate(invalid="ignore"):
        return lo_ok & (vals <= hi) & np.isfinite(vals)


def band_labels_for(vals: np.ndarray, valid: np.ndarray, status: np.ndarray, edges, labels) -> np.ndarray:
    lab = np.asarray(labels, dtype=object)
    idx = np.digitize(np.nan_to_num(vals, nan=0.0), np.asarray(edges, float))
    out = np.where(valid, lab[idx], "")
    out = np.where(~valid & (status != "VALUE"), np.char.add("~", status.astype(str)).astype(object), out)
    out = np.where(~valid & (status == "VALUE"), "~OUTSIDE_DOMAIN", out)
    return out.astype(object)


# =============================================================== xây khung master cho một file
def build_master_frame(df: pd.DataFrame, cfg, defs: Defs, part_no: int) -> pd.DataFrame:
    """df: cột chuẩn (chuỗi, không null). Trả về khung master đã có biến phái sinh và cờ."""
    n = len(df)
    plaus = cfg["plausibility"]
    out: dict[str, np.ndarray | pd.Series] = {}

    for c in S.CATEGORICAL_RAW:
        s = df[c].str.strip()
        out[c] = s.where(s != "", S.MISSING).to_numpy(object)

    # --- số + trạng thái + hợp lệ
    num: dict[str, tuple] = {}
    for c in S.NUMERIC_RAW:
        vals, status = parse_numeric(df[c], c)
        spec_key = c
        valid = (status == "VALUE") & in_range(vals, plaus[spec_key])
        num[c] = (vals, status, valid)
        out[c] = vals
        out[f"{c}_status"] = status
        out[f"v_{c}"] = valid.astype(np.int8)

    # --- DTI: band chữ hoặc số
    raw = df[S.DTI_RAW].str.strip()
    txt = raw.str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
    dti_vals, dti_status = parse_numeric(raw, "dti")
    is_text = txt.isin(list(DTI_TEXT_MAP)).to_numpy(bool)
    dti_status = np.where(is_text, "VALUE", dti_status).astype(object)
    dti_valid_numeric = (dti_status == "VALUE") & in_range(dti_vals, plaus["dti"])
    dti_valid = is_text | dti_valid_numeric
    dti_band = np.where(is_text, txt.map(DTI_TEXT_MAP).fillna("").to_numpy(object), dti_band_numpy(dti_vals))
    dti_band = np.where(~dti_valid & (dti_status != "VALUE"),
                        np.char.add("~", dti_status.astype(str)).astype(object), dti_band)
    dti_band = np.where(~dti_valid & (dti_status == "VALUE"), "~OUTSIDE_DOMAIN", dti_band)
    out["dti_status"] = dti_status
    out["dti_band"] = dti_band.astype(object)
    out["v_dti"] = dti_valid.astype(np.int8)

    # --- LTI
    v_inc, v_amt = num["income"][2], num["loan_amount"][2]
    ok = v_inc & v_amt
    with np.errstate(divide="ignore", invalid="ignore"):
        lti_raw = np.where(ok, num["loan_amount"][0] / (num["income"][0] * 1000.0), np.nan)
    v_lti = ok & in_range(lti_raw, plaus["lti"])
    lti = np.where(v_lti, lti_raw, np.nan)
    lti_status = np.where(num["income"][1] != "VALUE", num["income"][1],
                          np.where(num["loan_amount"][1] != "VALUE", num["loan_amount"][1], "VALUE")).astype(object)
    out["lti"] = lti
    out["v_lti"] = v_lti.astype(np.int8)

    # --- band
    out["income_band"] = band_labels_for(num["income"][0], v_inc, num["income"][1],
                                         defs.edges["income_band"], defs.labels["income_band"])
    out["loan_amount_band"] = band_labels_for(num["loan_amount"][0], v_amt, num["loan_amount"][1],
                                              defs.edges["loan_amount_band"], defs.labels["loan_amount_band"])
    out["lti_band"] = band_labels_for(lti, v_lti, lti_status, defs.edges["lti_band"], defs.labels["lti_band"])
    out["cltv_band"] = band_labels_for(num["cltv"][0], num["cltv"][2], num["cltv"][1],
                                       defs.edges["cltv_band"], defs.labels["cltv_band"])
    out["property_value_band"] = band_labels_for(num["property_value"][0], num["property_value"][2],
                                                 num["property_value"][1], defs.edges["property_value_band"],
                                                 defs.labels["property_value_band"])

    # --- log, năm, bin histogram
    def safe_log(v, valid):
        return np.where(valid, np.log(np.where(valid, v, 1.0)), np.nan)

    out["log_loan_amount"] = safe_log(num["loan_amount"][0], v_amt)
    out["log_income"] = safe_log(num["income"][0], v_inc)
    out["log_property_value"] = safe_log(num["property_value"][0], num["property_value"][2])
    out["term_years"] = np.where(num["loan_term"][2], num["loan_term"][0] / 12.0, np.nan)

    def lbin(v, valid):
        x = np.where(valid, v, 1.0)
        return np.where(valid, np.floor(np.log10(x) * 10.0), -9999).astype(np.int16)

    out["loan_amount_lbin"] = lbin(num["loan_amount"][0], v_amt)
    out["income_lbin"] = lbin(num["income"][0], v_inc)
    v_rate = num["interest_rate"][2]
    out["rate_bin"] = np.where(v_rate, np.floor(np.where(v_rate, num["interest_rate"][0], 0.0) * 4.0), -9999).astype(np.int16)

    # --- sản phẩm / phân khúc
    sep = S.SEP
    out["product"] = (out["loan_type"].astype(str) + sep + out["loan_purpose"].astype(str)).astype(object)
    out["segment"] = (out["product"].astype(str) + sep + out["lien_status"].astype(str)).astype(object)

    # --- cờ kết quả
    act = out["action_taken"].astype(str)
    out["is_app"] = np.isin(act, ["1", "2", "3", "4", "5"]).astype(np.int8)
    out["is_decided"] = np.isin(act, ["1", "2", "3"]).astype(np.int8)
    out["is_denied"] = (act == "3").astype(np.int8)
    out["is_orig"] = (act == "1").astype(np.int8)
    bcp, oel, rev = (out[c].astype(str) for c in ("business_or_commercial_purpose", "open_end_line_of_credit", "reverse_mortgage"))
    out["is_business"] = (bcp == "1").astype(np.int8)
    out["is_open_end"] = (oel == "1").astype(np.int8)
    out["is_reverse"] = (rev == "1").astype(np.int8)
    consumer_cf = (bcp == "2") & (oel == "2") & (rev == "2")
    out["is_consumer_cf"] = consumer_cf.astype(np.int8)

    exempt = np.isin(bcp, ["1111"]) | np.isin(oel, ["1111"]) | np.isin(rev, ["1111"]) | (dti_status == "EXEMPT")
    for c in ("income", "property_value", "loan_term", "interest_rate", "rate_spread", "total_loan_costs", "cltv"):
        exempt |= num[c][1] == "EXEMPT"
    out["has_reported_exempt_value"] = exempt.astype(np.int8)

    # --- mẫu hồi quy Q4
    base = (act == "1") & consumer_cf & v_amt & v_inc & num["cltv"][2] & num["loan_term"][2] & num["property_value"][2]
    out["q4_reg_rate"] = (base & num["interest_rate"][2]).astype(np.int8)
    out["q4_reg_spread"] = (base & num["rate_spread"][2]).astype(np.int8)

    # --- khoá lấy mẫu
    row_idx = np.arange(n, dtype=np.int64)
    out["part_no"] = np.full(n, part_no, dtype=np.int32)
    out["row_idx"] = row_idx.astype(np.int32)
    out["sample_u"] = sample_key(np.full(n, part_no), row_idx, int(cfg["project"]["seed"]))

    return pd.DataFrame(out)


MASTER_SCHEMA_OVERRIDES = {"part_no": pa.int32(), "row_idx": pa.int32()}


def to_arrow(df: pd.DataFrame) -> pa.Table:
    fields = []
    for c in df.columns:
        dt = df[c].dtype
        if c in MASTER_SCHEMA_OVERRIDES:
            fields.append(pa.field(c, MASTER_SCHEMA_OVERRIDES[c]))
        elif dt == object:
            fields.append(pa.field(c, pa.string()))
        elif dt == np.int8:
            fields.append(pa.field(c, pa.int8()))
        elif dt == np.int16:
            fields.append(pa.field(c, pa.int16()))
        elif np.issubdtype(dt, np.floating):
            fields.append(pa.field(c, pa.float64()))
        else:
            fields.append(pa.field(c, pa.from_numpy_dtype(dt)))
    return pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)


# =============================================================== liệt kê file nguồn
def list_lar_files(cfg) -> list[tuple[int, Path]]:
    d = cfg.path("lar_dir")
    rx = re.compile(cfg["paths"]["part_regex"])
    files = []
    for p in sorted(d.glob("*.parquet")):
        m = rx.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    if not files:
        raise FileNotFoundError(f"Không thấy file *.parquet khớp part_regex trong {d}")
    return sorted(files)


def dup_columns(all_cols) -> list[str]:
    """Cột dùng để băm trùng lặp: mọi cột TRỪ nhóm bị cấm (nhân khẩu học / sau quyết định)."""
    return [c for c in all_cols if not S.FORBIDDEN_RE.search(S.norm_name(c))]


# =============================================================== lớp engine
class PandasEngine(Engine):
    name = "pandas"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._files: list[Path] = []
        self._cache: dict[str, pd.Series] = {}
        self._cols: list[str] = []

    # ---------------------------------------------------------------- prepare
    def prepare(self, ctx) -> dict:
        cfg, defs = self.cfg, ctx.defs
        t0 = time.time()
        src = list_lar_files(cfg)
        first_cols = pq.ParquetFile(src[0][1]).schema_arrow.names
        colmap = S.resolve_columns(first_cols)
        dcols = dup_columns(first_cols)
        S.assert_no_forbidden(dcols)
        master = ctx.master_dir()
        stage = master.parent / f"{self.name}_stage"
        for d in (master, stage):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        inv = {v: k for k, v in colmap.items()}
        hashes, exempt_value_lei, rows = [], set(), {}
        log.info("[pandas] pass A: %d file", len(src))
        for k, (part_no, path) in enumerate(src):
            want = list(dict.fromkeys(dcols))
            tbl = pq.read_table(path, columns=want)
            tbl = tbl.cast(pa.schema([pa.field(f.name, pa.string()) for f in tbl.schema]))
            raw = tbl.to_pandas()
            raw = raw.astype(object).where(raw.notna(), "\x01NULL")
            h = pd.util.hash_pandas_object(raw[dcols], index=False).to_numpy(np.uint64)
            hashes.append(h)
            canon = raw[list(colmap.values())].rename(columns=inv)
            canon = canon.replace("\x01NULL", "").rename(columns={S.NUMERIC_RAW["cltv"]: "cltv"})
            m = build_master_frame(canon, cfg, defs, part_no)
            exempt_value_lei.update(m.loc[m["has_reported_exempt_value"] == 1, "lei"].unique().tolist())
            pq.write_table(to_arrow(m), stage / f"part-{part_no:05d}.parquet", compression="snappy")
            rows[part_no] = len(m)
            if (k + 1) % 25 == 0:
                log.info("[pandas]   %d/%d file", k + 1, len(src))
        allh = np.concatenate(hashes)
        uniq, cnt = np.unique(allh, return_counts=True)
        dup_vals = uniq[cnt > 1]
        n_dup_rows = int(cnt[cnt > 1].sum())
        n_excess_dup_rows = int((cnt[cnt > 1] - 1).sum())
        log.info("[pandas] pass B (gắn cờ trùng lặp): %d giá trị băm lặp, %d dòng", len(dup_vals), n_dup_rows)
        for (part_no, _), h in zip(src, hashes):
            t = pq.read_table(stage / f"part-{part_no:05d}.parquet")
            flag = np.isin(h, dup_vals).astype(np.int8)
            t = t.append_column("is_dup", pa.array(flag, type=pa.int8()))
            pq.write_table(t, master / f"part-{part_no:05d}.parquet", compression="snappy")
        shutil.rmtree(stage)
        pd.DataFrame({"lei": sorted(exempt_value_lei)}).to_csv(ctx.info_dir() / "lenders_with_exempt_values.csv", index=False)
        meta = {"engine": "pandas", "n_files": len(src), "n_rows": int(sum(rows.values())),
                "rows_per_part": {str(k): v for k, v in rows.items()}, "n_dup_rows": n_dup_rows,
                "n_excess_dup_rows": n_excess_dup_rows, "dup_columns_n": len(dcols),
                "lenders_with_exempt_values": len(exempt_value_lei),
                "seconds": round(time.time() - t0, 1), "pandas": pd.__version__, "pyarrow": pa.__version__}
        (ctx.info_dir() / "_meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    # ---------------------------------------------------------------- đọc master theo cột
    def open(self, ctx) -> None:
        d = ctx.master_dir()
        self._files = sorted(d.glob("part-*.parquet"))
        if not self._files:
            raise FileNotFoundError(f"Chưa có master Pandas ở {d}. Chạy bước 01 trước.")
        self._cols = pq.ParquetFile(self._files[0]).schema_arrow.names
        self._cache = {}
        self._ctx = ctx

    def master_columns(self) -> list[str]:
        return list(self._cols)

    def col(self, name: str) -> pd.Series:
        if name in self._cache:
            return self._cache[name]
        if name not in self._cols:
            raise KeyError(f"Master không có cột {name}")
        is_str = pa.types.is_string(pq.ParquetFile(self._files[0]).schema_arrow.field(name).type)
        chunks = []
        for f in self._files:
            t = pq.read_table(f, columns=[name], read_dictionary=[name] if is_str else None)
            chunks.extend(t.column(0).chunks)
        ca = pa.chunked_array(chunks)
        if is_str:
            ca = ca.unify_dictionaries()
        ser = ca.to_pandas()
        if not isinstance(ser, pd.Series):
            ser = pd.Series(ser)
        ser = ser.reset_index(drop=True)
        if pd.api.types.is_float_dtype(ser.dtype):
            ser = ser.astype(np.float64)
        self._cache[name] = ser
        return ser

    def read_columns(self, names: list[str]) -> pd.DataFrame:
        """Read one projected column set with a single Parquet open per file."""
        missing = [c for c in names if c not in self._cols]
        if missing:
            raise KeyError(f"Master không có cột {missing}")
        frames = [pq.read_table(f, columns=names).to_pandas() for f in self._files]
        out = pd.concat(frames, ignore_index=True)
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c].dtype):
                out[c] = out[c].astype(np.float64)
        return out

    def drop_cache(self) -> None:
        self._cache = {}

    def n_rows(self) -> int:
        return int(sum(pq.ParquetFile(f).metadata.num_rows for f in self._files))

    # ---------------------------------------------------------------- where
    def _mask(self, where) -> np.ndarray | None:
        mask = None
        for c, op, v in where:
            s = self.col(c)
            if op == "==":
                m = (s == v)
            elif op == "!=":
                m = (s != v)
            elif op == "in":
                m = s.isin(list(v))
            elif op == "not_in":
                m = ~s.isin(list(v))
            elif op == ">=":
                m = s >= v
            elif op == "<=":
                m = s <= v
            elif op == ">":
                m = s > v
            elif op == "<":
                m = s < v
            else:
                raise ValueError(f"toán tử where lạ: {op}")
            m = np.asarray(m, dtype=bool)
            mask = m if mask is None else (mask & m)
        return mask

    def _keyframe(self, keys, mask) -> pd.DataFrame:
        kd = pd.DataFrame({k: self.col(k) for k in keys})
        return kd if mask is None else kd[mask].reset_index(drop=True)

    # ---------------------------------------------------------------- group_sum
    def _group_sum(self, keys, sums, where):
        mask = self._mask(where)
        if not keys:
            n = int(mask.sum()) if mask is not None else self.n_rows()
            row = {"n": n}
            for s in sums:
                v = self.col(s).to_numpy()
                row[s] = int((v[mask] if mask is not None else v).astype(np.int64).sum())
            return pd.DataFrame([row])
        d = self._keyframe(keys, mask)
        d["__n"] = 1
        for s in sums:
            v = self.col(s).to_numpy()
            d[s] = (v[mask] if mask is not None else v).astype(np.int64)
        agg = {"__n": "sum", **{s: "sum" for s in sums}}
        g = d.groupby(keys, observed=True, sort=True).agg(agg).reset_index()
        return g.rename(columns={"__n": "n"})

    # ---------------------------------------------------------------- group_moments
    def _group_moments(self, keys, cols, where, cross, extremes):
        mask = self._mask(where)
        vals = {c: self.col(c).to_numpy(np.float64) for c in cols}
        ok = np.ones(len(next(iter(vals.values()))), dtype=bool)
        for c in cols:
            ok &= np.isfinite(vals[c])
        if mask is not None:
            ok &= mask
        if keys:
            d = self._keyframe(keys, ok)
            g = d.groupby(keys, observed=True, sort=True)
            codes = g.ngroup().to_numpy()
            kdf = g.size().reset_index()[keys]
            ng = len(kdf)
        else:
            codes = np.zeros(int(ok.sum()), dtype=np.int64)
            kdf = pd.DataFrame(index=[0])
            ng = 1
        res = kdf.copy()
        res["n"] = np.bincount(codes, minlength=ng).astype(np.int64)
        x = {c: vals[c][ok] for c in cols}
        for c in cols:
            res[f"sum__{c}"] = np.bincount(codes, weights=x[c], minlength=ng)
            if extremes:
                s = pd.Series(x[c]).groupby(codes)
                res[f"min__{c}"] = s.min().reindex(range(ng)).to_numpy()
                res[f"max__{c}"] = s.max().reindex(range(ng)).to_numpy()
        for a, b in pairs_of(cols, cross):
            res[f"prod__{a}__{b}"] = np.bincount(codes, weights=x[a] * x[b], minlength=ng)
        return res

    # ---------------------------------------------------------------- group_percentiles
    def _group_percentiles(self, keys, col, probs, where):
        mask = self._mask(where)
        v = self.col(col).to_numpy(np.float64)
        ok = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
        names = [pname(p) for p in probs]
        if not keys:
            x = v[ok]
            row = {"n_valid": int(x.size), "min": float(x.min()) if x.size else np.nan,
                   "max": float(x.max()) if x.size else np.nan}
            q = np.quantile(x, probs) if x.size else [np.nan] * len(probs)
            row.update({nm: float(val) for nm, val in zip(names, q)})
            return pd.DataFrame([row])
        d = self._keyframe(keys, ok)
        d["__v"] = v[ok]
        g = d.groupby(keys, observed=True, sort=True)["__v"]
        qd = g.quantile(probs).unstack(-1)
        qd.columns = names
        out = qd.copy()
        out["n_valid"] = g.count()
        out["min"] = g.min()
        out["max"] = g.max()
        return out.reset_index()
