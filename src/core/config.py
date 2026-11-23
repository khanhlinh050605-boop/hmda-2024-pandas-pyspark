"""Đọc config.yaml, giải quyết đường dẫn, băm cấu hình theo từng nhóm tham số."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """dict có thêm thuộc tính tiện dụng."""

    root: Path = ROOT
    source: str = ""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else (self.root / p)

    @property
    def out_dir(self) -> Path:
        return self.path("output_dir")

    @property
    def interim_dir(self) -> Path:
        return self.path("interim_dir")

    def engine_out(self, engine: str) -> Path:
        d = self.out_dir / engine
        d.mkdir(parents=True, exist_ok=True)
        return d

    def section_hash(self, sections: list[str]) -> str:
        code = hashlib.sha1()
        for p in sorted([self.root / "run.py", *self.root.glob("src/**/*.py")]):
            if p.exists():
                code.update(p.relative_to(self.root).as_posix().encode())
                code.update(p.read_bytes())
        sub = {s: self.get(s) for s in sections}
        sub["_code_fingerprint"] = code.hexdigest()
        raw = json.dumps(sub, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Cfg:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Cfg(copy.deepcopy(data))
    cfg.root = ROOT
    cfg.source = str(p.resolve())
    return cfg
