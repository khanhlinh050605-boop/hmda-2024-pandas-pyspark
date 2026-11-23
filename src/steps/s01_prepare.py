"""Bước 01 - Chuẩn bị dữ liệu (data engineering): mỗi engine đọc LAR thô và tạo master Parquet RIÊNG.

Quy tắc làm sạch (xem cleaning_rules.md) giống nhau ở hai engine:
  - đọc mọi cột dưới dạng chuỗi; ép kiểu số tường minh;
  - nhận diện NA / Exempt theo đúng từng trường; 1111 / 8888 / 9999 chỉ là
    mã đặc biệt ở các trường mà tài liệu 2024 quy định như vậy (ví dụ 1111
    ở ba cờ categorical), không áp dụng một quy tắc toàn cục;
  - giá trị ngoài miền hợp lý được GIỮ NGUYÊN và gắn cờ v_<field>=0 (không xoá);
  - không điền 0 thay thế giá trị thiếu; tỷ số phái sinh chỉ tính khi đủ điều kiện, còn lại là null.
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
