"""Vá lỗi PySpark trên Windows + Python 3.12+ (SPARK-53759) mà KHÔNG đổi phiên bản Python hay Spark.

Triệu chứng: "Python worker exited unexpectedly (crashed)" kèm java.io.EOFException khi dùng createDataFrame, UDF, rdd.map...
Nguyên nhân: worker (đường simple-worker, luôn dùng trên Windows) không flush socket trước khi tiến trình thoát; trên Python 3.12+
socket bị đóng trước khi dữ liệu được flush.  Bản vá: bọc main() bằng try/finally và flush socket (giống cách Spark đã sửa ở bản
phát triển mới).

Script sửa 2 nơi trong môi trường Python đang chạy (KHÔNG sửa mã trong repo):
  1) pyspark/worker.py
  2) pyspark/python/lib/pyspark.zip  (Spark nạp worker từ file zip này trước, nên bắt buộc phải vá cả zip)
Có sao lưu *.bak và chạy lại nhiều lần an toàn (idempotent).

Dùng:   python scripts/patch_pyspark_windows.py            # vá
        python scripts/patch_pyspark_windows.py --restore  # hoàn tác từ bản sao lưu
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

MARK = "# patched SPARK-53759"
OLD = "    sock_file.flush()\n    main(sock_file, sock_file)\n"
NEW = (
    "    sock_file.flush()\n"
    f"    try:  {MARK}\n"
    "        main(sock_file, sock_file)\n"
    "    finally:\n"
    "        try:\n"
    "            sock_file.flush()\n"
    "        except Exception:\n"
    "            pass\n"
)


def patch_text(src: str) -> tuple[str, str]:
    if MARK in src:
        return src, "already"
    if src.count(OLD) != 1:
        return src, "pattern-not-found"
    return src.replace(OLD, NEW), "patched"


def patch_file(p: Path) -> str:
    s = p.read_text(encoding="utf-8")
    new, st = patch_text(s)
    if st == "patched":
        shutil.copy2(p, p.with_name(p.name + ".bak"))
        p.write_text(new, encoding="utf-8", newline="")
    return st


def patch_zip(z: Path) -> str:
    with zipfile.ZipFile(z) as zin:
        data = zin.read("pyspark/worker.py").decode("utf-8")
    new, st = patch_text(data)
    if st != "patched":
        return st
    bak = z.with_name(z.name + ".bak")
    shutil.copy2(z, bak)
    tmp = z.with_name(z.name + ".tmp")
    with zipfile.ZipFile(bak) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            buf = zin.read(item.filename)
            if item.filename == "pyspark/worker.py":
                buf = new.encode("utf-8")
            zout.writestr(item, buf)
    tmp.replace(z)
    return st


def restore(paths) -> None:
    for p in paths:
        bak = p.with_name(p.name + ".bak")
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"khôi phục {p}")
        else:
            print(f"không có bản sao lưu cho {p}")


def main() -> int:
    import pyspark

    root = Path(pyspark.__file__).parent
    targets = [root / "worker.py", root / "python" / "lib" / "pyspark.zip"]
    print(f"pyspark {pyspark.__version__} tại {root}")
    if "--restore" in sys.argv:
        restore(targets)
        return 0
    rc = 0
    for t in targets:
        if not t.exists():
            print(f"bỏ qua (không thấy): {t}")
            continue
        st = patch_zip(t) if t.suffix == ".zip" else patch_file(t)
        print(f"{t.name}: {st}")
        if st == "pattern-not-found":
            rc = 1
    if rc:
        print("KHÔNG vá được: mã worker của phiên bản PySpark này khác mong đợi. Báo lại cho người hỗ trợ, đừng sửa tay.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
