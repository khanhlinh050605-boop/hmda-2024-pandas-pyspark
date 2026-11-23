"""Patch the PySpark bug on Windows + Python 3.12+ (SPARK-53759) WITHOUT changing the Python or Spark version.

Symptom: "Python worker exited unexpectedly (crashed)" with java.io.EOFException when using createDataFrame, UDFs, rdd.map...
Cause: the worker (simple-worker path, always used on Windows) does not flush the socket before the process exits; on Python 3.12+
the socket is closed before the data is flushed.  The patch wraps main() in try/finally and flushes the socket (the same way Spark fixed it in the
newer development version).

The script patches 2 places in the running Python environment (it does NOT modify code in the repo):
  1) pyspark/worker.py
  2) pyspark/python/lib/pyspark.zip  (Spark loads the worker from this zip first, so the zip must be patched too)
A *.bak backup is made and re-running is safe (idempotent).

Usage:   python scripts/patch_pyspark_windows.py            # patch
        python scripts/patch_pyspark_windows.py --restore  # restore from the backup
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
            print(f"restored {p}")
        else:
            print(f"no backup for {p}")


def main() -> int:
    import pyspark

    root = Path(pyspark.__file__).parent
    targets = [root / "worker.py", root / "python" / "lib" / "pyspark.zip"]
    print(f"pyspark {pyspark.__version__} at {root}")
    if "--restore" in sys.argv:
        restore(targets)
        return 0
    rc = 0
    for t in targets:
        if not t.exists():
            print(f"skipping (not found): {t}")
            continue
        st = patch_zip(t) if t.suffix == ".zip" else patch_file(t)
        print(f"{t.name}: {st}")
        if st == "pattern-not-found":
            rc = 1
    if rc:
        print("COULD NOT patch: the worker code of this PySpark version differs from what was expected. Report it to whoever supports you; do not edit by hand.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
