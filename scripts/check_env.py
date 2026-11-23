"""Kiểm tra môi trường trước khi chạy: Python 3.12, Java, HADOOP_HOME (Windows), cnsplots, đường dẫn dữ liệu."""
import importlib, os, platform, shutil, subprocess, sys
from pathlib import Path

ok = True
def say(flag, msg):
    global ok
    print(("[OK]   " if flag else "[LỖI] ") + msg)
    ok = ok and flag

say(sys.version_info[:2] == (3, 12), f"Python {platform.python_version()} (cần đúng 3.12)")
for m in ["pandas", "numpy", "pyarrow", "pyspark", "scipy", "sklearn", "yaml", "psutil", "cnsplots", "distutils"]:
    try:
        importlib.import_module(m); say(True, f"import {m}")
    except Exception as e:
        say(False, f"import {m}: {e}")
java = shutil.which("java")
if java:
    v = subprocess.run([java, "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    say("17" in v, f"Java: {v} (cần 17)")
else:
    say(False, "Không thấy java trong PATH")
if platform.system() == "Windows":
    h = os.environ.get("HADOOP_HOME")
    say(bool(h) and (Path(h) / "bin" / "winutils.exe").exists(), f"HADOOP_HOME={h} có bin/winutils.exe")
try:
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).parents[1] / "config.yaml"))
    p = Path(cfg["paths"]["lar_dir"])
    n = len(list(p.glob("*.parquet"))) if p.exists() else 0
    say(n > 0, f"Thư mục parquet {p}: {n} file")
except Exception as e:
    say(False, f"đọc config.yaml: {e}")
sys.exit(0 if ok else 1)
