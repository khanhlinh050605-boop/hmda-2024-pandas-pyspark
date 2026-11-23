"""Check the environment before running: Python 3.12, Java, HADOOP_HOME (Windows), cnsplots, data path."""
import importlib, os, platform, shutil, subprocess, sys
from pathlib import Path

ok = True
def say(flag, msg):
    global ok
    print(("[OK]   " if flag else "[ERROR] ") + msg)
    ok = ok and flag

say(sys.version_info[:2] == (3, 12), f"Python {platform.python_version()} (exactly 3.12 required)")
for m in ["pandas", "numpy", "pyarrow", "pyspark", "scipy", "sklearn", "yaml", "psutil", "cnsplots", "distutils"]:
    try:
        importlib.import_module(m); say(True, f"import {m}")
    except Exception as e:
        say(False, f"import {m}: {e}")
java = shutil.which("java")
if java:
    v = subprocess.run([java, "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    say("17" in v, f"Java: {v} (17 required)")
else:
    say(False, "java not found on PATH")
if platform.system() == "Windows":
    h = os.environ.get("HADOOP_HOME")
    say(bool(h) and (Path(h) / "bin" / "winutils.exe").exists(), f"HADOOP_HOME={h} has bin/winutils.exe")
try:
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).parents[1] / "config.yaml"))
    p = Path(cfg["paths"]["lar_dir"])
    n = len(list(p.glob("*.parquet"))) if p.exists() else 0
    say(n > 0, f"Parquet folder {p}: {n} files")
except Exception as e:
    say(False, f"reading config.yaml: {e}")
sys.exit(0 if ok else 1)
