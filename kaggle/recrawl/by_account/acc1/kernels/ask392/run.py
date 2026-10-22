"""Kaggle shard runner for the ViBioMIR re-crawl: ask392 (7961 urls, method direct)."""
import glob, subprocess, sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "curl_cffi", "trafilatura", "orjson", "rapidfuzz"], check=False)

def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) or glob.glob(f"/kaggle/working/ds/**/{pattern}", recursive=True)
    return Path(hits[0]) if hits else None

if find("scripts/r2ai2026_recrawl.py") is None:  # dataset folders uploaded as archives
    import shutil
    for arc in glob.glob("/kaggle/input/**/*.zip", recursive=True) + glob.glob("/kaggle/input/**/*.tar", recursive=True):
        shutil.unpack_archive(arc, f"/kaggle/working/ds/{Path(arc).stem}")
script = find("scripts/r2ai2026_recrawl.py")
targets = find("targets/ask392.tsv")
print("code:", script, "targets:", targets, flush=True)
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(script.parents[1]))
import r2ai2026_recrawl as rc

work = Path("/kaggle/working")
rc.TARGETS = targets
rc.STATE = work / "state.tsv"
rc.DOCS_RAW = work / "docs.jsonl"
sys.argv = ["r2ai2026_recrawl.py", "--method", "direct", "--progress", "500"] + ["--concurrency", "2", "--per-host", "2", "--delay", "1.0", "--kill-after", "12"]
rc.main()
