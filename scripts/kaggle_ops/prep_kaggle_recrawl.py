"""Shard the remaining re-crawl work into Kaggle CPU kernels (one IP each).

Builds
  kaggle/recrawl/dataset/   -> private dataset <user>/vibiomir-recrawl-targets
      code/r2ai2026/...           the same package the local crawler uses
      code/scripts/r2ai2026_recrawl.py
      targets/<shard>.tsv         id, url, host, reason (no header)
  kaggle/recrawl/kernels/<shard>/run.py + kernel-metadata.json

Each kernel runs ``r2ai2026_recrawl.py`` unchanged, only pointing its target/output
paths at /kaggle; outputs land in /kaggle/working/{docs.jsonl,state.tsv,run.log}.

    python scripts/r2ai2026_prep_kaggle_recrawl.py --user snnguynvnk19hl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.paths import REPORTS  # noqa: E402

OUT = ROOT / "kaggle" / "recrawl"
DATASET_SLUG = "vibiomir-recrawl-targets"
# outcomes after which a method will not be retried for an id
FINAL = {"ok", "no_snapshot", "too_short", "duplicate_template", "blocked_or_404"}

# group name -> (method, host filter, n_shards, extra recrawl args)
PLAN = {
    "lc": ("wayback", lambda h: h == "nhathuoclongchau.com.vn", 4,
           ["--concurrency", "6", "--per-host", "6", "--delay", "0.3", "--kill-after", "60"]),
    "ask39": ("direct", lambda h: h == "ask.39.net", 4,
              ["--concurrency", "2", "--per-host", "2", "--delay", "1.0", "--kill-after", "12"]),
    "misc": ("direct", lambda h: h in {
        "www.120ask.com", "zysjonline.com", "pmc-ecm-healthblog.beta.pharmacity.io",
        "thaythuocvietnam.vn", "www.familydoctor.com.cn", "www.msdmanuals.cn",
        "www.youlai.cn", "article.iiyi.com", "baby.39.net"}, 1,
        ["--concurrency", "8", "--per-host", "2", "--delay", "0.7", "--kill-after", "10"]),
}

RUN_PY = '''"""Kaggle shard runner for the ViBioMIR re-crawl: __SHARD__ (__N__ urls, method __METHOD__)."""
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
targets = find("targets/__SHARD__.tsv")
print("code:", script, "targets:", targets, flush=True)
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(script.parents[1]))
import r2ai2026_recrawl as rc

work = Path("/kaggle/working")
rc.TARGETS = targets
rc.STATE = work / "state.tsv"
rc.DOCS_RAW = work / "docs.jsonl"
sys.argv = ["r2ai2026_recrawl.py", "--method", "__METHOD__", "--progress", "500"] + __ARGS__
rc.main()
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True)
    args = ap.parse_args()

    targets = pd.read_csv(REPORTS / "recrawl_targets.tsv", sep="\t", header=None,
                          names=["id", "url", "host", "reason"])
    state = pd.read_csv(REPORTS / "recrawl_state.tsv", sep="\t", header=None,
                        names=["id", "method", "outcome", "detail"], dtype=str, on_bad_lines="skip")
    state["id"] = pd.to_numeric(state["id"], errors="coerce")
    done = state[state["outcome"].isin(FINAL)]
    done_ok = set(done.loc[done["outcome"] == "ok", "id"].dropna().astype(int))

    if OUT.exists():
        shutil.rmtree(OUT)
    ds = OUT / "dataset"
    (ds / "targets").mkdir(parents=True)
    shutil.copytree(ROOT / "r2ai2026", ds / "code" / "r2ai2026",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (ds / "code" / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "crawl" / "recrawl.py", ds / "code" / "scripts" / "r2ai2026_recrawl.py")

    kernels = []
    for group, (method, host_ok, n_shards, extra) in PLAN.items():
        tried = set(done.loc[done["method"] == method, "id"].dropna().astype(int))
        sel = targets[targets["host"].map(host_ok) & ~targets["id"].isin(tried | done_ok)]
        sel = sel.sample(frac=1, random_state=7)  # spread hosts/ids evenly over shards
        for i in range(n_shards):
            part = sel.iloc[i::n_shards]
            if part.empty:
                continue
            shard = f"{group}{i + 1}"
            part.to_csv(ds / "targets" / f"{shard}.tsv", sep="\t", header=False, index=False)
            kdir = OUT / "kernels" / shard
            kdir.mkdir(parents=True)
            code = (RUN_PY.replace("__SHARD__", shard).replace("__N__", str(len(part)))
                    .replace("__METHOD__", method).replace("__ARGS__", json.dumps(extra)))
            (kdir / "run.py").write_text(code, encoding="utf-8")
            meta = {
                "id": f"{args.user}/vibiomir-recrawl-{shard}",
                "title": f"vibiomir-recrawl-{shard}",
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": True,
                "dataset_sources": [f"{args.user}/{DATASET_SLUG}"],
                "competition_sources": [], "kernel_sources": [],
            }
            (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            kernels.append((shard, method, len(part)))

    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{args.user}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    for shard, method, n in kernels:
        print(f"{shard:8} {method:8} {n:>7,} urls")
    print(f"total {sum(n for *_, n in kernels):,} urls in {len(kernels)} kernels -> {OUT}")


if __name__ == "__main__":
    main()
