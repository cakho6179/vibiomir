"""End-to-end system runner: one YAML config, a fixed stage order, one cached output folder per stage.

    python -m r2ai2026 run --config configs/system_v1.yaml [--stages bm25,dense] [--force] [--dry-run]
                           [--smoke] [--allow-cpu] [--var name=value ...]

The runner is a thin orchestrator. Each stage is either a script that already produced scored submissions
(kaggle/*_exp/*.py, called with path arguments in a subprocess) or a small builtin. A stage is skipped when its
outputs exist and its resolved config is unchanged (``<work>/<stage>/.done.json``), so a run can be resumed or
split across machines (GPU stages on Kaggle, the rest anywhere) by copying stage folders.

Config keys (see configs/system_v1.yaml):
  data      corpus root: corpus/part-*.jsonl[.gz] ({"id","t","x"}) and optionally sections.parquet
  queries   query.parquet (id, query)
  work      run folder; stage outputs go to <work>/<stage>/
  prepare   true: stage a data root <work>/data = query.parquet + a link to ``data`` (the scripts expect both
            under one root); false: ``data`` already holds query.parquet (e.g. /kaggle/input)
  vars      extra template values
  stages    ordered mapping name -> {script, args, outputs, gpu, smoke_args} or {builtin, ...}
Templates in args and builtin options: {root} repo, {data} staged data root, {queries}, {work}, {out} this
stage's folder, {<stage>} another stage's folder, plus ``vars`` and --var. ``job`` (dict) is passed as JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RRF_K = 60


def log(msg: str) -> None:
    print(f"[system {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg.setdefault("prepare", True)
    cfg.setdefault("vars", {})
    return cfg


def fill(value, ctx: dict):
    """Substitute {name} templates in strings, recursively in lists and dicts."""
    if isinstance(value, str):
        out = value
        for k, v in ctx.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(value, list):
        return [fill(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, ctx) for k, v in value.items()}
    return value


def resolve(p: str | Path, base: Path = ROOT) -> Path:
    """Absolute path; a pattern with * (e.g. /kaggle/input/**/query.parquet) resolves to its first match."""
    s = str(p)
    if "*" in s:
        import glob

        hits = sorted(glob.glob(s if Path(s).is_absolute() else str(base / s), recursive=True))
        if not hits:
            raise SystemExit(f"no match for {s}")
        return Path(hits[0]).resolve()
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def link_dir(src: Path, dst: Path) -> None:
    """Directory link that needs no admin rights: a junction on Windows, a symlink elsewhere."""
    if dst.exists():
        return
    if platform.system() == "Windows":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def prepare_data(cfg: dict, work: Path) -> Path:
    data, queries = resolve(cfg["data"]), resolve(cfg["queries"])
    if not cfg["prepare"]:
        return data
    root = work / "data"
    root.mkdir(parents=True, exist_ok=True)
    link_dir(data, root / "store")
    if not (root / "query.parquet").exists():
        shutil.copyfile(queries, root / "query.parquet")
    return root


def builtin_cands(opts: dict, out: Path) -> None:
    """Weighted RRF (k=60) of ranking files -> cands.parquet (qid, doc_id, score, rank), top ``k`` per query."""
    parts = []
    for spec, w in zip(opts["systems"], opts.get("weights") or [1] * len(opts["systems"])):
        r = pd.read_parquet(spec, columns=["qid", "doc_id", "score"])
        if opts.get("vi_only"):
            r = r[r["doc_id"] <= 766441]
        r = r.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates(["qid", "doc_id"])
        r["rank"] = r.groupby("qid").cumcount()
        r = r[r["rank"] < opts.get("depth", 200)]
        parts.append(r.assign(score=w / (RRF_K + r["rank"] + 1))[["qid", "doc_id", "score"]])
    fused = pd.concat(parts).groupby(["qid", "doc_id"], as_index=False)["score"].sum()
    fused = fused.sort_values(["qid", "score"], ascending=[True, False])
    fused["rank"] = fused.groupby("qid").cumcount()
    fused[fused["rank"] < opts["k"]].to_parquet(out / "cands.parquet", index=False)


BUILTINS = {"cands": builtin_cands}


def stage_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run(cfg: dict, stages: list[str] | None = None, force: bool = False, dry_run: bool = False,
        smoke: bool = False, allow_cpu: bool = False, extra_vars: dict | None = None) -> dict:
    work = resolve(cfg["work"])
    work.mkdir(parents=True, exist_ok=True)
    data = work / "data" if cfg["prepare"] else resolve(cfg["data"])
    if not dry_run:
        data = prepare_data(cfg, work)
    names = list(cfg["stages"])
    ctx = {"root": ROOT.as_posix(), "data": data.as_posix(), "queries": resolve(cfg["queries"]).as_posix(),
           "work": work.as_posix(), **{n: (work / n).as_posix() for n in names},
           **cfg["vars"], **(extra_vars or {})}
    todo = names if not stages else [n for n in names if n in stages]
    unknown = set(stages or []) - set(names)
    if unknown:
        raise SystemExit(f"unknown stages: {sorted(unknown)}; known: {names}")

    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
    manifest.update({"config": cfg.get("name", ""), "git": git_commit(), "python": sys.version.split()[0]})
    for name in todo:
        spec = fill(cfg["stages"][name] or {}, {**ctx, "out": ctx[name]})
        out = work / name
        done = out / ".done.json"
        h = stage_hash(spec)
        outputs = [out / o for o in spec.get("outputs", [])]
        if not force and done.exists() and json.loads(done.read_text()).get("hash") == h \
                and all(o.exists() for o in outputs):
            log(f"{name}: cached")
            continue
        if spec.get("builtin"):
            cmd = None
        else:
            args = [str(a) for a in spec.get("args", [])] + ([str(a) for a in spec.get("smoke_args", [])]
                                                             if smoke else [])
            if "job" in spec:
                args += ["--jobs", json.dumps([spec["job"]])]
            cmd = [cfg.get("python", sys.executable), str(resolve(spec["script"]))] + args
        if dry_run:
            log(f"{name}: would run " + (f"builtin {spec['builtin']}" if cmd is None else " ".join(cmd)))
            continue
        if spec.get("gpu") and not has_gpu() and not allow_cpu:
            raise SystemExit(f"{name} needs a GPU (pass --allow-cpu to run it on CPU anyway)")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log(f"{name}: start")
        if cmd is None:
            BUILTINS[spec["builtin"]](spec, out)
        else:
            res = subprocess.run(cmd, cwd=str(ROOT))
            if res.returncode:
                raise SystemExit(f"{name} failed (exit {res.returncode})")
        missing = [str(o) for o in outputs if not o.exists()]
        if missing:
            raise SystemExit(f"{name} finished without its outputs: {missing}")
        sec = round(time.time() - t0)
        done.write_text(json.dumps({"hash": h, "seconds": sec, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
        manifest["stages"][name] = {"hash": h, "seconds": sec, "smoke": smoke}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        log(f"{name}: done in {sec}s")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r2ai2026 run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stages", default="", help="comma-separated subset, in config order")
    ap.add_argument("--force", action="store_true", help="rerun stages even when cached")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--smoke", action="store_true", help="append each stage's smoke_args (small data)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--var", action="append", default=[], help="name=value template override")
    args = ap.parse_args(argv)
    cfg = load_config(resolve(args.config, Path.cwd()))
    extra = dict(v.split("=", 1) for v in args.var)
    for key in ("data", "queries", "work"):  # top-level paths can be overridden too
        if key in extra:
            cfg[key] = extra.pop(key)
    run(cfg, [s for s in args.stages.split(",") if s] or None, args.force, args.dry_run, args.smoke,
        args.allow_cpu, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
