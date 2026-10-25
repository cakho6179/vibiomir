"""Upload submission zips to the Codabench server and wait for their scores.

    python scripts/submit/submit.py submissions/a.zip [submissions/b.zip ...] [--dry-run] [--no-wait]

Same credentials as fetch_scores.py (never printed). Every zip is checked before upload: readable, 1200 unique query
ids, well formed (r2ai2026.core.submission.validate_submission), no binary chunk text (they make the scorer fail), <= 100 MB.
The public phase allows 10 submissions a day; the script stops before going over --max-today (non-Failed
submissions of ours created today, UTC; if the server also counts Failed ones it simply rejects the extra upload).
The private phase (5 in total) is refused unless --private-ok is given,
which is only for files the user approved one by one. Each upload is appended to runs/reports/submit_log.jsonl.
Flow of the web UI (codabench src/static/js/ours/client.js): POST /api/datasets/ -> PUT file to sassy_url ->
PUT /api/datasets/completed/<key>/ -> POST /api/submissions/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from fetch_scores import BASE, COMPETITION_ID, OWNER, auth_token, get_all, scores  # noqa: E402
from r2ai2026.corpus.cleaning import is_binary_text  # noqa: E402
from r2ai2026.core.submission import read_submission_zip, submission_stats, validate_submission  # noqa: E402

PUBLIC_PHASE, PRIVATE_PHASE = 44, 45
N_QUERIES = 1200
MAX_ZIP_MB = 100
LOG = ROOT / "runs" / "reports" / "submit_log.jsonl"


def check_zip(path: Path) -> tuple[list[str], dict]:
    mb = path.stat().st_size / 1_048_576
    problems = [f"zip {mb:.1f} MB > {MAX_ZIP_MB} MB"] if mb > MAX_ZIP_MB else []
    records = read_submission_zip(path)
    problems += validate_submission(records)
    if len({r.get("id") for r in records}) != N_QUERIES:
        problems.append(f"{len(records)} records, expected {N_QUERIES} unique query ids")
    n_bin = sum(is_binary_text(c["chunk_text"]) for r in records for c in r.get("relevant_chunks", []))
    if n_bin:
        problems.append(f"{n_bin} binary chunk texts (run scripts/submit/drop_binary_chunks.py)")
    st = submission_stats(records)
    return problems, {"zip_mb": round(mb, 1), "docs_per_q": round(st["docs_per_query"], 2),
                      "chunks_per_q": round(st["chunks_per_query"], 2)}


def submitted_today(client: httpx.Client, phase: int) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    subs = get_all(client, f"{BASE}/api/submissions/?phase={phase}")
    return sum(s["owner"] == OWNER and s["created_when"][:10] == today and s["status"] != "Failed" for s in subs)


def upload(client: httpx.Client, path: Path, phase: int, task_ids: list[int]) -> int:
    r = client.post(f"{BASE}/api/datasets/", json={"type": "submission", "competition": COMPETITION_ID,
                                                   "request_sassy_file_name": path.name, "file_size": path.stat().st_size})
    if r.status_code >= 400:
        raise SystemExit(f"create dataset failed: HTTP {r.status_code} {r.text[:300]}")
    ds = r.json()
    headers = {"Content-Type": "application/zip"}
    if "blob.core.windows.net" in ds["sassy_url"]:
        headers |= {"x-ms-blob-type": "BlockBlob", "x-ms-version": "2018-03-28"}
    # the presigned storage URL must not receive our API token
    put = httpx.put(ds["sassy_url"], content=path.read_bytes(), headers=headers, timeout=1200)
    put.raise_for_status()
    client.put(f"{BASE}/api/datasets/completed/{ds['key']}/").raise_for_status()
    r = client.post(f"{BASE}/api/submissions/", json={"data": ds["key"], "phase": phase, "tasks": task_ids,
                                                      "fact_sheet_answers": None, "organization": None,
                                                      "queue": None, "selected_groups": []})
    if r.status_code >= 400:
        raise SystemExit(f"create submission failed: HTTP {r.status_code} {r.text[:300]}")
    return int(r.json()["id"])


def wait_score(client: httpx.Client, sub_id: int, timeout_s: int) -> dict:
    t0 = time.time()
    while True:
        s = client.get(f"{BASE}/api/submissions/{sub_id}/").json()
        if s["status"] in ("Finished", "Failed", "Cancelled") or time.time() - t0 > timeout_s:
            return {"status": s["status"], **scores(s)}
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--phase", type=int, default=PUBLIC_PHASE)
    ap.add_argument("--private-ok", action="store_true", help="required for the private phase (5 submissions total)")
    ap.add_argument("--max-today", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="check the zips and the daily count, upload nothing")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--wait-min", type=int, default=60)
    args = ap.parse_args()
    if args.phase != PUBLIC_PHASE and not args.private_ok:
        raise SystemExit(f"phase {args.phase} is not the public phase: pass --private-ok only for user-approved files")

    for z in args.zips:
        problems, info = check_zip(z)
        print(z.name, json.dumps(info), "OK" if not problems else problems, flush=True)
        if problems:
            raise SystemExit(f"refusing to submit {z.name}")

    with httpx.Client(timeout=120) as client:
        token = auth_token(client)
        if not token:
            raise SystemExit("no Codabench credentials (see fetch_scores.py)")
        client.headers["Authorization"] = f"Token {token}"
        comp = client.get(f"{BASE}/api/competitions/{COMPETITION_ID}/").json()
        phase = next(p for p in comp["phases"] if p["id"] == args.phase)
        task_ids = [t["id"] for t in phase["tasks"]]
        cap = min(args.max_today, phase.get("max_submissions_per_day") or args.max_today)
        used = submitted_today(client, args.phase)
        print(f"phase {args.phase} ({phase['name']}): {used}/{cap} used today (UTC)", flush=True)
        if used + len(args.zips) > cap:
            raise SystemExit(f"{len(args.zips)} zips would exceed {cap}/day")
        if args.dry_run:
            return
        for z in args.zips:
            sub_id = upload(client, z, args.phase, task_ids)
            print(f"{z.name}: submitted #{sub_id}", flush=True)
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "file": str(z), "phase": args.phase, "id": sub_id}) + "\n")
        if args.no_wait:
            return
        # scores arrive in submission order; wait on each one in turn
        for line in LOG.read_text(encoding="utf-8").splitlines()[-len(args.zips):]:
            rec = json.loads(line)
            print(Path(rec["file"]).name, f"#{rec['id']}", json.dumps(wait_score(client, rec["id"], args.wait_min * 60)),
                  flush=True)


if __name__ == "__main__":
    main()
