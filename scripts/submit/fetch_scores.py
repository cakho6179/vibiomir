"""Fetch leaderboard and our submission scores from the Codabench server (read-only, never submits).

    python scripts/submit/fetch_scores.py      # leaderboard + every submission of ours when credentials exist

Credentials (never printed, never in the repo), first found wins:
  $env:CODABENCH_TOKEN | ~/.codabench_token (one line) | ~/.codabench.json {"username": "...", "password": "..."}
(the last one is exchanged for a token at /api/api-token-auth/). Without them the server only exposes each team's
leaderboard entry. Writes runs/reports/codabench_scores.json
and prints rows in the format of EXPERIMENTS.md table A.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
BASE = "https://leaderboard.aiguru.com.vn"
COMPETITION_ID = 15
OWNER = "ca_kho"  # our Codabench username; the leaderboard shows the team name instead
TEAM = "Băng Cá Khô"
COLS = ["FINAL_SCORE", "DOCS_F2MACRO", "CHUNKS_F2MACRO", "DOCS_PRECISION", "DOCS_RECALL", "CHUNKS_PRECISION",
        "CHUNKS_RECALL"]


def scores(sub: dict) -> dict[str, float]:
    return {s["column_key"]: float(s["score"]) for s in sub.get("scores", [])}


def get_all(client: httpx.Client, url: str) -> list[dict]:
    out = []
    while url:
        d = client.get(url.replace("http://", "https://")).json()
        out += d["results"]
        url = d["next"]
    return out


def auth_token(client: httpx.Client) -> str | None:
    if os.environ.get("CODABENCH_TOKEN"):
        return os.environ["CODABENCH_TOKEN"].strip()
    token_file = Path.home() / ".codabench_token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    cred_file = Path.home() / ".codabench.json"
    if cred_file.exists():
        cred = json.loads(cred_file.read_text(encoding="utf-8"))
        r = client.post(f"{BASE}/api/api-token-auth/", data={"username": cred["username"], "password": cred["password"]})
        if r.status_code != 200:
            raise SystemExit(f"login failed: HTTP {r.status_code} (check ~/.codabench.json)")
        return r.json()["token"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, default=0, help="phase id (default: the current one)")
    args = ap.parse_args()
    with httpx.Client(timeout=60) as client:
        token = auth_token(client)
        headers = {"Authorization": f"Token {token}"} if token else {}
        client.headers.update(headers)
        comp = client.get(f"{BASE}/api/competitions/{COMPETITION_ID}/").json()
        phase = args.phase or next(p["id"] for p in comp["phases"] if p.get("status") == "Current")
        board = client.get(f"{BASE}/api/phases/{phase}/get_leaderboard/").json()["submissions"]
        subs = get_all(client, f"{BASE}/api/submissions/?phase={phase}")
    ranked = sorted(board, key=lambda s: -scores(s)["FINAL_SCORE"])
    rank = next((i for i, s in enumerate(ranked, 1) if s["owner"] in (OWNER, TEAM)), None)
    mine = sorted((s for s in subs if s["owner"] == OWNER), key=lambda s: s["created_when"])
    print(f"phase {phase}: {len(ranked)} teams; top {scores(ranked[0])['FINAL_SCORE']} ({ranked[0]['owner']}); "
          f"ours rank {rank}; our submissions visible: {len(mine)} ({'token' if headers else 'public API only'})")
    for s in mine:
        d = scores(s)
        print(f"| {s['created_when'][8:10]}/{s['created_when'][5:7]} | {s['filename']} | {s['status']} #{s['id']} | | "
              f"{d.get('FINAL_SCORE')} | {d.get('DOCS_PRECISION')} / {d.get('DOCS_RECALL')} | "
              f"{d.get('CHUNKS_PRECISION')} / {d.get('CHUNKS_RECALL')} |")
    out = ROOT / "runs/reports/codabench_scores.json"
    out.write_text(json.dumps({"phase": phase, "rank": rank, "teams": len(ranked),
                               "leaderboard": [{"owner": s["owner"], **scores(s)} for s in ranked],
                               "ours": [{"id": s["id"], "file": s["filename"], "when": s["created_when"],
                                         "status": s["status"], **scores(s)} for s in mine]},
                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
