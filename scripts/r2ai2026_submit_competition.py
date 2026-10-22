"""Submit a submission ZIP to the R2AI2026 Codabench competition.

    # what the API says about this competition (no token needed)
    python scripts/r2ai2026_submit_competition.py --probe

    # validate the ZIP locally, then upload (needs an account token)
    python scripts/r2ai2026_submit_competition.py --zip runs/submissions/submission_v1.zip \
        --token $env:CODABENCH_TOKEN

    # collect the score of a submission that was uploaded by hand
    python scripts/r2ai2026_submit_competition.py --status 12345 --token $env:CODABENCH_TOKEN

Measured on 2026-10-19: this competition sets ``allow_robot_submissions = false``,
so the upload endpoint answers "Robot submissions are not allowed" -- the ZIP then
has to be uploaded through the web UI (Submit tab), and ``--status`` still lets us
track the score once it is queued.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai2026.submit import read_submission_zip, submission_stats, validate_submission  # noqa: E402

BASE = "https://leaderboard.aiguru.com.vn"
COMPETITION_ID = 15


def api_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def get_competition(token: str | None) -> dict:
    response = httpx.get(f"{BASE}/api/competitions/{COMPETITION_ID}/", headers=api_headers(token), timeout=60.0)
    response.raise_for_status()
    return response.json()


def current_phase(competition: dict) -> dict | None:
    for phase in competition.get("phases", []):
        if phase.get("status") == "Current":
            return phase
    return None


def probe(token: str | None) -> int:
    competition = get_competition(token)
    phase = current_phase(competition)
    print(f"competition {competition['id']}: {competition['title']}")
    print(f"allow_robot_submissions: {competition.get('allow_robot_submissions')}")
    if phase:
        print(f"current phase {phase['id']}: {phase['name']} ({phase['start']} -> {phase['end']})")
        print(f"execution_time_limit: {phase.get('execution_time_limit')}s")
    else:
        print("no phase marked 'Current' -- the public phase may have closed")
    if not competition.get("allow_robot_submissions"):
        print("=> API uploads are disabled; the ZIP must be uploaded from the web UI:")
        print(f"   {BASE}/competitions/{COMPETITION_ID}/#/participate")
    return 0


def validate(zip_path: Path) -> int:
    records = read_submission_zip(zip_path)
    problems = validate_submission(records)
    stats = submission_stats(records)
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")
    print(f"stats: {json.dumps(stats, indent=2)}")
    if problems:
        print(f"INVALID: {len(problems)} problems, first: {problems[:3]}")
        return 1
    print("validation: ok")
    return 0


def submit(zip_path: Path, token: str, name: str, description: str) -> int:
    competition = get_competition(token)
    phase = current_phase(competition)
    if phase is None:
        print("refusing to submit: the competition has no current phase")
        return 1
    with open(zip_path, "rb") as handle:
        files = {
            "file": (zip_path.name, handle, "application/zip"),
        }
        data = {"phase": str(phase["id"]), "name": name, "description": description}
        response = httpx.post(
            f"{BASE}/api/competitions/{COMPETITION_ID}/submissions/",
            headers=api_headers(token),
            data=data,
            files=files,
            timeout=600.0,
        )
    print(f"POST -> {response.status_code}")
    print(response.text[:1200])
    if response.status_code in (200, 201):
        payload = response.json()
        print(f"submission id: {payload.get('id')} status: {payload.get('status')}")
        return 0
    return 1


def status(submission_id: int, token: str | None) -> int:
    response = httpx.get(f"{BASE}/api/submissions/{submission_id}/", headers=api_headers(token), timeout=60.0)
    if response.status_code != 200:
        print(f"GET -> {response.status_code}: {response.text[:400]}")
        return 1
    payload = response.json()
    keys = ("id", "status", "detailed_status", "created_when", "execution_time", "scores")
    print(json.dumps({key: payload.get(key) for key in keys}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=ROOT / "runs" / "submissions" / "submission_v1.zip")
    parser.add_argument("--token", default=os.environ.get("CODABENCH_TOKEN", ""))
    parser.add_argument("--probe", action="store_true", help="print competition/phase metadata and stop")
    parser.add_argument("--validate-only", action="store_true", help="validate the local ZIP and stop")
    parser.add_argument("--status", type=int, default=None, help="print the status/score of a submission id")
    parser.add_argument("--name", default="bm25-v1")
    parser.add_argument("--description", default="BM25 over the crawled multilingual corpus")
    args = parser.parse_args()

    if args.probe:
        return probe(args.token or None)
    if args.status is not None:
        return status(args.status, args.token or None)

    code = validate(args.zip)
    if code != 0 or args.validate_only:
        return code
    if not args.token:
        print("no --token/CODABENCH_TOKEN given: validated only, nothing uploaded")
        return 0
    return submit(args.zip, args.token, args.name, args.description)


if __name__ == "__main__":
    raise SystemExit(main())
