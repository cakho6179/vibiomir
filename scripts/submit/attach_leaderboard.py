"""Put one of our submissions on its phase leaderboard (Codabench "Add" rule; replaces the one shown before).

    python scripts/submit/attach_leaderboard.py 7892            # attach
    python scripts/submit/attach_leaderboard.py 7892 --check    # only show the leaderboard field of our submissions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from fetch_scores import BASE, OWNER, auth_token, get_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sid", type=int)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with httpx.Client(timeout=60) as client:
        token = auth_token(client)
        if not token:
            raise SystemExit("no Codabench credentials")
        client.headers["Authorization"] = f"Token {token}"
        sub = client.get(f"{BASE}/api/submissions/{args.sid}/").json()
        if sub.get("owner") != OWNER:
            raise SystemExit(f"#{args.sid} is not ours")
        if not args.check:
            r = client.post(f"{BASE}/api/submissions/{args.sid}/submission_leaderboard_connection/")
            print(f"attach #{args.sid}: HTTP {r.status_code} {r.text[:200]}")
        for s in get_all(client, f"{BASE}/api/submissions/?phase={sub['phase']}"):
            if s["owner"] == OWNER:
                print(f"#{s['id']} {s['filename']} status={s['status']} leaderboard={s.get('leaderboard')} "
                      f"on_leaderboard={s.get('on_leaderboard')}")


if __name__ == "__main__":
    main()
