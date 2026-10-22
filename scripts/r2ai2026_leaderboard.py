"""Read the public Codabench standings for the R2AI2026 shared task.

The leaderboard feed is public (``can_participants_make_submissions_public`` is
true and the UI shows every entry to every participant), so this prints the
aggregate columns only. Per-query detailed results belong to the team that
submitted them; this tool never pulls them for anybody else, and uses the
scores of our own submission solely to compare our pipeline against the field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://leaderboard.aiguru.com.vn"
COMPETITION_ID = 15
LEADERBOARD_ID = 93


def fetch_json(url: str, timeout: float = 240.0) -> dict:
    last: Exception | None = None
    for _ in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise SystemExit(f"could not read {url}: {type(last).__name__}: {last}")


def score_map(entry: dict) -> dict[str, float]:
    return {score["column_key"]: float(score["score"]) for score in entry.get("scores") or []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=12, help="how many standings rows to print")
    parser.add_argument("--submission", type=int, default=None, help="print one submission in full")
    parser.add_argument("--save", type=Path, default=None, help="write the raw standings JSON here")
    args = parser.parse_args()

    if args.submission is not None:
        detail = fetch_json(f"{BASE}/api/submissions/{args.submission}/")
        print(json.dumps({key: detail.get(key) for key in
                          ("id", "name", "owner", "phase_name", "created_when", "filename", "on_leaderboard")},
                         ensure_ascii=False, indent=2))
        for score in detail.get("scores") or []:
            print(f"  {score['column_key']:16s} {score['score']}")
        return 0

    feed = fetch_json(f"{BASE}/api/leaderboards/{LEADERBOARD_ID}/")
    entries = feed.get("submissions") or []
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(feed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {len(entries)} entries -> {args.save}")

    ordered = sorted(entries, key=lambda entry: score_map(entry).get("FINAL_SCORE", 0.0), reverse=True)
    columns: list[str] = []
    for entry in ordered:
        for key in score_map(entry):
            if key not in columns:
                columns.append(key)
    header = ["rank", "owner", *columns]
    print("\t".join(header))
    for rank, entry in enumerate(ordered[: args.top], start=1):
        scores = score_map(entry)
        row = [str(rank), str(entry.get("owner") or entry.get("display_name") or "?")]
        row.extend(f"{scores.get(column, float('nan')):.4f}" for column in columns)
        print("\t".join(row))
    print(f"\nentries: {len(ordered)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
