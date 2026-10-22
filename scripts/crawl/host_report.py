"""Per-host success/latency breakdown of a crawl log (docs.jsonl)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    per_host: dict[str, Counter] = defaultdict(Counter)
    rows = 0
    for line in open(args.docs, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        rows += 1
        host = record.get("host", "?")
        per_host[host]["ok" if record.get("ok") else record.get("error", "?")] += 1

    summary = []
    for host, counter in sorted(per_host.items(), key=lambda item: -sum(item[1].values()))[: args.top]:
        total = sum(counter.values())
        summary.append(
            {
                "host": host,
                "total": total,
                "ok": counter["ok"],
                "ok_rate": counter["ok"] / total,
                "403": counter["http_403"],
                "empty": counter["empty_text"],
                "robots": counter["robots_disallowed"],
                "other": total - counter["ok"] - counter["http_403"] - counter["empty_text"] - counter["robots_disallowed"],
            }
        )

    header = f"{'host':<40}{'total':>6}{'ok':>6}{'403':>6}{'empty':>6}{'robot':>6}{'other':>6}   ok%"
    print(f"rows={rows} hosts={len(per_host)}")
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['host']:<40}{row['total']:>6}{row['ok']:>6}{row['403']:>6}{row['empty']:>6}"
            f"{row['robots']:>6}{row['other']:>6}   {row['ok_rate']:>4.0%}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
