"""Refresh the R2AI2026 ViBioMIR competition workspace from live sources.

Pulls fresh Codabench competition/leaderboard data plus the HuggingFace dataset
inventory, re-renders the workspace knowledge base, re-audits the local corpus
snapshot and rewrites the control-plane state (hunt/execution/portfolio/plan).

Usage:
    python scripts/refresh_r2ai2026_workspace.py [--offline] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_hunter.ingestion.corpus_audit import (  # noqa: E402
    audit_retrieval_data,
    render_audit_markdown,
)
from competition_hunter.models import (  # noqa: E402
    CompetitionContext,
    CompetitionFile,
    CompetitionHost,
    CompetitionSnapshot,
    FileRole,
    now_iso,
)
from competition_hunter.strategy.retrieval import build_retrieval_experiment_plan  # noqa: E402
from competition_hunter.workspace import write_json  # noqa: E402

COMP_ID = 15
LEADERBOARD_ID = 93
BASE = "https://leaderboard.aiguru.com.vn"
COMP_URL = f"{BASE}/competitions/{COMP_ID}/"
SLUG = "r2ai2026-vibiomir"
HF_DATASET = "AIGuruTinix/ViBioMIR"
HF_CORPUS_FILE = "links_corpus_vi_en_zh.parquet"
HF_QUERY_FILE = "query.parquet"
ICT = timezone(timedelta(hours=7))
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

METRIC = (
    "FINAL_SCORE = mean(DOCS_F2MACRO, CHUNKS_F2MACRO); per-query Precision/Recall/F2 "
    "macro-averaged over all queries, scored separately at document level and chunk level"
)
RULES_SUMMARY = (
    "One account per user; one team max per participant; one submission account per team. "
    "External data allowed but must be cited with full provenance. "
    "Open-weight PLMs/LLMs only (model and/or training data public, e.g. Hugging Face); "
    "closed LLMs (GPT-4o, Gemini, ...) forbidden; models must be released before "
    "2026-08-01 (ICT) and be <= 15B parameters. "
    "Max 10 submissions/day in public phase; max 5 submissions total in private phase. "
    "Final ranking requires a working-notes paper describing the method. "
    "Organizers may disqualify non-compliant teams; winners grant usage rights over "
    "submissions, code and generated data."
)


# ----------------------------------------------------------------------------
# network helpers
# ----------------------------------------------------------------------------
def fetch_json(url: str, retries: int = 3, timeout: int = 90) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(2 + attempt * 3)
    return {"__fetch_error__": f"{type(last_error).__name__}: {last_error}"}


def fetch_text(url: str, retries: int = 3, timeout: int = 90) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2 + attempt * 3)
    return f"__fetch_error__: {type(last_error).__name__}: {last_error}"


def html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", raw)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h1|h2|h3|h4|ul|ol|pre|table|code)>", "\n", text)
    text = re.sub(r"(?i)</t[dh]>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ |]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_ict(iso_stamp: str) -> str:
    if not iso_stamp:
        return ""
    try:
        parsed = datetime.fromisoformat(iso_stamp.replace("Z", "+00:00"))
    except ValueError:
        return iso_stamp
    return parsed.astimezone(ICT).strftime("%Y-%m-%d %H:%M ICT")


# ----------------------------------------------------------------------------
# workspace loading / merging
# ----------------------------------------------------------------------------
def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
    return default


def leaderboard_entries(leaderboard: dict) -> list[dict]:
    entries = []
    for sub in leaderboard.get("submissions", []) or []:
        scores = {s["column_key"]: float(s["score"]) for s in (sub.get("scores") or [])}
        entries.append(
            {
                "submission_id": sub.get("id"),
                "display_name": sub.get("display_name"),
                "owner": sub.get("owner"),
                "created_when": sub.get("created_when"),
                "scores": scores,
                "final_score": scores.get("FINAL_SCORE"),
            }
        )
    entries.sort(key=lambda e: e["final_score"] if e["final_score"] is not None else -1, reverse=True)
    return entries


def entry_final(entry: dict) -> float | None:
    """Final score of a leaderboard entry, tolerating older snapshot shapes."""
    if entry.get("final_score") is not None:
        return entry["final_score"]
    scores = entry.get("scores") or {}
    value = scores.get("FINAL_SCORE")
    return float(value) if value is not None else None


def pick_baseline(entries: list[dict]) -> dict:
    for entry in entries:
        if (entry["display_name"] or "").strip().lower().startswith("bm25"):
            return entry
    return entries[0] if entries else {}


def pick_best_competitor(entries: list[dict]) -> dict:
    for entry in entries:
        name = (entry["display_name"] or "").strip().lower()
        if name.startswith("bm25") or name in ("admin", "baseline"):
            continue
        return entry
    return {}


# ----------------------------------------------------------------------------
# renderers
# ----------------------------------------------------------------------------
def render_brief(
    comp: dict,
    entries: list[dict],
    hf_info: dict,
    hf_tree: list[dict],
    audit: dict | None,
) -> str:
    phases = comp.get("phases", [])
    public_phase = phases[0] if phases else {}
    private_phase = phases[1] if len(phases) > 1 else {}
    baseline = pick_baseline(entries)
    best = pick_best_competitor(entries)
    base_scores = baseline.get("scores", {})
    best_scores = best.get("scores", {})

    lines = [
        f"# {comp.get('title', SLUG)} — Competition Brief (refreshed)",
        "",
        f"- Slug: `{SLUG}` | Host: Codabench | Competition id: {COMP_ID} | Leaderboard id: {LEADERBOARD_ID}",
        f"- URL: {COMP_URL} (tab: #/pages-tab)",
        f"- Description: {comp.get('description', '')}",
        f"- Participants: **{comp.get('participants_count')}** | Submissions: **{comp.get('submissions_count')}** | Created: {comp.get('created_when', '')}",
        f"- Docker image: `{comp.get('docker_image', '')}` | Execution time limit: 600s | Robot submissions: {comp.get('allow_robot_submissions')}",
        f"- Refreshed at: {now_iso()}",
        "",
        "## Timeline (all deadlines 23:59 ICT unless noted)",
        "",
        "| Milestone | Window |",
        "|---|---|",
        f"| Public phase (platform api) | {public_phase.get('start', '')} -> {public_phase.get('end', '')} |",
        f"| Public phase (ICT) | {to_ict(public_phase.get('start', ''))} -> **{to_ict(public_phase.get('end', ''))}** |",
        f"| Private phase (platform api) | {private_phase.get('start', '')} -> {private_phase.get('end', '')} |",
        f"| Private phase (ICT) | {to_ict(private_phase.get('start', ''))} -> **{to_ict(private_phase.get('end', ''))}** |",
        "| Public submission deadline (doc) | 2026-10-31 23:59 ICT |",
        "| Final system deadline (doc) | 2026-11-04 23:59 ICT |",
        "| Results announced | 2026-11-11 |",
        "",
        "> Platform phase windows close earlier than the documented deadlines "
        "(public phase ends at the platform `end` above, i.e. same calendar day but "
        "not 23:59). Treat the platform window as the binding constraint.",
        "",
        "## Data (HuggingFace: https://huggingface.co/datasets/" + HF_DATASET + ")",
        "",
        f"- HF repo created/last modified: {hf_info.get('createdAt', '?')} / {hf_info.get('lastModified', '?')}"
        f" | downloads: {hf_info.get('downloads', '?')} | gated: {hf_info.get('gated', '?')}",
        f"- Files: {', '.join(f.get('path', '') for f in hf_tree if f.get('type') == 'file')}",
        f"- `{HF_QUERY_FILE}`: `id:int`, `query:str` — Vietnamese medical queries (both phases).",
        f"- `{HF_CORPUS_FILE}`: `id:int`, `url:str` — source article URLs; teams crawl/clean/chunk/index themselves.",
        "- Ground-truth doc labels use the integer `id` in the corpus file; documents outside that list are never positive.",
        "",
    ]

    if audit:
        query = audit["query"]
        corpus = audit["corpus"]
        lines += [
            "### Corpus snapshot (local audit)",
            "",
            f"- Queries: **{query['rows']:,}** (median {query['words_median']} words, p95 {query['words_p95']}, max {query['words_max']})",
            f"- Corpus rows: **{corpus['rows']:,}** | unique urls: {corpus['unique_urls']:,} | id range {corpus['id_min']}..{corpus['id_max']}",
            f"- Language heuristic: `{corpus['languages_heuristic']}`",
            f"- PubMed URLs (bulk-harvestable via E-utilities): **{corpus['pubmed_urls']:,}**",
            f"- Top hosts: " + ", ".join(f"{h} ({n:,})" for h, n in corpus["top_hosts"][:8]),
            "",
        ]

    lines += [
        "## Metric",
        "",
        "- Document level: Precision = |D∩G|/|D|, Recall = |D∩G|/|G|.",
        "- Chunk level: a submitted chunk counts as correct when LCS-token overlap with any reference "
        "chunk of the same query+doc is >= 0.40 (text normalized: NFKC, HTML entities, lowercase, whitespace).",
        "- F2 = 5·P·R / (4·P + R), macro-averaged per query at both levels.",
        f"- **{METRIC}**",
        "",
        "## Submission format",
        "",
        "Single `.json` array zipped alone (no folder), uploaded via My Submissions:",
        "",
        "```json",
        '[{"id": 1, "relevant_docs": [123], "relevant_chunks": [{"doc_id": 123, "chunk_text": "<verbatim>"}]}]',
        "```",
        "",
        "- Empty lists are mandatory when a level has no prediction; a submission missing queries is not evaluated (and burns no quota).",
        "- `chunk_text` must be verbatim text extracted from the source document.",
        "",
        "## Leaderboard (refreshed snapshot)",
        "",
        f"- Entries in snapshot: **{len(entries)}** (this is the public leaderboard feed, newest first by score).",
        "",
        "| Rank | Entry | FINAL | DOCS F2 | CHUNKS F2 | DOCS P/R | CHUNKS P/R | When |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for rank, entry in enumerate(entries[:15], start=1):
        s = entry["scores"]
        name = entry["display_name"] or "anonymous"
        lines.append(
            f"| {rank} | {name} | {s.get('FINAL_SCORE', 0):.4f} | {s.get('DOCS_F2MACRO', 0):.4f} "
            f"| {s.get('CHUNKS_F2MACRO', 0):.4f} "
            f"| {s.get('DOCS_PRECISION', 0):.4f}/{s.get('DOCS_RECALL', 0):.4f} "
            f"| {s.get('CHUNKS_PRECISION', 0):.4f}/{s.get('CHUNKS_RECALL', 0):.4f} "
            f"| {(entry['created_when'] or '')[:10]} |"
        )

    if baseline:
        lines += [
            "",
            f"- Admin baseline `{baseline['display_name']}`: FINAL {base_scores.get('FINAL_SCORE', 0):.4f} "
            f"(docs {base_scores.get('DOCS_F2MACRO', 0):.4f} / chunks {base_scores.get('CHUNKS_F2MACRO', 0):.4f}).",
        ]
    if best and best is not baseline:
        lines += [
            f"- Best non-admin entry: FINAL {best_scores.get('FINAL_SCORE', 0):.4f} "
            f"(docs {best_scores.get('DOCS_F2MACRO', 0):.4f} / chunks {best_scores.get('CHUNKS_F2MACRO', 0):.4f}).",
        ]

    lines += [
        "",
        "## Quotas",
        "",
        f"- Public phase: {public_phase.get('max_submissions_per_day')}/day per team "
        f"(cap {public_phase.get('max_submissions_per_person')} per person).",
        f"- Private phase: {private_phase.get('max_submissions_per_day')}/day — only "
        f"**{private_phase.get('max_submissions_per_person')} total** submissions per user.",
        "",
        "## Rules",
        "",
        RULES_SUMMARY,
        "",
    ]
    return "\n".join(lines)


def render_retrieval_plan(plan, comp: dict, entries: list[dict], audit: dict | None) -> str:
    phases = comp.get("phases", [])
    public_end = phases[0].get("end", "") if phases else ""
    private_end = phases[1].get("end", "") if len(phases) > 1 else ""
    baseline = pick_baseline(entries)

    lines = [
        f"# Retrieval plan — {SLUG}",
        "",
        f"- Generated: {now_iso()}",
        f"- Public phase closes: **{to_ict(public_end)}** ({public_end})",
        f"- Private phase closes: **{to_ict(private_end)}** ({private_end})",
        f"- Bar to clear: `{baseline.get('display_name', 'BM25')}` at "
        f"FINAL {baseline.get('scores', {}).get('FINAL_SCORE', 0):.4f} "
        f"(docs {baseline.get('scores', {}).get('DOCS_F2MACRO', 0):.4f}, chunks {baseline.get('scores', {}).get('CHUNKS_F2MACRO', 0):.4f})",
        "",
        "## Hypotheses",
        "",
    ]
    lines += [f"- {h}" for h in plan.hypotheses]

    lines += [
        "",
        "## Branches",
        "",
        "| # | Branch | Cost | Depends on | Objective |",
        "|---:|---|---|---|---|",
    ]
    for experiment in plan.experiments:
        deps = next((e.family for e in plan.experiments if e.experiment_id == experiment.parent_experiment_id), "")
        lines.append(
            f"| {experiment.priority} | `{experiment.family}` | {experiment.estimated_cost} | {deps or '-'} | {experiment.objective} |"
        )

    lines += [
        "",
        "## Branch detail",
        "",
    ]
    for experiment in plan.experiments:
        lines.append(f"### {experiment.priority}. `{experiment.family}` — {experiment.objective}")
        for label, notes in (
            ("Preprocessing", experiment.preprocessing),
            ("Work", experiment.training),
            ("Serving", experiment.inference),
        ):
            if notes:
                lines.append(f"- **{label}**")
                lines += [f"  - {note.item} — {note.reasoning}" for note in notes]
        lines.append("")

    if audit:
        corpus = audit["corpus"]
        lines += [
            "## Corpus reality check",
            "",
            f"- {corpus['rows']:,} URLs, {corpus['pubmed_urls']:,} of them PubMed (bulk harvest, no HTML crawl).",
            f"- Non-PubMed share: {corpus['rows'] - corpus['pubmed_urls']:,} URLs across {len(corpus['top_hosts'])}+ hosts.",
            "- Crawl budget must be planned per host: a handful of Vietnamese portals dominate the URL list, "
            "so per-host rate limits (not global concurrency) decide the wall-clock time.",
            "",
        ]

    lines += [
        "## Critical path",
        "",
        "1. **Corpus** — PubMed harvest + VI/ZH crawl; nothing downstream can beat a missing document.",
        "2. **Scorer replica** — reproduce normalization + token-LCS locally so tuning does not spend quota.",
        "3. **Sparse baseline** — match/beat the admin BM25 configuration on the local proxy.",
        "4. **Dense + hybrid + rerank** — recall then precision, each measured on the proxy.",
        "5. **Selection** — per-query depth tuned for F2, then submit.",
    ]
    return "\n".join(lines) + "\n"


def render_status_report(
    comp: dict,
    entries: list[dict],
    previous_entries: list[dict],
    audit: dict | None,
) -> str:
    phases = comp.get("phases", [])
    public_end = phases[0].get("end", "") if phases else ""
    days_left = ""
    if public_end:
        try:
            parsed = datetime.fromisoformat(public_end.replace("Z", "+00:00"))
            days_left = f"{(parsed - datetime.now(timezone.utc)).days} days"
        except ValueError:
            days_left = "?"
    previous_top = entry_final(previous_entries[0]) if previous_entries else None
    current_top = entry_final(entries[0]) if entries else None
    lines = [
        f"# Status — {SLUG}",
        "",
        f"- Refreshed: {now_iso()}",
        f"- Public phase closes: {to_ict(public_end)} ({days_left} left)",
        f"- Participants: {comp.get('participants_count')} | submissions: {comp.get('submissions_count')}",
        f"- Leaderboard entries captured: {len(entries)} (previous snapshot: {len(previous_entries)})",
        f"- Top FINAL score: {current_top:.4f}" if current_top is not None else "- Top FINAL score: n/a",
        f"- Previous top FINAL score: {previous_top:.4f}" if previous_top is not None else "- Previous top FINAL score: n/a",
        "",
        "## Where we stand",
        "",
        "- Workspace has research/knowledge + plan artifacts; no bundle, run or submission exists yet.",
        "- Blocking workstream: corpus acquisition (crawl) — everything else is downstream of it.",
        "",
        "## Next actions",
        "",
        "1. Produce the first sparse system on a crawled subset and validate the scorer replica locally.",
        "2. Scale corpus coverage (PubMed bulk + top VI/ZH hosts).",
        "3. Push a first real submission early — the quota (10/day) is only useful if used before the window closes.",
        "",
    ]
    if audit:
        corpus = audit["corpus"]
        lines += [
            "## Data facts",
            "",
            f"- {audit['query']['rows']:,} queries, {corpus['rows']:,} corpus URLs, id range {corpus['id_min']}..{corpus['id_max']}.",
            f"- PubMed URLs: {corpus['pubmed_urls']:,}.",
            "",
        ]
    return "\n".join(lines) + "\n"


def render_readme(comp: dict, entries: list[dict], audit: dict | None) -> str:
    phases = comp.get("phases", [])
    public_end = phases[0].get("end", "") if phases else ""
    private_end = phases[1].get("end", "") if len(phases) > 1 else ""
    baseline = pick_baseline(entries)
    lines = [
        f"# {comp.get('title', SLUG)}",
        "",
        f"- Slug: `{SLUG}` | Host: Codabench ({COMP_URL})",
        "- Task: Vietnamese queries -> multilingual (vi/en/zh) biomedical documents + verbatim chunks",
        f"- Metric: {METRIC}",
        f"- Public phase closes: {to_ict(public_end)} ({public_end})",
        f"- Private phase closes: {to_ict(private_end)} ({private_end})",
        f"- Bar to clear: `{baseline.get('display_name', 'BM25')}` FINAL "
        f"{baseline.get('scores', {}).get('FINAL_SCORE', 0):.4f}",
        "",
        "## Contents",
        "",
        "- `knowledge/competition_brief.md` — **start here**: refreshed brief (task, data, metric, rules, leaderboard)",
        "- `knowledge/corpus_audit.md` / `.json` — local audit of query + corpus parquet",
        "- `knowledge/leaderboard_snapshot.json` — parsed standings with capture timestamp",
        "- `knowledge/pages/*.md` — rendered official pages (Overview / Evaluation / Submission / Data / Terms)",
        "- `knowledge/codabench_*.json`, `knowledge/huggingface_dataset_raw.json` — raw API dumps",
        "- `plans/retrieval_plan.md` — retrieval branch plan (corpus -> sparse -> dense -> hybrid -> rerank -> selection)",
        "- `experiment_plan.json` — machine-readable plan (families, priorities, parent links)",
        "- `reports/status_*.md` — dated status snapshots",
        "- `hunt_state.json` / `execution_state.json` / `portfolio_state.json` — run control plane",
        "- `bundles/`, `submissions/`, `runs/`, `artifacts/` — standard slots (empty until execution starts)",
        "",
        "## Refresh",
        "",
        "```powershell",
        "python scripts/refresh_r2ai2026_workspace.py            # live refresh",
        "python scripts/refresh_r2ai2026_workspace.py --offline  # reuse cached API dumps",
        "python scripts/analysis/audit_corpus.py                 # corpus/query audit only",
        "```",
        "",
    ]
    if audit:
        corpus = audit["corpus"]
        lines += [
            "## Data snapshot",
            "",
            f"- Queries: {audit['query']['rows']:,} | corpus URLs: {corpus['rows']:,} "
            f"({corpus['pubmed_urls']:,} PubMed, {corpus['rows'] - corpus['pubmed_urls']:,} crawl targets)",
            f"- Ground-truth doc ids: integer `id` column of the corpus file (range {corpus['id_min']}..{corpus['id_max']})",
            "",
        ]
    lines += [
        "## Next steps",
        "",
        "1. Harvest PubMed by PMID (E-utilities) and crawl the VI/ZH URLs; store extracted text sharded.",
        "2. Replicate the official scorer locally (macro F2 + token-LCS chunk overlap >= 0.40) for offline tuning.",
        "3. Build the sparse baseline, then dense/hybrid/rerank branches, gated by the dev proxy.",
        "4. Emit `submission.json` for all 1,200 queries, zip it alone, upload via My Submissions.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached raw dumps instead of live APIs")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    workspace = ROOT / ".competition_hunter" / SLUG
    knowledge = workspace / "knowledge"
    if not knowledge.exists():
        raise SystemExit(f"workspace not bootstrapped: {workspace}")
    (knowledge / "pages").mkdir(parents=True, exist_ok=True)

    data_dir = (ROOT / args.data_dir).resolve()
    query_path = data_dir / HF_QUERY_FILE
    corpus_path = data_dir / HF_CORPUS_FILE

    print(f"[refresh] workspace: {workspace}")
    if args.offline:
        comp = load_json(knowledge / "codabench_competition_raw.json", {})
        leaderboard = load_json(knowledge / "codabench_leaderboard_raw.json", {})
        hf_info = load_json(knowledge / "huggingface_dataset_raw.json", {})
        hf_tree = hf_info.get("siblings", [])
        hf_readme = ""
        print("[refresh] offline mode: reusing cached API dumps")
    else:
        print("[refresh] fetching Codabench competition ...")
        comp = fetch_json(f"{BASE}/api/competitions/{COMP_ID}/")
        print("[refresh] fetching Codabench leaderboard ...")
        leaderboard = fetch_json(f"{BASE}/api/leaderboards/{LEADERBOARD_ID}/")
        print("[refresh] fetching HuggingFace dataset metadata ...")
        hf_info = fetch_json(f"https://huggingface.co/api/datasets/{HF_DATASET}")
        hf_tree = fetch_json(f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main")
        hf_readme = fetch_text(f"https://huggingface.co/datasets/{HF_DATASET}/raw/main/README.md")

    if "__fetch_error__" in comp:
        cached = load_json(knowledge / "codabench_competition_raw.json", {})
        if cached:
            print(f"[refresh] competition fetch failed ({comp['__fetch_error__']}); using cached dump")
            comp = cached
    if "__fetch_error__" in leaderboard:
        cached = load_json(knowledge / "codabench_leaderboard_raw.json", {})
        if cached:
            print(f"[refresh] leaderboard fetch failed; using cached dump")
            leaderboard = cached
    if "__fetch_error__" in hf_info:
        hf_info = load_json(knowledge / "huggingface_dataset_raw.json", {})
    if not isinstance(hf_tree, list):
        hf_tree = hf_info.get("siblings", []) or []

    entries = leaderboard_entries(leaderboard)
    previous_entries = load_json(knowledge / "leaderboard_snapshot.json", {}).get("entries", [])

    audit = None
    if query_path.exists() and corpus_path.exists():
        print("[refresh] auditing local corpus snapshot ...")
        audit = audit_retrieval_data(query_path, corpus_path)
    else:
        print(f"[refresh] local data not found in {data_dir} — skipping corpus audit")

    # ----- raw dumps -----
    write_json(knowledge / "codabench_competition_raw.json", comp)
    write_json(knowledge / "codabench_leaderboard_raw.json", leaderboard)
    write_json(knowledge / "huggingface_dataset_raw.json", hf_info)
    if hf_readme and not hf_readme.startswith("__fetch_error__"):
        (knowledge / "hf_readme.md").write_text(hf_readme, encoding="utf-8")

    # ----- rendered pages -----
    for page in comp.get("pages", []):
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", page["title"])
        (knowledge / "pages" / f"{page['index']:02d}_{safe}.md").write_text(
            f"# {page['title']}\n\nSource: {COMP_URL}#/pages-tab (page id {page['id']})\n\n"
            + html_to_text(page["content"]),
            encoding="utf-8",
        )
    if comp.get("terms"):
        (knowledge / "pages" / "99_Terms.md").write_text(
            "# Terms / Quy định chung\n\n" + html_to_text(comp["terms"]), encoding="utf-8"
        )

    # ----- leaderboard snapshot -----
    write_json(
        knowledge / "leaderboard_snapshot.json",
        {
            "captured_at": now_iso(),
            "leaderboard_id": LEADERBOARD_ID,
            "entries": entries,
            "previous_captured_at": load_json(knowledge / "leaderboard_snapshot.json", {}).get("captured_at"),
        },
    )

    # ----- audit artifacts -----
    if audit:
        write_json(knowledge / "corpus_audit.json", audit)
        (knowledge / "corpus_audit.md").write_text(
            render_audit_markdown(audit, title=f"{SLUG} corpus audit"), encoding="utf-8"
        )

    # ----- brief -----
    (knowledge / "competition_brief.md").write_text(
        render_brief(comp, entries, hf_info, hf_tree, audit), encoding="utf-8"
    )
    (workspace / "README.md").write_text(render_readme(comp, entries, audit), encoding="utf-8")

    # ----- normalized context / snapshot -----
    context = CompetitionContext(
        host=CompetitionHost.CODABENCH,
        url=COMP_URL,
        slug=SLUG,
        title=comp.get("title", SLUG),
        description=comp.get("description", ""),
        metric=METRIC,
        deadline=(
            "Platform public phase end: "
            + str((comp.get("phases") or [{}])[0].get("end", ""))
            + f" ({to_ict((comp.get('phases') or [{}])[0].get('end', ''))}); "
            "doc deadline 2026-10-31 23:59 ICT; final system 2026-11-04 23:59 ICT"
        ),
        quota_daily_submissions=int((comp.get("phases") or [{}])[0].get("max_submissions_per_day") or 10),
        rules_summary=RULES_SUMMARY,
        resource_permissions=load_json(
            workspace / "competition_context.json",
            {},
        ).get(
            "resource_permissions",
            {
                "external_data_allowed": True,
                "internet_allowed": True,
                "public_pretrained_models_allowed": True,
                "closed_llm_allowed": False,
                "model_size_limit_15b": True,
                "model_release_cutoff_2026_08_01": True,
                "public_notebooks_allowed": True,
                "discussion_access_allowed": False,
            },
        ),
    )
    write_json(workspace / "competition_context.json", context.to_dict())

    phases = comp.get("phases", [])
    file_inventory = [
        CompetitionFile(name=HF_QUERY_FILE, size_bytes=0, role=FileRole.TEST),
        CompetitionFile(name=HF_CORPUS_FILE, size_bytes=0, role=FileRole.AUXILIARY),
    ]
    for phase in phases:
        for task in phase.get("tasks", []):
            for dataset in task.get("public_datasets", []):
                file_inventory.append(
                    CompetitionFile(
                        name=f"phase{phase['index']}:{dataset.get('type', 'dataset')}:{dataset.get('name', '')}",
                        size_bytes=int(dataset.get("file_size") or 0),
                        role=FileRole.AUXILIARY,
                    )
                )
    snapshot = CompetitionSnapshot(
        competition_slug=SLUG,
        generated_at=now_iso(),
        overview_url=COMP_URL,
        page_title=comp.get("title", ""),
        page_description=html_to_text(
            next((p["content"] for p in comp.get("pages", []) if p["title"] == "Overview"), "")
        )[:4000],
        metric=METRIC,
        deadline=context.deadline,
        file_inventory=file_inventory,
        sample_submission_file="",
        test_files=[HF_QUERY_FILE],
        reference_files=[HF_CORPUS_FILE],
        column_hints={
            HF_QUERY_FILE: ["id:integer", "query:string (Vietnamese medical query)"],
            HF_CORPUS_FILE: ["id:integer", "url:string"],
            "submission.json": ["id:integer", "relevant_docs:list[int]", "relevant_chunks:list[{doc_id, chunk_text}]"],
        },
        preview_rows={
            HF_QUERY_FILE: [{"id": "1", "query": "<Vietnamese medical query>"}],
            HF_CORPUS_FILE: [{"id": "1", "url": "<source article url>"}],
            "submission.json": [
                {
                    "id": "1",
                    "relevant_docs": "[123, 456]",
                    "relevant_chunks": '[{"doc_id": 123, "chunk_text": "<verbatim span>"}]',
                }
            ],
        },
        target_candidates=["relevant_docs", "relevant_chunks"],
        task_hints=[
            "multilingual_medical_document_retrieval",
            "chunk_level_retrieval",
            "vietnamese_query_understanding",
            "cross_lingual_vi_en_zh",
            "corpus_construction_required",
            "codabench_zip_json_submission",
        ],
        errors=[],
    )
    write_json(workspace / "competition_snapshot.json", snapshot.to_dict())

    # ----- plan -----
    plan = build_retrieval_experiment_plan(context, snapshot)
    write_json(workspace / "experiment_plan.json", plan.to_dict())
    (workspace / "plans").mkdir(exist_ok=True)
    (workspace / "plans" / "retrieval_plan.md").write_text(
        render_retrieval_plan(plan, comp, entries, audit), encoding="utf-8"
    )

    # ----- states -----
    previous_execution = load_json(workspace / "execution_state.json", {})
    write_json(
        workspace / "execution_state.json",
        {
            "competition_slug": SLUG,
            "host": "codabench",
            "available": False,
            "platform": "codabench",
            "dashboard": COMP_URL,
            "submission_method": "ZIP (single .json) upload via My Submissions on the dashboard",
            "kaggle": {"available": False, "reason": "not_a_kaggle_competition"},
            "execution_time_limit_seconds": 600,
            "docker_image": comp.get("docker_image", ""),
            "phase_windows": {
                "public": {"start": phases[0].get("start"), "end": phases[0].get("end")} if phases else {},
                "private": {"start": phases[1].get("start"), "end": phases[1].get("end")} if len(phases) > 1 else {},
            },
            "quotas": {
                "public_phase": {
                    "max_per_day": (phases[0].get("max_submissions_per_day") if phases else None),
                    "max_per_person": (phases[0].get("max_submissions_per_person") if phases else None),
                },
                "private_phase": {
                    "max_per_day": (phases[1].get("max_submissions_per_day") if len(phases) > 1 else None),
                    "max_total_per_person": (phases[1].get("max_submissions_per_person") if len(phases) > 1 else None),
                },
            },
            "planned_kernels": previous_execution.get("planned_kernels", []),
            "refreshed_at": now_iso(),
        },
    )

    previous_hunt = load_json(workspace / "hunt_state.json", {})
    write_json(
        workspace / "hunt_state.json",
        {
            "competition_slug": SLUG,
            "started_at": previous_hunt.get("started_at", now_iso()),
            "refreshed_at": now_iso(),
            "current_stage": "planning",
            "detected_pack": plan.detected_pack.value,
            "detection_reasoning": plan.detection_reasoning,
            "candidate_families": plan.candidate_families,
            "target_candidates": snapshot.target_candidates,
            "reference_files": snapshot.reference_files,
            "stages": {
                "ingestion": "ready",
                "research": "ready",
                "planning": "ready",
                "execution": "blocked",
                "judging": "pending",
                "reporting": "pending",
            },
            "blockers": [
                "corpus_not_crawled_yet (1,220,598 urls; 240,000 pubmed urls harvestable via e-utilities)",
                "no_local_scorer_replica_yet (metric: macro F2 + token-LCS chunk overlap >= 0.40)",
                "no_dev_proxy_labels_yet (tuning must not burn the 10/day public quota)",
                "submission_requires_dashboard_account (codabench host, no kaggle kernels)",
            ],
        },
    )

    previous_portfolio = load_json(workspace / "portfolio_state.json", {})
    write_json(
        workspace / "portfolio_state.json",
        {
            "competition_slug": SLUG,
            "phase": "discovery",
            "active_branches": plan.candidate_families,
            "killed_branches": previous_portfolio.get("killed_branches", []),
            "promoted_branches": previous_portfolio.get("promoted_branches", []),
            "top_candidates": previous_portfolio.get("top_candidates", []),
            "submission_history": previous_portfolio.get("submission_history", []),
            "generated_at": now_iso(),
        },
    )

    # ----- reports -----
    (workspace / "reports").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (workspace / "reports" / f"status_{stamp}.md").write_text(
        render_status_report(comp, entries, previous_entries, audit), encoding="utf-8"
    )

    # ----- orchestrator knowledge store -----
    try:
        from orchestrator.core.models import KnowledgeRecord, KnowledgeType, SystemID
        from orchestrator.knowledge.store import KnowledgeStore

        store = KnowledgeStore(ROOT / ".orchestrator" / f"{SLUG}_knowledge.json")
        baseline = pick_baseline(entries)
        best = pick_best_competitor(entries)
        store.add(
            KnowledgeRecord(
                type=KnowledgeType.COMPETITION_OVERVIEW,
                content=(knowledge / "competition_brief.md").read_text(encoding="utf-8"),
                source_system=SystemID.COMPETITION_HUNTER,
                competition_slug=SLUG,
                tags=["codabench", "refresh", "brief"],
            )
        )
        store.add(
            KnowledgeRecord(
                type=KnowledgeType.METRIC_RESULT,
                content=(
                    f"Leaderboard refresh {now_iso()}: {len(entries)} entries. "
                    f"Admin baseline {baseline.get('display_name')} FINAL "
                    f"{baseline.get('scores', {}).get('FINAL_SCORE', 0):.4f}; best non-admin "
                    f"{best.get('scores', {}).get('FINAL_SCORE', 0):.4f}. "
                    f"Participants {comp.get('participants_count')}, submissions {comp.get('submissions_count')}."
                ),
                source_system=SystemID.COMPETITION_HUNTER,
                competition_slug=SLUG,
                tags=["leaderboard", "refresh"],
                metric=float(best.get("scores", {}).get("FINAL_SCORE", 0) or 0) or None,
            )
        )
        if audit:
            corpus = audit["corpus"]
            store.add(
                KnowledgeRecord(
                    type=KnowledgeType.EDA_REPORT,
                    content=(
                        f"Corpus audit: {corpus['rows']:,} urls ({corpus['pubmed_urls']:,} pubmed), "
                        f"id range {corpus['id_min']}..{corpus['id_max']}, languages {corpus['languages_heuristic']}. "
                        f"Queries: {audit['query']['rows']:,}, median {audit['query']['words_median']} words. "
                        "HF corpus file renamed to links_corpus_vi_en_zh.parquet (dataset re-created 2026-10-12)."
                    ),
                    source_system=SystemID.COMPETITION_HUNTER,
                    competition_slug=SLUG,
                    tags=["corpus", "audit", "hf"],
                )
            )
        store.add(
            KnowledgeRecord(
                type=KnowledgeType.TECHNIQUE,
                content=(
                    "Retrieval plan branches: "
                    + ", ".join(plan.candidate_families)
                    + ". Ordered corpus -> sparse -> dense -> hybrid -> rerank -> chunking -> selection, "
                    "with a local scorer replica and dev proxy gating every leaderboard probe."
                ),
                source_system=SystemID.COMPETITION_HUNTER,
                competition_slug=SLUG,
                tags=["plan", "retrieval"],
            )
        )
        print(f"[refresh] knowledge store updated: {store.path}")
    except Exception as exc:  # pragma: no cover - store is optional
        print(f"[refresh] knowledge store update skipped: {type(exc).__name__}: {exc}")

    print("[refresh] done")
    print(f"  entries: {len(entries)} | participants: {comp.get('participants_count')} | submissions: {comp.get('submissions_count')}")
    if entries:
        top = entries[0]
        print(f"  top entry: {top['display_name'] or 'anonymous'} FINAL {top['final_score']:.4f}")
    print(f"  brief:   {knowledge / 'competition_brief.md'}")
    print(f"  plan:    {workspace / 'plans' / 'retrieval_plan.md'}")
    print(f"  report:  {workspace / 'reports' / f'status_{stamp}.md'}")


if __name__ == "__main__":
    main()
