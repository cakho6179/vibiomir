# Retrieval plan — r2ai2026-vibiomir

- Generated: 2026-10-18T14:36:11+00:00
- Public phase closes: **2026-10-31 07:00 ICT** (2026-10-31T00:00:00Z)
- Private phase closes: **2026-11-04 07:00 ICT** (2026-11-04T00:00:00Z)
- Bar to clear: `BM25 Baseline` at FINAL 0.3764 (docs 0.5587, chunks 0.1941)

## Hypotheses

- Corpus coverage is the recall ceiling: unreachable documents can never be retrieved.
- A tuned sparse baseline is the bar to clear; every later branch is measured against it on a local proxy.
- Chunk-level score is the bottleneck on dual-level metrics (doc P is easy, chunk overlap is not).
- Selection policy tuned for the official F-beta family is worth more than another model swap.

## Branches

| # | Branch | Cost | Depends on | Objective |
|---:|---|---|---|---|
| 1 | `corpus-crawl-pipeline` | high | - | Build the retrieval base: crawl every provided URL, extract main text, dedup, shard to disk. |
| 2 | `pubmed-bulk-harvest` | medium | - | Harvest the PubMed slice of the corpus through E-utilities instead of crawling HTML. |
| 3 | `sparse-bm25` | medium | - | Reproduce and then beat the organizer BM25 baseline; it is the strongest known public entry. |
| 4 | `dense-multilingual` | high | - | Cross-lingual dense retrieval so Vietnamese queries can hit English/Chinese evidence. |
| 5 | `hybrid-rrf` | medium | sparse-bm25 | Fuse sparse and dense rankings; hybrid retrieval is the standard recall jump for multilingual IR. |
| 6 | `chunk-retrieval-pipeline` | medium | - | Answer at chunk level: segment documents and emit verbatim spans that match reference chunks. |
| 7 | `cross-encoder-rerank` | medium | hybrid-rrf | Rerank the top candidates with a multilingual cross-encoder to lift precision at the submitted depth. |
| 8 | `query-medical-vi` | high | - | Normalize and expand Vietnamese medical queries before retrieval. |
| 9 | `f2-selection-tuning` | medium | - | Choose submitted depth per query so macro F2 is maximized, not MAP. |
| 10 | `dev-proxy-eval` | medium | - | Create a labelled local proxy so tuning does not burn submission quota. |

## Branch detail

### 1. `corpus-crawl-pipeline` — Build the retrieval base: crawl every provided URL, extract main text, dedup, shard to disk.
- **Preprocessing**
  - url inventory audit — Split the URL list by host/TLD so crawl rate limits are planned per domain.
  - language routing — Route vi/en/zh hosts to the matching extraction config (charset, boilerplate rules).
  - robots/ToS + rate policy — Crawl politely: per-host concurrency caps, backoff, cached HTML.
- **Work**
  - extractor comparison — Compare readability/trafilatura/resiliparse on a stratified sample before the full run.
  - idempotent sharded store — One shard per host bucket; re-crawls must merge, not duplicate.
- **Serving**
  - dead-url fallback — Wayback/cache fallback for hosts that block or 404; log coverage per host.
  - coverage report — Report crawlable ratio per language so downstream recall limits are explicit.

### 2. `pubmed-bulk-harvest` — Harvest the PubMed slice of the corpus through E-utilities instead of crawling HTML.
- **Preprocessing**
  - pmid extraction — Parse PMIDs straight from the URL list; ids map 1:1 to E-utilities calls.
- **Work**
  - efetch batching — Batch efetch/esummary with API keys to respect 3-10 req/s limits.
  - abstract vs full text — Decide title+abstract vs PMC full text per availability; store the raw XML too.
- **Serving**
  - xml -> text parity — Keep the XML->text conversion deterministic so chunk text stays verbatim-reproducible.

### 3. `sparse-bm25` — Reproduce and then beat the organizer BM25 baseline; it is the strongest known public entry.
- **Preprocessing**
  - analyzer parity — Match tokenization/normalization of the reference scorer before tuning anything.
  - field weighting — URL slug, title, headings and body carry different signal; index them separately.
- **Work**
  - k1/b sweep — Cheap sweep on a labelled dev proxy; BM25 parameters dominate early score moves.
  - query-side term weighting — Boost medical term matches (ICD/SNOMED/DrugBank vocab overlap).
- **Serving**
  - top-k policy — F2 rewards recall; pick k per query with a score-threshold rule instead of a global k.

### 4. `dense-multilingual` — Cross-lingual dense retrieval so Vietnamese queries can hit English/Chinese evidence.
- **Preprocessing**
  - embedder shortlist audit — Confirm release date < cutoff and params <= limit for every candidate (e.g. BGE-M3 class).
  - chunk embedding cache — Embed once per chunk; keep float16 shards keyed by corpus id.
- **Work**
  - no-training first — Frozen embeddings + FAISS/HNSW usually clears the BM25 bar before any finetune.
  - domain finetune — Optional: contrastive finetune on query/doc pairs mined from the sparse branch.
- **Serving**
  - index sharding — Shard the ANN index per language so a single host can rebuild it incrementally.
  - score calibration — Normalize per-shard cosine scores before any fusion step.

### 5. `hybrid-rrf` — Fuse sparse and dense rankings; hybrid retrieval is the standard recall jump for multilingual IR.
- **Preprocessing**
  - rank-only fusion — RRF needs ranks, not scores: keeps calibration problems out of the blend.
- **Work**
  - weight sweep — Tune sparse/dense weights and fusion depth on the dev proxy, not on the leaderboard.
- **Serving**
  - per-language depth — vi/en/zh sources need different candidate depth before fusion.

### 6. `chunk-retrieval-pipeline` — Answer at chunk level: segment documents and emit verbatim spans that match reference chunks.
- **Preprocessing**
  - chunk size grid — Reference chunks are scored by token-LCS/|gold| >= threshold; chunk size directly caps achievable overlap.
  - verbatim guarantee — Never rewrite or normalize chunk text: scoring is done on extracted substrings.
  - boundary hygiene — Cut on paragraph/sentence boundaries to keep overlap with gold spans high.
- **Work**
  - overlap budget — Small overlap between neighbouring chunks raises recall without flooding the submission.
  - chunk scorer — Score chunks (query-chunk similarity) and pick per-document best chunks rather than whole docs.
- **Serving**
  - per-query chunk budget — F2 punishes precision; cap chunks per query by score gap, not a fixed number.

### 7. `cross-encoder-rerank` — Rerank the top candidates with a multilingual cross-encoder to lift precision at the submitted depth.
- **Preprocessing**
  - pair budget — Rerank only top-N per query: cross-encoders are the latency bottleneck.
- **Work**
  - reranker shortlist audit — Prefer rerankers with Vietnamese coverage; verify release-date and param constraints.
- **Serving**
  - truncation policy — Long chunks must be truncated consistently between dev proxy and submission.

### 8. `query-medical-vi` — Normalize and expand Vietnamese medical queries before retrieval.
- **Preprocessing**
  - term normalization — Map lay Vietnamese medical phrasing to canonical terms/abbreviations.
- **Work**
  - expansion variants — Query expansion / pseudo-document generation usually raises recall for recall-weighted metrics.
  - cross-lingual reformulation — Add vi->en/zh query variants to reach non-Vietnamese documents.
- **Serving**
  - multi-query fusion — Fuse expansions with RRF instead of concatenating raw text.

### 9. `f2-selection-tuning` — Choose submitted depth per query so macro F2 is maximized, not MAP.
- **Preprocessing**
  - scorer replica — Reimplement the official scorer (normalization + token LCS threshold) locally first.
- **Work**
  - k sweep on proxy — Macro-F2 vs submitted depth curve; recall weighting dominates the optimum.
  - abstain rule — Empty predictions cost nothing when no candidate clears the score floor.
- **Serving**
  - quota discipline — Public quota is small: every leaderboard probe must follow a proxy-measured decision.

### 10. `dev-proxy-eval` — Create a labelled local proxy so tuning does not burn submission quota.
- **Preprocessing**
  - pseudo-label mining — Generate query->doc labels from the strongest system plus lexical judges, then hand-audit a sample.
- **Work**
  - held-out proxy split — Freeze a proxy set; all sweeps report macro F2 on it before a leaderboard push.
- **Serving**
  - proxy-lb drift check — Track proxy vs leaderboard correlation; stale proxies quietly overfit.

## Corpus reality check

- 1,220,598 URLs, 240,000 of them PubMed (bulk harvest, no HTML crawl).
- Non-PubMed share: 980,598 URLs across 30+ hosts.
- Crawl budget must be planned per host: a handful of Vietnamese portals dominate the URL list, so per-host rate limits (not global concurrency) decide the wall-clock time.

## Critical path

1. **Corpus** — PubMed harvest + VI/ZH crawl; nothing downstream can beat a missing document.
2. **Scorer replica** — reproduce normalization + token-LCS locally so tuning does not spend quota.
3. **Sparse baseline** — match/beat the admin BM25 configuration on the local proxy.
4. **Dense + hybrid + rerank** — recall then precision, each measured on the proxy.
5. **Selection** — per-query depth tuned for F2, then submit.
