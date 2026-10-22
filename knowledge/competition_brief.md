# R2AI2026 TRUY HỒI TÀI LIỆU Y KHOA ĐA NGÔN NGỮ — Competition Brief (refreshed)

- Slug: `r2ai2026-vibiomir` | Host: Codabench | Competition id: 15 | Leaderboard id: 93
- URL: https://leaderboard.aiguru.com.vn/competitions/15/ (tab: #/pages-tab)
- Description: R2AI2026 TRUY HỒI TÀI LIỆU Y KHOA ĐA NGÔN NGỮ - Shared Task 3
- Participants: **88** | Submissions: **1997** | Created: 2026-09-30T15:34:45.636301Z
- Docker image: `codalab/codalab-legacy:py37` | Execution time limit: 600s | Robot submissions: False
- Refreshed at: 2026-10-18T14:36:11+00:00

## Timeline (all deadlines 23:59 ICT unless noted)

| Milestone | Window |
|---|---|
| Public phase (platform api) | 2026-09-30T00:00:00Z -> 2026-10-31T00:00:00Z |
| Public phase (ICT) | 2026-09-30 07:00 ICT -> **2026-10-31 07:00 ICT** |
| Private phase (platform api) | 2026-11-01T00:00:00Z -> 2026-11-04T00:00:00Z |
| Private phase (ICT) | 2026-11-01 07:00 ICT -> **2026-11-04 07:00 ICT** |
| Public submission deadline (doc) | 2026-10-31 23:59 ICT |
| Final system deadline (doc) | 2026-11-04 23:59 ICT |
| Results announced | 2026-11-11 |

> Platform phase windows close earlier than the documented deadlines (public phase ends at the platform `end` above, i.e. same calendar day but not 23:59). Treat the platform window as the binding constraint.

## Data (HuggingFace: https://huggingface.co/datasets/AIGuruTinix/ViBioMIR)

- HF repo created/last modified: 2026-10-12T13:52:14.000Z / 2026-10-12T13:52:14.000Z | downloads: 3185 | gated: False
- Files: 
- `query.parquet`: `id:int`, `query:str` — Vietnamese medical queries (both phases).
- `links_corpus_vi_en_zh.parquet`: `id:int`, `url:str` — source article URLs; teams crawl/clean/chunk/index themselves.
- Ground-truth doc labels use the integer `id` in the corpus file; documents outside that list are never positive.

### Corpus snapshot (local audit)

- Queries: **1,200** (median 18 words, p95 84, max 281)
- Corpus rows: **1,220,598** | unique urls: 1,220,598 | id range 583..7466161
- Language heuristic: `{'vi': 694909, 'unknown': 55592, 'en': 240000, 'zh': 230097}`
- PubMed URLs (bulk-harvestable via E-utilities): **240,000**
- Top hosts: pubmed.ncbi.nlm.nih.gov (240,000), suckhoecongdongonline.vn (154,503), www.120ask.com (98,160), suckhoedoisong.vn (85,823), nhathuoclongchau.com.vn (79,598), thanhnien.vn (72,111), www.cnkang.com (40,046), ask.39.net (32,772)

## Metric

- Document level: Precision = |D∩G|/|D|, Recall = |D∩G|/|G|.
- Chunk level: a submitted chunk counts as correct when LCS-token overlap with any reference chunk of the same query+doc is >= 0.40 (text normalized: NFKC, HTML entities, lowercase, whitespace).
- F2 = 5·P·R / (4·P + R), macro-averaged per query at both levels.
- **FINAL_SCORE = mean(DOCS_F2MACRO, CHUNKS_F2MACRO); per-query Precision/Recall/F2 macro-averaged over all queries, scored separately at document level and chunk level**

## Submission format

Single `.json` array zipped alone (no folder), uploaded via My Submissions:

```json
[{"id": 1, "relevant_docs": [123], "relevant_chunks": [{"doc_id": 123, "chunk_text": "<verbatim>"}]}]
```

- Empty lists are mandatory when a level has no prediction; a submission missing queries is not evaluated (and burns no quota).
- `chunk_text` must be verbatim text extracted from the source document.

## Leaderboard (refreshed snapshot)

- Entries in snapshot: **50** (this is the public leaderboard feed, newest first by score).

| Rank | Entry | FINAL | DOCS F2 | CHUNKS F2 | DOCS P/R | CHUNKS P/R | When |
|---:|---|---:|---:|---:|---|---|---|
| 1 | BM25 Baseline | 0.3764 | 0.5587 | 0.1941 | 0.8805/0.5146 | 0.3403/0.1755 | 2026-09-30 |
| 2 | Bien Tran Ngoc | 0.3502 | 0.3617 | 0.3388 | 0.2398/0.4669 | 0.2822/0.4082 | 2026-10-12 |
| 3 | anonymous | 0.3187 | 0.3250 | 0.3125 | 0.2746/0.3968 | 0.2677/0.3749 | 2026-10-16 |
| 4 | anonymous | 0.3076 | 0.3207 | 0.2945 | 0.1698/0.5219 | 0.2958/0.3481 | 2026-10-14 |
| 5 | anonymous | 0.2683 | 0.2730 | 0.2636 | 0.1516/0.4071 | 0.1864/0.3424 | 2026-10-16 |
| 6 | anonymous | 0.2672 | 0.2903 | 0.2441 | 0.1558/0.4663 | 0.1557/0.3482 | 2026-10-06 |
| 7 | anonymous | 0.2427 | 0.2820 | 0.2033 | 0.1800/0.3821 | 0.1986/0.2342 | 2026-10-18 |
| 8 | anonymous | 0.2394 | 0.2655 | 0.2132 | 0.1451/0.4078 | 0.1867/0.2559 | 2026-10-12 |
| 9 | anonymous | 0.2326 | 0.2948 | 0.1703 | 0.1998/0.3824 | 0.1410/0.2047 | 2026-10-06 |
| 10 | anonymous | 0.2112 | 0.2438 | 0.1785 | 0.1949/0.3150 | 0.1634/0.2199 | 2026-10-17 |
| 11 | anonymous | 0.2049 | 0.2240 | 0.1857 | 0.1411/0.3245 | 0.1744/0.2245 | 2026-10-13 |
| 12 | anonymous | 0.1985 | 0.2257 | 0.1712 | 0.1496/0.3046 | 0.1805/0.1969 | 2026-10-12 |
| 13 | anonymous | 0.1943 | 0.2049 | 0.1837 | 0.1148/0.3324 | 0.1491/0.2439 | 2026-10-09 |
| 14 | anonymous | 0.1727 | 0.1817 | 0.1637 | 0.1271/0.2433 | 0.2040/0.1746 | 2026-10-18 |
| 15 | anonymous | 0.1626 | 0.1890 | 0.1363 | 0.1136/0.2888 | 0.1054/0.1841 | 2026-10-18 |

- Admin baseline `BM25 Baseline`: FINAL 0.3764 (docs 0.5587 / chunks 0.1941).
- Best non-admin entry: FINAL 0.3502 (docs 0.3617 / chunks 0.3388).

## Quotas

- Public phase: 10/day per team (cap 310 per person).
- Private phase: 5/day — only **5 total** submissions per user.

## Rules

One account per user; one team max per participant; one submission account per team. External data allowed but must be cited with full provenance. Open-weight PLMs/LLMs only (model and/or training data public, e.g. Hugging Face); closed LLMs (GPT-4o, Gemini, ...) forbidden; models must be released before 2026-08-01 (ICT) and be <= 15B parameters. Max 10 submissions/day in public phase; max 5 submissions total in private phase. Final ranking requires a working-notes paper describing the method. Organizers may disqualify non-compliant teams; winners grant usage rights over submissions, code and generated data.
