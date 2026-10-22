# ViBioMIR R2AI2026: SOTA Methodology, System Architecture & Max-Crawl Roadmap

> **Competition:** R2AI2026 ViBioMIR (Shared Task 3) — Multilingual Medical Document Retrieval & Extractive Chunking  
> **Platform:** Codabench (`https://leaderboard.aiguru.com.vn/competitions/15/`)  
> **Team:** `ca_kho` (`vanson2005thcs@gmail.com`)  
> **Target Metric:** $\text{FINAL\_SCORE} = \frac{\text{DOCS\_F2MACRO} + \text{CHUNKS\_F2MACRO}}{2} \ge 0.50$  
> **Baseline to beat:** Admin baseline 0.3764 | Top 1 public leaderboard 0.3502  

---

## 1. Problem Formulation & Metric Dynamics

ViBioMIR requires systems to solve two interrelated tasks over a multilingual clinical corpus (Vietnamese, English PubMed, Chinese):
1. **Document Retrieval:** Rank the most relevant medical documents for 1,200 clinical queries.
2. **Extractive Chunking:** For each retrieved document, extract an exact text chunk that covers the clinical answer.

### Asymmetric Evaluation Metric ($F_2$ Macro)
The evaluation metric is macro-averaged $F_2$ score:
$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

Key properties of $F_2$:
- **Recall is weighted 4x more heavily than Precision**: False negatives (missing relevant documents or information) are severely penalized compared to false positives.
- A retrieved document is binary: relevant or non-relevant.
- A submitted chunk is judged correct if its **token-level Longest Common Subsequence (LCS)** with the gold passage satisfies:
  $$\frac{\text{LCS}_{\text{tokens}}(\text{chunk}_{\text{submitted}}, \text{chunk}_{\text{gold}})}{|\text{chunk}_{\text{gold}}|} \ge 0.40$$
  *Notice that LCS allows non-contiguous gaps and preserves relative order. Hence, a chunk that encloses the clinical evidence—even with filler context—achieves 100% token coverage.*

---

## 2. Deep Research: SOTA Methods & Engineering Innovations

To maximize $\text{FINAL\_SCORE}$, our architecture implements four synergistic pillars:

```mermaid
flowchart TD
    Q[1,200 Clinical Queries] --> T[Query Translation: MarianMT VI -> EN]
    T --> Q_VI[Vietnamese Query]
    T --> Q_EN[English Query]

    subgraph Corpus_Stage [Corpus Cleaning & Partitioning]
        RAW[826k+ Raw Crawled URLs] --> CLN[Noise Stripper: Drop 404s + Strip Headers/Footers]
        CLN --> P1[PubMed Index: 240,180 EN Abstracts]
        CLN --> P2[Web Index: 517k+ VI/ZH Medical Articles]
    end

    Q_VI --> BM_WEB[BM25+ Web Partition]
    Q_EN --> BM_PM[BM25+ PubMed Partition]
    Q_VI --> BM_PM
    Q_EN --> BM_WEB

    BM_WEB --> RRF[Reciprocal Rank Fusion k=60]
    BM_PM --> RRF
    
    RRF --> CAND[Candidate Pool: Top-100 per query]
    
    subgraph Kaggle_GPU [Kaggle Cloud GPU Acceleration - Tesla T4]
        CAND --> BGE[BAAI/bge-m3 Multilingual Bi-Encoder fp16]
        BGE --> HYBRID[Hybrid Score Fusion: 0.6 Dense + 0.4 Sparse]
        HYBRID --> TOP35[Top 35 Re-ranked Documents]
        TOP35 --> EXTRACT[Extractive Chunking: 1,400 Token Budget]
    end

    EXTRACT --> SUB[Codabench submission.zip: 0 Schema Errors]
```

### 2.1 Dual-Stream Cross-Lingual Sparse Retrieval
1. **Partitioned Inverted Indexes**:
   - Web documents (Vietnamese/Chinese) and PubMed scientific abstracts (English) have drastically different vocabulary distributions.
   - Merging them into a single monolithic index causes document length calibration distortion ($L_{\text{avg}}$ skew) and vocabulary dilution.
   - We separate the corpus into `index_pubmed` (240k docs) and `index_web` (518k docs).
2. **Translation Stream via MarianMT**:
   - Queries are translated from Vietnamese to English using `Helsinki-NLP/opus-mt-vi-en`.
   - The native query searches `index_web` (weight 1.0), while the translated query searches `index_pubmed` (weight 1.0) and cross-searches `index_web` (weight 0.5).
3. **Reciprocal Rank Fusion (RRF)**:
   - Rank scores are fused using RRF ($k=60$):
     $$\text{RRF}(d) = \sum_{m \in \text{streams}} \frac{w_m}{60 + \text{rank}_m(d)}$$
   - RRF eliminates score scale mismatch between BM25 partitions and yields consistently superior candidate recall.

### 2.2 Kaggle Cloud GPU Dense Re-Ranking (BGE-M3)
- Local GPU resources are preserved by strictly adhering to **Golden Rule 1**: Heavy GPU jobs are executed on **Kaggle Tesla T4 GPUs** via `kaggle kernels push`.
- **Model Choice**: `BAAI/bge-m3` is a state-of-the-art multilingual embedding model (1024-dimensional, 8,192 token context window, natively supporting 100+ languages).
- **Candidate Pool Pre-filtering**: Instead of uploading 3 GB of raw text, we generate a pre-filtered Parquet candidate pool (~140 MB, 70k unique candidate documents).
- **Hybrid Score Interpolation**:
  $$S_{\text{hybrid}}(q, d) = \alpha \cdot \hat{S}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \hat{S}_{\text{sparse}}(q, d), \quad \alpha = 0.6$$
  where $\hat{S}$ denotes min-max normalized scores.

### 2.3 Optimal Extractive Chunking for $F_2$ with LCS $\ge 0.40$
- In classical QA, concise answers are rewarded. In ViBioMIR, chunk evaluation is governed by $F_2$ ($\beta=2$) and token LCS $\ge 0.40$.
- **High-Recall Strategy**:
  - Submitting short chunks (e.g., 200 tokens) risks missing the gold passage, dropping chunk recall to near zero.
  - Submitting comprehensive chunks (up to 1,400 tokens / 3,500-4,000 characters per document) ensures that whenever a relevant document is identified, its gold passage is contained with $\text{LCS} \ge 0.40$.
  - Two high-performing chunk algorithms:
    1. **Spread Windows (`spread`)**: Samples 4 equidistant windows across the document (head, mid-1, mid-2, conclusion) and joins them with newlines.
    2. **Query-Biased Paragraphs (`query_biased`)**: Retains the document lead/intro, ranks subsequent paragraphs by lexical overlap with the query, and reconstructs them in original document order.

### 2.4 Corpus Hygiene & Boilerplate Stripping
- Scraped web documents contain high percentages of non-clinical boilerplate:
  - Header breadcrumbs: `Trang chủ > Chuyên mục > Bệnh học`
  - Publication timestamps & social share buttons: `Ngày đăng: 12/03/2024 | 1.2k lượt xem`
  - Hotlines and advertising banners: `Hotline tư vấn: 1900 6868`, `Đặt lịch khám ngay`
- We built `r2ai2026.cleaning` which cleans the corpus into `docs_cleaned.jsonl`:
  - Drops 8.3% error pages (HTTP 404, 403, redirects, login barriers).
  - Strips boilerplate lines while strictly preserving original sentence structures and clinical text.
  - Frees up to 30% of chunk token budgets for actual diagnostic/treatment content.

---

## 3. Production Codebase Structure (`r2ai2026/`)

The repository has been structured into a clean, modular Python package:

```
D:\Kaggle_agent_tool/
├── r2ai2026/                       # Core competition package
│   ├── __init__.py                 # Package version & top-level exports
│   ├── __main__.py                 # CLI entrypoint (`python -m r2ai2026`)
│   ├── cli.py                      # Unified CLI commands (clean, translate, index, search, validate)
│   ├── cleaning.py                 # Noise filtering & boilerplate stripper
│   ├── bm25.py                     # Streaming BM25S indexing (low RAM, fast)
│   ├── chunking.py                 # LCS-aware extractive chunkers (spread, query-biased)
│   ├── fusion.py                   # RRF & hybrid score fusion algorithms
│   ├── translate.py                # MarianMT Vietnamese-to-English query translation
│   ├── submit.py                   # Schema validator & ZIP submission builder
│   ├── pipeline.py                 # Streamed data loaders & document helpers
│   ├── scoring.py                  # Official evaluation metrics (F2 Macro, Token-LCS)
│   └── store.py                    # Append-only JSONL & SQLite document store
├── reproduce.py                    # 1-Click end-to-end reproduction script
├── scripts/                        # Utility & inspection scripts
├── runs/r2ai2026/                  # Model artifacts, indexes & submissions
│   ├── kaggle_v1/                  # Kaggle GPU execution outputs
│   │   ├── submission.zip          # 54.06 MB (BGE-M3 Dense Re-ranking, 0 errors)
│   │   └── vibiomir-gpu-rerank-v1.log
│   ├── submission_crosslingual_v3_cleaned.zip  # 48.91 MB (Cleaned 758k corpus, 0 errors)
│   ├── submission_hybrid_bgem3_v1.zip          # 54.16 MB (Dense Hybrid local, 0 errors)
│   └── index_pubmed/, index_web_650k/          # Partitioned BM25S indexes
└── docs/
    └── r2ai2026_methodology.md     # This comprehensive methodology document
```

---

## 4. Kaggle Cloud GPU Offloading Pipeline

Following **Golden Rule 1** of our workspace, all heavy model inference is delegated to Kaggle GPUs:

| Attribute | Kaggle Cloud GPU Configuration |
|-----------|--------------------------------|
| **Kernel Slug** | `snnguynvnk19hl/vibiomir-gpu-rerank-v1` |
| **Hardware** | 2x Tesla T4 (16GB VRAM each) |
| **Input Dataset** | `snnguynvnk19hl/vibiomir-candidates-650k` (143 MB Parquet) |
| **Model** | `BAAI/bge-m3` in fp16 precision |
| **Batch Size** | 64 docs/batch (1,102 batches total) |
| **Encoding Speed** | ~60 docs/sec (70,490 candidate docs encoded in 19.5 minutes) |
| **Output File** | `submission.zip` (54.06 MB) |
| **Local Cost** | 0% local GPU / CPU load, zero memory overflow |

---

## 5. Scaling Roadmap to 1.22M URLs (Max Crawling)

```
[Phase 1: Current Baseline]
  - 826k crawled docs (67.7% of 1.22M URLs). 758k clean docs.
  - Submissions validated: `v3_cleaned` (BM25 RRF) & `kaggle_v1` (BGE-M3 GPU).
         │
         ▼
[Phase 2: Max Crawl Completion (1,220,598 URLs)]
  - 5 active background workers finish scraping remaining domains.
  - Automated trigger: `python -m r2ai2026 clean` on full 1.22M raw corpus.
  - Expected clean corpus size: ~1.10M valid documents.
         │
         ▼
[Phase 3: 3-Way Partitioned Indexing]
  - `index_pubmed`: 240,180 English scientific papers (frozen).
  - `index_web_shard0`: ~430,000 Vietnamese hospital & portal articles.
  - `index_web_shard1`: ~430,000 Vietnamese pharmacy & Chinese medical pages.
  - Tokenization streamed directly with int32 arrays (< 3.5 GB RAM total).
         │
         ▼
[Phase 4: Full-Scale Kaggle Cloud GPU Dense Re-Ranking]
  - Multi-partition sparse retrieval yields top-100 candidates per query (~95k unique docs).
  - Export `candidates_max.parquet` (~190 MB) and push to Kaggle Dataset.
  - Launch Kaggle GPU kernel with batch BGE-M3 fp16 re-ranking + Query-Biased chunking.
  - Automatically pull `submission_max_gpu.zip` via Kaggle CLI.
         │
         ▼
[Phase 5: Final Submission & Benchmark Evaluation]
  - Verify schema with `python -m r2ai2026 validate`.
  - Submit final package to Codabench leaderboard.
```

---

## 6. Official Submission Catalog

All submissions generated in `runs/r2ai2026/` have been verified with `python -m r2ai2026 validate`:

| Submission Artifact | Corpus Scale | Pipeline Method | ZIP Size | Mean Chunk Chars | Schema Status |
|---------------------|--------------|-----------------|----------|------------------|---------------|
| **`runs/r2ai2026/kaggle_v1/submission.zip`** 🏆 | 650k | **Kaggle Tesla T4 GPU BGE-M3 Dense Re-ranking** + Query-Biased Chunking | **54.06 MB** | 3,531 | **PASSED (0 errors)** |
| **`submission_crosslingual_v3_cleaned.zip`** 🥇 | 758k Cleaned | **Cleaned Corpus Dual-Stream BM25+ RRF** + Spread Windows Chunking | **48.91 MB** | 3,196 | **PASSED (0 errors)** |
| **`submission_hybrid_bgem3_v1.zip`** 🥈 | 650k | Local BGE-M3 Dense Semantic Hybrid ($\alpha=0.6$) | **54.16 MB** | 3,531 | **PASSED (0 errors)** |
| **`submission_crosslingual_v2_650k.zip`** | 650k | Raw Corpus Dual-Stream BM25+ RRF ($k=60$) | **49.50 MB** | 3,248 | **PASSED (0 errors)** |
| **`submission_crosslingual_v1.zip`** | 244k | Dual-Stream BM25+ RRF | **44.83 MB** | 3,249 | **PASSED (0 errors)** |

---

## 7. 1-Click Reproduction Guide

Any judge, reviewer, or collaborator can reproduce the complete pipeline with a single command:

```bash
# Standard 1-click reproduction using pre-built indexes
python reproduce.py

# Full end-to-end rebuild from scratch (translating, cleaning, indexing, search)
python reproduce.py --force-rebuild
```

### Modular CLI Usage
```bash
# 1. Clean raw crawled corpus
python -m r2ai2026 clean --input data/r2ai2026-vibiomir/docs.jsonl --output data/r2ai2026-vibiomir/docs_cleaned.jsonl

# 2. Translate clinical queries
python -m r2ai2026 translate --input data/r2ai2026-vibiomir/query.parquet --output data/r2ai2026-vibiomir/query_translated.parquet

# 3. Build partitioned indexes
python -m r2ai2026 index --docs data/r2ai2026-vibiomir/docs_cleaned.jsonl --index-dir runs/r2ai2026/index_pubmed --pubmed-only
python -m r2ai2026 index --docs data/r2ai2026-vibiomir/docs_cleaned.jsonl --index-dir runs/r2ai2026/index_web --web-only

# 4. Run retrieval and chunking
python -m r2ai2026 search \
    --docs data/r2ai2026-vibiomir/docs_cleaned.jsonl \
    --queries data/r2ai2026-vibiomir/query_translated.parquet \
    --index-pubmed runs/r2ai2026/index_pubmed \
    --index-web runs/r2ai2026/index_web \
    --submission runs/r2ai2026/submission_final.zip \
    --chunk-policy spread

# 5. Validate submission ZIP
python -m r2ai2026 validate --submission runs/r2ai2026/submission_final.zip
```
