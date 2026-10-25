"""System stage scripts, run by the runner as files with path arguments (see configs/*.yaml).

Each file is standalone so it can also be wrapped as a Kaggle kernel (main scripts are ASCII-only; the wrapper
escapes helper modules); helpers are imported as flat sibling modules (docorg, query_frame), never relatively.
  corpus   docorg.py (cleaning/sections), near_dups.py, hf_recover.py
  index    bm25_experiment.py (doc BM25), build_section_index.py (section BM25), dense_embed.py, embed_sections.py
  query    query_frame.py (LLM frame), ner_tag.py + entity_index.py (entities)
  rank     rerank.py (cross-encoder), build_pairs.py + judge.py (Qwen3-Reranker decision model)
  output   build_submission.py (fusion, cut, chunks, LZMA zip)
"""
