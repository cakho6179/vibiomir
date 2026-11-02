"""Per-language candidate pools as probe sets: VI from pool5000, ZH / EN by RRF over every ranking of that language.

    python scripts/kaggle_ops/lang_pool_sets.py --vi 3000 --zh 1000 --en 1000 --additions kaggle/datasets/probe_inputs/additions.parquet

The channels can be re-pointed at a run's own rankings with --vi-pool / --zh-list / --en-list (each list written
``[VARIANT=]PATH``, the variant first so a Windows drive letter is not read as a separator); without them the
defaults below rebuild the production sets.

Sets written (qid, set, rank, doc_id), replacing sets of the same name in --additions (created when absent):
  vi<N>  pool5000_vi.parquet rank < N (dense VI + BM25 doc + BM25 section, RRF 2:1:1)
  zh<N>  RRF k=60, equal weights, over: dense ZH (VI query), dense ZH (zh_q, zh_kw), BM25 ZH (zh_q, zh_kw)
  en<N>  RRF k=60, equal weights, over: dense EN (VI query, en_kw), BM25 EN (VI query, en_kw)
Each (file, variant) pair is one ranked list; dense shards restart at rank 0, so ranks are recomputed by score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data" / "rankings_v21"
Q = ROOT / "kaggle" / "outputs"
DEFAULT_LISTS = {
    "zh": [(R / "m3_zh.parquet", None), (Q / "qv3-m3-zh/ranking.parquet", "zh_q"),
           (Q / "qv3-m3-zh/ranking.parquet", "zh_kw"), (Q / "qv3-bm25-zh/ranking.parquet", "zh_q"),
           (Q / "qv3-bm25-zh/ranking.parquet", "zh_kw")],
    "en": [(R / "m3_en.parquet", None), (Q / "qv3-m3-en/ranking.parquet", "en_kw"),
           (R / "bm25_en_viquery.parquet", None), (Q / "qv3-bm25-en/ranking.parquet", "en_kw")],
}
VI_MAX, ZH_MAX = 766441, 4392214


def in_lang(doc: pd.Series, lang: str) -> pd.Series:
    return (doc > VI_MAX) & (doc <= ZH_MAX) if lang == "zh" else doc > ZH_MAX


def parse_list_spec(spec: str) -> tuple[Path, str | None]:
    """One ranked list as ``[VARIANT=]PATH``; without a variant the whole file is one list.

    The variant comes first so a Windows drive letter in PATH cannot be read as a separator.
    """
    variant, sep, path = spec.partition("=")
    return (Path(path), variant) if sep else (Path(spec), None)


def fused(lang: str, n: int, lists: list[tuple[Path, str | None]]) -> pd.DataFrame:
    parts = []
    for path, variant in lists:
        r = pd.read_parquet(path)
        if variant is not None:
            r = r[r["variant"] == variant]
        r = r[in_lang(r["doc_id"], lang)].sort_values(["qid", "score"], ascending=[True, False])
        r = r.assign(rank=r.groupby("qid").cumcount())
        parts.append(r[["qid", "doc_id", "rank"]].assign(rrf=lambda x: 1.0 / (60 + x["rank"])))
    s = pd.concat(parts).groupby(["qid", "doc_id"], as_index=False)["rrf"].sum()
    s = s.sort_values(["qid", "rrf"], ascending=[True, False])
    s["rank"] = s.groupby("qid").cumcount()
    return s[s["rank"] < n][["qid", "rank", "doc_id"]].assign(set=f"{lang}{n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vi", type=int, default=3000)
    ap.add_argument("--zh", type=int, default=1000)
    ap.add_argument("--en", type=int, default=1000)
    ap.add_argument("--additions", type=Path, required=True)
    ap.add_argument("--vi-pool", type=Path, default=R / "pool5000_vi.parquet",
                    help="VI pool with a rank column (default: dense VI + BM25 doc + BM25 section, RRF 2:1:1)")
    ap.add_argument("--zh-list", action="append", default=[], metavar="[VARIANT=]PATH",
                    help="one ZH ranked list; repeatable, replaces the ZH channel set")
    ap.add_argument("--en-list", action="append", default=[], metavar="[VARIANT=]PATH",
                    help="one EN ranked list; repeatable, replaces the EN channel set")
    args = ap.parse_args()
    lists = {"zh": [parse_list_spec(s) for s in args.zh_list] or DEFAULT_LISTS["zh"],
             "en": [parse_list_spec(s) for s in args.en_list] or DEFAULT_LISTS["en"]}
    pool = pd.read_parquet(args.vi_pool, columns=["qid", "doc_id", "rank"])
    sets = [pool[pool["rank"] < args.vi].assign(set=f"vi{args.vi}"),
            fused("zh", args.zh, lists["zh"]), fused("en", args.en, lists["en"])]
    new = pd.concat(sets, ignore_index=True)[["qid", "set", "rank", "doc_id"]]
    for name, g in new.groupby("set"):
        per_q = g.groupby("qid").size()
        print(f"{name}: {per_q.mean():.0f} docs/query (min {per_q.min()}), {per_q.size} queries")
    if args.additions.exists():  # keep the sets this run does not replace
        old = pd.read_parquet(args.additions)
        old = old[~old["set"].isin(new["set"].unique())]
        out = pd.concat([old, new.astype(old.dtypes.to_dict())], ignore_index=True)
    else:
        out = new
    out.to_parquet(args.additions, index=False)
    print(f"{args.additions}: {len(out):,} rows, sets {sorted(out['set'].unique())}")


if __name__ == "__main__":
    main()
