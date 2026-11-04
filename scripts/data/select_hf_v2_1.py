"""Select the conservative v2.1 merge set from the tmh1008 external crawl.

The selection is intentionally split from the docstore builder so every
decision is auditable before any asset is rewritten.

Missing documents
-----------------
* only rows flagged ``is_dup == False`` by the external audit;
* only VI/ZH; IDs must be absent from v2 and present in the official link list;
* remove the two known template farms (``thaythuocvietnam.vn`` and
  ``zhongyibaodian.net``);
* apply the conservative external cleaner and require >=200 cleaned chars.

Better bodies
-------------
Only existing VI documents on hosts where the crawl is a full article rather
than a category/menu page.  The gates are deliberately high precision:

* >=1.3x longer after cleaning and at least 400 extra chars;
* >=95% of the old body tokens are still present (same article);
* the cleaned crawl contains the old body, or token coverage is >=98%;
* no more than one boilerplate cue after cleaning;
* medical title (the current retrieval target is health queries);
* skip duplicate anchors and content-key collisions in the builder.

Usage::

    VIBIOMIR_HF_DIR=/path/to/hf_tmh_conv/default/train \
      python scripts/data/select_hf_v2_1.py --force
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import orjson
import pandas as pd
import pyarrow.dataset as ds

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.corpus.processing import content_key  # noqa: E402
from scripts.data.external_clean import clean_external_body  # noqa: E402

CANDIDATES = ROOT / "data" / "external" / "hf_tmh" / "candidates.parquet"
FEATURES = ROOT / "data" / "external" / "hf_tmh" / "vi_refresh_features.parquet"
LINKS = ROOT / "data" / "raw" / "btc" / "links_corpus_vi_en_zh.parquet"
BASE = ROOT / "data" / "docstore" / "v2"
OUT = ROOT / "data" / "external" / "hf_tmh"

BAD_HOSTS = {"thaythuocvietnam.vn", "zhongyibaodian.net", "www.cnkang.com"}
MISSING_VI_BAD_HOSTS = {
    "baolangson.vn", "baovinhlong.com.vn", "khoahocphothong.vn", "medlatec.vn",
    "phuyen.baodaklak.vn", "suckhoecongdongonline.vn", "vietnamnet.vn",
}
PAYWALL = ("Tại trang vừa mở", "Bạn cần đăng nhập", "Đăng nhập để đọc",
           "Đăng ký để đọc", "Vui lòng đăng nhập")

# Hosts observed to expose the full article in the HF body.  Category/menu
# pages (vinmec, hellobacsi, medlatec, youmed) are deliberately absent.
REFRESH_HOSTS = {
    "thanhnien.vn",
    "phuyen.baodaklak.vn",
    "www.vietnamplus.vn",
    "baogialai.com.vn",
    "khoahocphothong.vn",
    "suckhoedoisong.vn",
    "tienphong.vn",
    "tuoitre.vn",
    "baonghean.vn",
    "vietnamnet.vn",
    "baohaiphong.vn",
    "baodanang.vn",
    "phunusuckhoe.giadinhonline.vn",
    "baovinhlong.com.vn",
    "baoquangtri.vn",
    "baocantho.com.vn",
    "www.qdnd.vn",
    "nhandan.vn",
    "suckhoeviet.org.vn",
    "giadinhonline.vn",
    "tiemchunglongchau.com.vn",
    "bachmai.gov.vn",
    "baochinhphu.vn",
    "www.sggp.org.vn",
    "baoquangninh.vn",
    "vnexpress.net",
    "vov2.vov.vn",
    "benhvienvietduc.org",
    "baolangson.vn",
    "benhviennhitrunguong.gov.vn",
}

HEALTH = re.compile(
    r"(?<!\w)(?:bệnh|sức khỏe|ung thư|gan|huyết áp|tiểu đường|dạ dày|thuốc|"
    r"triệu chứng|điều trị|viêm|nhiễm|đau|sốt|bác sĩ|bệnh viện|xét nghiệm|"
    r"dinh dưỡng|máu|phổi|thận|não|khớp|mũi|họng|răng|vắc xin|vaccine|"
    r"virus|vi khuẩn|ký sinh|nội tiết|tiêu hóa|hô hấp|tuần hoàn|thần kinh|"
    r"tâm thần|ung bướu|canxi|vitamin|men gan|cholesterol|đột quỵ|nhồi máu|"
    r"loãng xương|béo phì|sụt cân|mệt mỏi|chóng mặt|buồn nôn|táo bón|tiêu chảy|"
    r"khó thở|đau đầu|mất ngủ|chán ăn|ngứa|nổi mẩn|mụn|vết thương|gãy xương|"
    r"thoái hóa|thoát vị|trĩ|phụ khoa|nam khoa|nhi khoa|chẩn đoán|phác đồ|"
    r"tác dụng phụ|sản khoa|phụ sản|sản phụ|hậu sản|mang thai|thai nhi|thai kỳ|"
    r"bà bầu|mẹ bầu|trẻ em|trẻ nhỏ|trẻ sơ sinh|suy thận|suy tim|suy hô hấp|"
    r"suy nhược|suy giảm|xơ gan|xơ vữa|xơ phổi|mỡ máu|mỡ thừa|tiểu đường|"
    r"đường huyết|liều dùng|liều thuốc|quá liều|bệnh mắt|mắt đỏ|thị lực|"
    r"nhãn khoa|tai mũi họng|ù tai|đau tai|thính lực|da liễu|bệnh da|ngứa da|"
    r"viêm da|ho khan|ho có đờm|ho ra máu|ho gà|sỏi thận|sỏi mật|"
    r"tim mạch|bệnh tim|suy tim|nhồi máu cơ tim|đau tim|răng miệng|viêm miệng|"
    r"loét miệng|ung thư miệng)(?!\w)",
    re.IGNORECASE,
)
SITE_SUFFIX = re.compile(
    r"\s*[|–-]\s*(?:Sức Khỏe Cộng Đồ(?:ng)?|Phụ Nữ Sức Khỏe|Tuổi Trẻ Online|"
    r"Phú Yên online|Báo [^|]{2,30}|Tạp chí [^|]{2,30}|Sức khỏe đời sống)\s*$",
    re.IGNORECASE,
)

CUE = re.compile(
    r"(?i)(Menu|Trang chủ|Đăng nhập|Đăng ký|Liên hệ|Hotline|Bình luận|Chia sẻ|"
    r"Đọc nhiều|Bài viết|Quý khách|Lựa chọn dịch vụ|Nguồn:|Ảnh:|Copyright|"
    r"相关阅读|推荐阅读|上一篇|下一篇)"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def toks(s: str) -> set[str]:
    return set(re.findall(r"\w+", norm(s)))


def read_v2_bodies(ids: set[int]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for path in sorted((BASE / "corpus").glob("part-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                doc_id = int(rec["id"])
                if doc_id in ids:
                    out[doc_id] = (str(rec.get("t", "")), str(rec.get("x", "")))
        log(f"  v2 bodies {path.name}: {len(out):,}/{len(ids):,}")
    return out


def select_missing(base_meta: pd.DataFrame, links: pd.DataFrame) -> tuple[list[dict], dict]:
    c = pd.read_parquet(CANDIDATES)
    base_ids = set(base_meta["id"].astype("int64"))
    official = set(links["id"].astype("int64"))
    c = c[(~c["is_dup"]) & c["lang"].isin(["vi", "zh"])].copy()
    c = c[~c["id"].astype("int64").isin(base_ids)]
    c = c[~c["host"].isin(BAD_HOSTS)]
    c = c[c["id"].astype("int64").isin(official)]
    log(f"missing candidates after host/id gates: {len(c):,}")

    rows: list[dict] = []
    dropped: Counter = Counter()
    for r in c.itertuples(index=False):
        lang = str(r.lang)
        if lang == "vi":
            if str(r.host) in MISSING_VI_BAD_HOSTS:
                dropped[f"bad_host:{r.host}"] += 1
                continue
            if "OPC" in str(r.title):
                dropped["vi_advertorial"] += 1
                continue
        body = clean_external_body(str(r.title), str(r.body), lang, str(r.host))
        if len(body) < 200:
            dropped[f"clean_short:{r.host}"] += 1
            continue
        if lang == "vi":
            if len(body) < 400:
                dropped[f"vi_short:{r.host}"] += 1
                continue
            if any(m in body for m in PAYWALL):
                dropped[f"paywall:{r.host}"] += 1
                continue
        if lang == "zh" and str(r.host).endswith("120ask.com"):
            if "健康咨询描述" not in body and "病情分析" not in body:
                dropped[f"120ask_unstructured:{r.host}"] += 1
                continue
        rows.append({
            "id": int(r.id), "url": str(r.url), "host": str(r.host), "lang": str(r.lang),
            "title": str(r.title), "body": body, "kind": "missing",
            "raw_chars": int(r.n_chars), "clean_chars": len(body),
        })
    rows.sort(key=lambda x: x["id"])
    stats = {
        "selected": len(rows),
        "by_lang": dict(Counter(x["lang"] for x in rows)),
        "by_host": dict(Counter(x["host"] for x in rows).most_common()),
        "chars": sum(x["clean_chars"] for x in rows),
        "dropped": dict(dropped),
    }
    return rows, stats


def select_refresh(base_meta: pd.DataFrame, hf_dir: Path | None) -> tuple[list[dict], dict]:
    f = pd.read_parquet(FEATURES)
    base_ids = set(base_meta["id"].astype("int64"))
    base_url = dict(zip(base_meta["id"].astype("int64"), base_meta["url"].astype(str)))
    f = f[f["id"].astype("int64").isin(base_ids)].copy()
    f["clean_title"] = f["title"].astype(str).map(lambda x: SITE_SUFFIX.sub("", x).strip())
    f["health"] = f["clean_title"].str.contains(HEALTH, regex=True, na=False)
    gates = (
        (f["host"].isin(REFRESH_HOSTS))
        & f["health"]
        & (f["ratio"] >= 1.3)
        & (f["cov"] >= 0.98)
        & (f["cues"] <= 1)
        & (f["hf_n"] >= 500)
    )
    cand = f[gates].copy().sort_values("id")
    log(f"refresh candidates after host/health/gate filters: {len(cand):,}")
    target = set(cand["id"].astype("int64"))
    old = read_v2_bodies(target)
    missing = target - set(old)
    if missing:
        raise SystemExit(f"v2 bodies missing for {len(missing)} refresh ids")

    # Pull the raw crawl rows for the selected ids.  Features carry only the
    # comparison numbers, not the body.
    raw: dict[int, tuple[str, str]] = {}
    if hf_dir is None:
        raise SystemExit("refresh selection needs --hf-dir or VIBIOMIR_HF_DIR")
    d = ds.dataset(hf_dir, format="parquet")
    for batch in d.to_batches(columns=["id", "title", "content", "success"], batch_size=50000):
        for rec in batch.to_pylist():
            i = int(rec["id"])
            if i in target and bool(rec["success"]):
                raw[i] = (str(rec.get("title") or ""), str(rec.get("content") or ""))
    log(f"raw crawl rows loaded: {len(raw):,}/{len(target):,}")

    rows: list[dict] = []
    rejected: Counter = Counter()
    for r in cand.itertuples(index=False):
        i = int(r.id)
        title, content = raw.get(i, ("", ""))
        if not content:
            rejected["no_raw"] += 1
            continue
        old_title, old_body = old[i]
        body = clean_external_body(title, content, "vi", str(r.host))
        if len(body) < 500:
            rejected["clean_short"] += 1
            continue
        if any(m in body for m in PAYWALL):
            rejected["paywall"] += 1
            continue
        old_n = max(len(old_body), 1)
        if len(body) < old_n * 1.3 or len(body) - old_n < 400:
            rejected["gain_too_small"] += 1
            continue
        ot, nt = toks(old_body), toks(body)
        cov = len(ot & nt) / max(len(ot), 1)
        if cov < 0.95:
            rejected["coverage_low"] += 1
            continue
        if not (norm(old_body) in norm(body) or cov >= 0.98):
            rejected["not_same_article"] += 1
            continue
        if len(CUE.findall(body)) > 1:
            rejected["boilerplate_cue"] += 1
            continue
        rows.append({
            "id": i, "url": str(base_url.get(i, "")), "host": str(r.host), "lang": "vi",
            "title": title or old_title, "body": body, "kind": "refresh",
            "raw_chars": len(content), "clean_chars": len(body),
            "old_chars": len(old_body), "cov": round(cov, 6),
        })
    rows.sort(key=lambda x: x["id"])
    stats = {
        "selected": len(rows),
        "by_host": dict(Counter(x["host"] for x in rows).most_common()),
        "old_chars": sum(x["old_chars"] for x in rows),
        "new_chars": sum(x["clean_chars"] for x in rows),
        "gain_chars": sum(x["clean_chars"] - x["old_chars"] for x in rows),
        "rejected": dict(rejected),
        "rules": {
            "hosts": sorted(REFRESH_HOSTS), "min_ratio": 1.3, "min_gain": 400,
            "min_cov": 0.95, "same_article_cov": 0.98, "max_cues": 1,
        },
    }
    return rows, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wb", compresslevel=5) as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--add-only", action="store_true", help="rebuild only the missing-doc selection")
    ap.add_argument("--refresh-only", action="store_true", help="rebuild only the better-body selection")
    ap.add_argument("--hf-dir", type=Path,
                    default=Path(os.environ["VIBIOMIR_HF_DIR"]) if os.environ.get("VIBIOMIR_HF_DIR") else None,
                    help="parquet directory of tmh1008/ViBioMIR-crawled-corpus (or set VIBIOMIR_HF_DIR)")
    ap.add_argument("--add-out", type=Path, default=OUT / "v2_1_add.jsonl.gz")
    ap.add_argument("--refresh-out", type=Path, default=OUT / "v2_1_refresh.jsonl.gz")
    ap.add_argument("--report-out", type=Path, default=OUT / "v2_1_selection.json")
    args = ap.parse_args()

    for p in (args.add_out, args.refresh_out, args.report_out):
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists; use --force")

    base_meta = pd.read_parquet(BASE / "docs_meta.parquet")
    links = pd.read_parquet(LINKS, columns=["id"])
    log(f"base {len(base_meta):,} docs; links {len(links):,}")

    previous = {}
    if args.report_out.exists():
        try:
            previous = json.loads(args.report_out.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

    add: list[dict] = []
    refresh: list[dict] = []
    if not args.refresh_only:
        add, add_stats = select_missing(base_meta, links)
        log(f"missing selected: {len(add):,} {add_stats['by_lang']}")
        write_jsonl(args.add_out, add)
    else:
        add_stats = previous.get("missing", {})
    if not args.add_only:
        refresh, refresh_stats = select_refresh(base_meta, args.hf_dir)
        log(f"refresh selected: {len(refresh):,}")
        write_jsonl(args.refresh_out, refresh)
    else:
        refresh_stats = previous.get("refresh", {})

    add_ids = {x["id"] for x in add} if add else set()
    refresh_ids = {x["id"] for x in refresh} if refresh else set()
    if add_ids & refresh_ids:
        raise SystemExit("add/refresh id overlap")

    report = {
        "base": str(BASE),
        "inputs": {
            "candidates": str(CANDIDATES), "features": str(FEATURES), "links": str(LINKS),
        },
        "missing": add_stats,
        "refresh": refresh_stats,
        "add_ids": len(add_ids) if add else int(previous.get("add_ids", 0)),
        "refresh_ids": len(refresh_ids) if refresh else int(previous.get("refresh_ids", 0)),
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {args.add_out} ({args.add_out.stat().st_size / 1e6:.1f} MB)")
    log(f"wrote {args.refresh_out} ({args.refresh_out.stat().st_size / 1e6:.1f} MB)")
    log(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
