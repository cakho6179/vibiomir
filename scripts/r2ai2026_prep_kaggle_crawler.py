"""Generate lightweight Kaggle CPU crawler bundles linked to vibiomir-missing-targets dataset.

Partitions:
  - P1: Vinmec, HelloBacsi, Pharmacity (High priority gold medical Q&A)
  - P2: Nha Thuoc Long Chau (Pharmaceuticals & clinical guides)
  - P3: Vietnamese Health Portals & Regional News (suckhoecongdong, vov, etc.)
  - P4: Chinese 120ask.com (Large clinical Q&A)
  - P5: Chinese Medical Portals (ask.39.net, familydoctor.com.cn, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CRAWLER_CODE_TEMPLATE = '''"""ViBioMIR Distributed Cloud Crawler (Kaggle CPU Edition)
Partition: __PARTITION_NAME__ (__TOTAL_URLS__ URLs)
Target file: __PARTITION_FILE__.parquet
"""

import asyncio
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

# Install extraction libraries
os.system("pip install -q httpx trafilatura beautifulsoup4 pyarrow")

import httpx
import pandas as pd
import trafilatura
from bs4 import BeautifulSoup

def extract_clean_text(html: str) -> tuple[str, str]:
    if not html or len(html) < 100:
        return "", ""
    try:
        soup = BeautifulSoup(html[:15000], "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:
        title = ""
        
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True
        ) or ""
    except Exception:
        text = ""
        
    if not text:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = " ".join(soup.stripped_strings)
        except Exception:
            text = ""
            
    return title, text

async def fetch_one(client: httpx.AsyncClient, doc_id: int, url: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        for attempt in range(2):
            try:
                resp = await client.get(url, timeout=12.0, follow_redirects=True)
                if resp.status_code == 200:
                    title, text = extract_clean_text(resp.text)
                    if len(text) >= 60:
                        return {"id": doc_id, "url": url, "title": title, "text": text, "ok": 1}
                    return {"id": doc_id, "url": url, "title": title, "text": text, "ok": 0}
                elif resp.status_code in (404, 410):
                    return {"id": doc_id, "url": url, "title": "", "text": "", "ok": 0}
            except Exception:
                await asyncio.sleep(0.5)
        return {"id": doc_id, "url": url, "title": "", "text": "", "ok": 0}

async def main():
    ds_dir = Path("/kaggle/input/vibiomir-missing-targets")
    p_file = ds_dir / "__PARTITION_FILE__.parquet"
    if not p_file.exists():
        # Fallback search
        p_candidates = list(Path("/kaggle/input").glob("**/__PARTITION_FILE__.parquet"))
        if p_candidates:
            p_file = p_candidates[0]
        else:
            raise FileNotFoundError(f"Could not find __PARTITION_FILE__.parquet in /kaggle/input")
            
    print(f"Reading targets from {p_file}...")
    df_targets = pd.read_parquet(p_file)
    targets = list(zip(df_targets["id"].astype(int), df_targets["url"].astype(str)))
    print(f"Loaded {len(targets):,} URLs for partition: __PARTITION_NAME__")
    
    sem = asyncio.Semaphore(50)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,zh-CN;q=0.8",
    }
    
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "__PARTITION_NAME__.parquet"
    
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=64)
    results = []
    success_count = 0
    start_time = time.time()
    
    async with httpx.AsyncClient(headers=headers, limits=limits, verify=False) as client:
        batch_size = 500
        for i in range(0, len(targets), batch_size):
            chunk = targets[i:i + batch_size]
            tasks = [fetch_one(client, item[0], item[1], sem) for item in chunk]
            batch_res = await asyncio.gather(*tasks)
            for r in batch_res:
                if r["ok"] == 1:
                    results.append(r)
                    success_count += 1
            
            elapsed = time.time() - start_time
            rate = (i + len(chunk)) / max(elapsed, 1.0)
            print(f"[{i + len(chunk):,}/{len(targets):,}] OK: {success_count:,} ({rate:.1f} urls/s)", flush=True)
            
            if len(results) >= 5000:
                df = pd.DataFrame(results)
                df.to_parquet(out_parquet, index=False)
                
    df = pd.DataFrame(results)
    df.to_parquet(out_parquet, index=False)
    print(f"Finished! Stored {len(df):,} valid pages in {out_parquet}")

if __name__ == "__main__":
    asyncio.run(main())
'''

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("kaggle/crawler_kernels"))
    parser.add_argument("--username", type=str, default="snnguynvnk19hl")
    args = parser.parse_args(argv)

    partitions = [
        ("vibiomir-crawl-p1-vinmec", "p1_vinmec", 43209, "Vinmec, HelloBacsi, Pharmacity (High Yield)"),
        ("vibiomir-crawl-p2-longchau", "p2_longchau", 73422, "Nha Thuoc Long Chau"),
        ("vibiomir-crawl-p3-vnhealth", "p3_vnhealth", 180585, "Vietnamese Health News & Provincial Sites"),
        ("vibiomir-crawl-p4-120ask", "p4_120ask", 98159, "Chinese 120ask.com"),
        ("vibiomir-crawl-p5-zhmed", "p5_zhmed", 71323, "Chinese 39.net, familydoctor, zhongyi"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for p_name, p_file, p_cnt, p_desc in partitions:
        p_dir = args.out_dir / p_name
        p_dir.mkdir(parents=True, exist_ok=True)

        code = (
            CRAWLER_CODE_TEMPLATE
            .replace("__PARTITION_NAME__", p_name)
            .replace("__PARTITION_FILE__", p_file)
            .replace("__TOTAL_URLS__", f"{p_cnt:,}")
        )

        with open(p_dir / "crawl.py", "w", encoding="utf-8") as f:
            f.write(code)

        meta = {
            "id": f"{args.username}/{p_name}",
            "title": p_name,
            "code_file": "crawl.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [
                f"{args.username}/vibiomir-missing-targets"
            ],
            "competition_sources": [],
            "kernel_sources": [],
        }

        with open(p_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Generated {p_name:32s} (Linked to {p_file}.parquet) -> {p_dir}")

    print("\nAll 5 Kaggle CPU crawler partitions updated to lightweight dataset architecture!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
