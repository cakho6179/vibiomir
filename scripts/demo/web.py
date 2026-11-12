"""Local demo website: pick any of the 1,200 questions and replay it through the production run, step by step.

    python run.py --web                 # http://127.0.0.1:8765  (python run.py --web --port 9000 for another port)

Standard library only (http.server), bound to 127.0.0.1. Each question page is scripts/demo/trace_query.py's
walkthrough, built on request; the checks against the submitted lists run on every page. Document and chunk text come
from `python run.py --demo --prepare --qids all` (once, a few minutes).
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import trace_query as tq  # noqa: E402

FEATURED = [(401, "xét nghiệm IgE: phần lớn tài liệu được chọn là tiếng Trung"),
            (556, "Kwashiorkor: tài liệu tiếng Anh nhiều nhất"),
            (1, "câu dài, nhiều bệnh: độ sâu lớn (324 tài liệu)")]


def build_index() -> list[dict]:
    q = pd.read_parquet(tq.ROOT / "data/raw/btc/query.parquet")
    n = pd.read_parquet(tq.PG / "n_per_query.parquet").set_index("qid")
    lists = pd.read_parquet(tq.LISTS["pN3_base"])
    docs = lists[(lists["job"] == "pN3_base") & (lists["kind"] == "doc")][["qid", "doc_id"]]
    lang = pd.read_parquet(tq.PG / "cands_p.parquet", columns=["qid", "doc_id", "lang"])
    mix = docs.merge(lang, on=["qid", "doc_id"]).groupby(["qid", "lang"]).size().unstack(fill_value=0)
    have = set(pd.read_parquet(tq.DOCS, columns=["qid"])["qid"]) if tq.DOCS.exists() else set()
    rows = []
    for r in q.itertuples():
        qid = int(r.id)
        m = mix.loc[qid] if qid in mix.index else {}
        rows.append({"qid": qid, "q": str(r.query), "n": int(n.loc[qid, "n_docs"]),
                     "mix": " · ".join(f"{k.upper()} {int(m.get(k, 0))}" for k in ("vi", "zh", "en")),
                     "zh": int(m.get("zh", 0)), "en": int(m.get("en", 0)), "text": qid in have})
    return rows


HOME_CSS = tq.CSS + """
.wrap{max-width:1300px;margin:0 auto;padding:24px 28px 60px} h1{font-size:34px;margin:8px 0}
input{font:20px 'Segoe UI',Arial;padding:10px 14px;width:100%;border:2px solid var(--line);border-radius:8px;margin:10px 0 16px}
.feat{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:12px 0 24px}
.feat a{display:block;background:var(--card);border:2px solid var(--line);border-left:6px solid var(--acc);padding:14px;color:var(--ink);text-decoration:none}
.feat a:hover,.qrow:hover{border-color:var(--acc)} .feat b{font-size:20px} .feat span{display:block;color:var(--mute);font-size:15px}
table.qs td{font-size:16px} table.qs a{color:var(--ink);text-decoration:none} table.qs tr:hover{background:#fff3c4}
"""


def cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


def home(rows: list[dict]) -> str:
    e = html.escape
    feat = "".join(f'<a href="/q/{q}"><b>#{q}</b> {e(cut(next(r["q"] for r in rows if r["qid"] == q), 90))}'
                   f"<span>{e(why)}</span></a>" for q, why in FEATURED)
    trs = "".join(f'<tr data-s="{e((str(r["qid"]) + " " + r["q"]).lower())}"><td><a href="/q/{r["qid"]}">#{r["qid"]}</a></td>'
                  f'<td><a href="/q/{r["qid"]}">{e(cut(r["q"], 150))}</a></td><td class=n>{r["n"]}</td><td style="white-space:nowrap">{r["mix"]}</td></tr>'
                  for r in rows)
    js = """const I=document.getElementById('s'),R=[...document.querySelectorAll('tr[data-s]')];
I.oninput=()=>{const v=I.value.toLowerCase().trim();let k=0;R.forEach(r=>{const on=!v||r.dataset.s.includes(v);r.style.display=on?'':'none';k+=on});
document.getElementById('c').textContent=k;};"""
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ViBioMIR · demo</title><style>{HOME_CSS}</style></head><body>
<header><b>ViBioMIR · Băng Cá Khô</b><span style="flex:1"></span><a class=home href="/random">câu ngẫu nhiên →</a></header>
<div class=wrap><h1>Phát lại hệ thống trên 1.200 câu hỏi của cuộc thi</h1>
<p class=note>Chọn một câu hỏi để xem từng bước, từ thu thập dữ liệu tới bài nộp, với một tài liệu thật được theo dõi xuyên suốt.
Mọi số lấy từ lần chạy sản xuất. Mỗi trang tự đối chiếu với danh sách đã nộp.</p>
<div class=feat>{feat}</div>
<input id=s placeholder="Tìm theo số câu hoặc từ khoá, ví dụ: tiểu đường, IgE, 556" autofocus>
<p class=note><span id=c>{len(rows)}</span> câu hỏi · N_q = số tài liệu nộp (#7892) · ngôn ngữ của các tài liệu đó</p>
<table class=qs><tr><th>#</th><th>Câu hỏi</th><th class=n>N_q</th><th>Ngôn ngữ</th></tr>{trs}</table></div>
<script>{js}</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []

    def send(self, code: int, body: str, ctype: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if not parts:
                return self.send(200, home(self.rows))
            if parts == ["random"]:
                self.send_response(302)
                self.send_header("Location", f"/q/{random.randint(1, 1200)}")
                return self.end_headers()
            if len(parts) == 2 and parts[0] == "q" and parts[1].isdigit():
                doc = urllib.parse.parse_qs(url.query).get("doc", [None])[0]
                t0 = time.time()
                t = tq.trace(int(parts[1]), int(doc) if doc and doc.isdigit() else None)
                ok = all(g == w for g, w in t["checks"].values())
                t["web"] = True  # grid cells become links to ?doc=
                page = tq.render_html(t, ok, home="/")
                print(f"q{parts[1]} doc {t['doc']} {'OK' if ok else 'MISMATCH'} {time.time() - t0:.1f}s", flush=True)
                return self.send(200, page)
            return self.send(404, "<p>Không có trang này. <a href='/'>Về danh sách</a></p>")
        except SystemExit as e:
            return self.send(400, f"<p>{html.escape(str(e))}</p><p><a href='/'>Về danh sách</a></p>")
        except Exception:  # keep the server up during a demo
            traceback.print_exc()
            return self.send(500, f"<pre>{html.escape(traceback.format_exc())}</pre>")

    def log_message(self, *args) -> None:  # quiet: one line per page from do_GET
        pass


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--warm", type=int, default=401, help="question rendered at start so tables are loaded (0: none)")
    args = ap.parse_args()
    tq.CACHE = {}
    t0 = time.time()
    Handler.rows = build_index()
    if args.warm:
        tq.trace(args.warm, None)
    print(f"ready in {time.time() - t0:.0f}s: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
