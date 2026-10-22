import re,sys,json
sys.stdout.reconfigure(encoding="utf-8")
s=open("medkb_index.js",encoding="utf-8",errors="replace").read()
pats=[r"[\"'`](/api/[A-Za-z0-9_\-./${}]*)[\"'`]",
      r"[\"'`](https://[a-z0-9\-\.]+\.(?:ai|com|vn|io|org|net)[A-Za-z0-9_\-./]*)[\"'`]",
      r"similar_queries",
      r"section_types",
      r"chunk_id",
      r"top_k",
      r"per_page|page_size|limit:",
      r"turnstile|Turnstile|0x4AAAA",
      r"auth_token|localStorage",
      r"X-Medical-Anti-Bot"]
for pat in pats:
    ms=re.findall(pat,s)
    uniq=sorted(set(ms))
    print("=== ",pat," total:",len(ms)," uniq:",len(uniq))
    for m in uniq[:40]:
        print("     ",(m if isinstance(m,str) else str(m))[:150])
