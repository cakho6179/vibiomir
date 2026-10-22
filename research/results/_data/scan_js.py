import re,sys
sys.stdout.reconfigure(encoding="utf-8")
s=open("medkb_index.js",encoding="utf-8",errors="replace").read()
print("len",len(s))
pats=[r"fetch\([^)]{0,140}", r"baseURL[^,;]{0,100}", r"[\"'`](/(?:api|v1|search|docs|corpus|query)[A-Za-z0-9_\-./{}:?=&%]*)[\"'`]", r"VITE_[A-Z_]+", r"XMLHttpRequest", r"axios"]
for pat in pats:
    ms=set(re.findall(pat,s))
    print("=== PATTERN",pat,"count",len(ms))
    for m in sorted(ms)[:50]: print("   ",(m if isinstance(m,str) else str(m))[:170])
