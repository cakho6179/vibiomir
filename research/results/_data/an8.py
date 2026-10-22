import pandas as pd, sys, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
c["path"]=u.str.replace(r"^[a-z]+://[^/]+","",regex=True)
# URL-level near-duplicate detection: lowercase-normalized full url
lu=u.str.lower().str.rstrip("/")
print("exact dup (after lower+strip slash):",lu.duplicated().sum())
# path-level dup across hosts (same article path reused)
c["p_norm"]=c.path.str.lower().str.rstrip("/")
pd_=c.p_norm.duplicated().sum()
print("paths appearing on >1 host:",pd_)
# duplicate/pagination patterns: trailing page numbers
print()
print("=== SUSPECTED PAGINATED / BOILERPLATE PATTERNS ===")
for pat,name in [(r"-\d{1,3}\.html?$","trailing page number"),(r"page[=/]\d+","page param"),(r"\?id=\d+","?id="),(r"[?&]p=\d+","p param"),(r"tag/|/tags?/","tag pages"),(r"category|/chuyen-muc|/chuyen-de|/danh-muc","category pages"),(r"sitemap","sitemap")]:
    m=c.path.str.contains(pat,regex=True,case=False)
    print(f"  {name:<22} {m.sum():>8}  ({m.mean()*100:.2f}%)")
print()
print("=== VN NEWS HOST: article-suffix patterns ===")
for h in ["suckhoedoisong.vn","thanhnien.vn","laodong.vn","giadinhonline.vn","vietnamnet.vn","nhandan.vn","vietnamplus.vn","baocantho.com.vn"]:
    s=c[c.host==h].path
    if len(s)==0: continue
    ext=s.str.extract(r"(\.[a-z]{2,4})$")[0].fillna("(none)").value_counts().head(3).to_dict()
    print(f"  {h:<28} n={len(s):>7} ext={ext}")
print()
print("=== ZH path detail ===")
for h in ["120ask.com","cnkang.com","familydoctor.com.cn","ask.39.net","a-hospital.com","zysjonline.com","zhongyibaodian.net","youlai.cn"]:
    s=c[c.host==h]
    print(f"  {h:<24} n={len(s):>7} sample:")
    for x in s.path.head(3): print("       ",x[:130])
