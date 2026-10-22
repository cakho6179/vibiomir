import pandas as pd, sys, re, json, collections
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
path=u.str.replace(r"^[a-z]+://[^/]+","",regex=True)
print("=== QUERY-STRING / FRAGMENT / ENCODING ===")
print("with '?':",path.str.contains(r"\?").mean().round(4), path.str.contains(r"\?").sum())
print("with '#':",path.str.contains(r"#").sum())
print("with final slash:",path.str.endswith("/").sum())
print("percent-encoded (%XX):",path.str.contains(r"%[0-9A-Fa-f]{2}").sum())
print("non-ascii chars:",path.str.contains(r"[^\x00-\x7F]").sum())
print("uppercase in path:",path.str.contains(r"[A-Z]").sum())
print("depth distribution:")
d=path.str.strip("/").str.replace(r"^$","",regex=True).apply(lambda s: 0 if s=="" else s.count("/")+1)
print(d.value_counts().sort_index().to_string())
print()
print("=== PATH TEMPLATES (digits->D, long hex->H, uuid->U) ===")
def tmpl(p):
    p=re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}","U",p,flags=re.I)
    p=re.sub(r"[0-9a-f]{16,}","H",p,flags=re.I)
    p=re.sub(r"\d+","D",p)
    return p
t=path.apply(tmpl)
print(t.value_counts().head(25).to_string())
print()
print("=== PER-HOST TEMPLATE SAMPLES (top 14 hosts) ===")
c["path"]=path; c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0]
for h in ["pubmed.ncbi.nlm.nih.gov","suckhoecongdongonline.vn","120ask.com","suckhoedoisong.vn","nhathuoclongchau.com.vn","thanhnien.vn","cnkang.com","ask.39.net","laodong.vn","phunusuckhoe.giadinhonline.vn","familydoctor.com.cn","vinmec.com","medlatec.vn","giadinhonline.vn"]:
    sub=c[c.host==h]
    tt=sub.path.apply(tmpl).value_counts().head(4)
    print("--",h,"n=",len(sub))
    for k,v in tt.items(): print(f"     {v:>7}  {k[:150]}")
