import pandas as pd, sys, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
c["path"]=u.str.replace(r"^[a-z]+://[^/]+","",regex=True)
def tmpl(p):
    p=re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}","U",p,flags=re.I)
    p=re.sub(r"[0-9a-f]{16,}","H",p,flags=re.I)
    p=re.sub(r"\d+","D",p); return p
print("=== PATHS FOR ZH+missing hosts ===")
for h in ["120ask.com","cnkang.com","familydoctor.com.cn","vinmec.com","pmc-ecm-healthblog.beta.pharmacity.io","youmed.vn","hellobacsi.com","a-hospital.com"]:
    sub=c[c.host==h]
    print("--",h,"n=",len(sub)," http_share=%.2f"%sub.url.str.startswith("http://").mean())
    for k,v in sub.path.apply(tmpl).value_counts().head(4).items(): print(f"     {v:>7}  {k[:140]}")
print()
print("=== ID RANGE BY HOST (sorted by min id) ===")
g=c.groupby("host").id.agg(["min","max","count"]).sort_values("min")
print(g.to_string())
