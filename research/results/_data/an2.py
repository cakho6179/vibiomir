import pandas as pd, sys, re, json
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",250); pd.set_option("display.max_rows",300)
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
host=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
c["host"]=host
vc=host.value_counts()
print("=== ALL 94 HOSTS ===")
for h,n in vc.items(): print(f"{n:>8}  {h}  http_share={ (u[c.host==h].str.startswith('http://')).mean():.2f}")
print()
print("total hosts",len(vc),"share of top10", vc.head(10).sum()/vc.sum())
