import pyarrow.parquet as pq, pandas as pd, sys, re, json, collections
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",200)
c=pq.read_table("links_corpus_vi_en_zh.parquet").to_pandas()
qd=pq.read_table("query.parquet").to_pandas()
print("=== SHAPES ==="); print("corpus",c.shape,"query",qd.shape)
print("corpus dtypes",dict(c.dtypes.astype(str)))
print("id nulls",c.id.isna().sum(),"url nulls",c.url.isna().sum())
print("id min/max",c.id.min(),c.id.max(),"unique ids",c.id.nunique())
print("url unique",c.url.nunique(),"dupe urls",len(c)-c.url.nunique())
print("dup id count",(c.id.duplicated()).sum())
# gaps / contiguity
s=c.id.sort_values().values
print("id range span",s[-1]-s[0]+1,"present",len(s),"density",round(len(s)/(s[-1]-s[0]+1),4))
# id clusters
print("first 20 ids",s[:20])
print("last 10 ids",s[-10:])
# scheme
u=c.url.astype(str)
print("=== SCHEME ==="); print(u.str.extract(r"^(https?)://")[0].value_counts(dropna=False))
host=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
print("unique hosts",host.nunique())
print("=== TOP 30 HOSTS ===")
print(host.value_counts().head(30).to_string())
print("=== TLDS ===")
print(host.str.extract(r"\.([a-z]+)$")[0].value_counts().head(20).to_string())
c.to_pickle("corpus.pkl"); qd.to_pickle("query.pkl")
