import json,sys,pandas as pd,numpy as np
sys.stdout.reconfigure(encoding="utf-8")
d=json.load(open("liveness_get.json",encoding="utf-8"))
df=pd.DataFrame(d)
print("n tested:",len(df))
print(df.groupby("host").agg(n=("url","size"),ok200=("status",lambda s:(s==200).sum()),med_chars=("text_chars","median"),med_words=("text_words","median"),med_bytes=("bytes","median"),med_sec=("sec","median")).to_string())
print()
print("STATUS:",df.status.value_counts(dropna=False).to_dict())
df.to_csv("liveness_get.csv",index=False)
# aggregate reachable text yield
reach=df[(df.status==200)&(df.text_chars>500)]
print("reachable+parsed:",len(reach),"/",len(df))
print("median words of parsed:",reach.text_words.median(),"median chars:",reach.text_chars.median())
print("median HTML bytes:",df[df.status==200].bytes.median())
