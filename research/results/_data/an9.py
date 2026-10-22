import json,sys,pandas as pd,numpy as np,re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0,".")
py=json.load(open("medkb_query_probe.json",encoding="utf-8"))
ok=[x for x in py if "err" not in x]
sect=sum(x["sect"] for x in ok); print("total section_types across 196 results:",sect,"avg",round(sect/196,2))
print("text length per query (median over 20 results):",[round(x["textlen_med"]) for x in ok])
print("max text in each query:",[round(x["maxlen"]) for x in ok])
print("langs per query sample:",[x["langs"] for x in ok[:6]])
print("doc_types per query:",[x["dtypes"] for x in ok[:5]])
print()
# queries
q=pd.read_pickle("query.pkl")
w=q["query"].astype(str).str.split().apply(len)
print("=== QUERY FILE ===")
print("n",len(q),"id min/max",q.id.min(),q.id.max(),"contiguous",bool((np.diff(sorted(q.id))==1).all()))
print("word stats: min %d p25 %d median %d p75 %d p95 %d max %d"%(w.min(),np.percentile(w,25),w.median(),np.percentile(w,75),np.percentile(w,95),w.max()))
print("char len stats: min %d median %d max %d"%(q["query"].str.len().min(),q["query"].str.len().median(),q["query"].str.len().max()))
print()
print("=== 15 SAMPLE QUERIES ===")
for i in [0,1,2,100,300,500,700,900,1000,1100,1199,50,150,250,350]:
    print(f"  id={q.id.iloc[i]:>4} | {q['query'].iloc[i][:190]}")
print()
# keyword/intent signal
allq=" ".join(q["query"].astype(str)).lower()
for kw in ["thuốc","triệu chứng","bệnh","điều trị","xét nghiệm","có sao không","nguy hiểm","mang thai","trẻ","bao lâu","nên","tôi","bị","đau","sốt","ung thư","tiểu đường","huyết áp","gan","thận","tim","phổi","dạ dày","covid"]:
    c=allq.count(kw)
    if c: print(f"   {kw:<18} {c:>5}  ({c/len(q)*100:.1f}% of queries)")
print()
print("avg words:",round(w.mean(),1))
