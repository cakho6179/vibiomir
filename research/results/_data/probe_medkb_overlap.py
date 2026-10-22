import json,sys,time,random,concurrent.futures as cf
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
qd=pd.read_pickle("query.pkl"); c=pd.read_pickle("corpus.pkl")
corpus_urls=set(c.url.astype(str))
E="https://medkb.tinix.ai/api/search"
H={"Content-Type":"application/json","X-Medical-Anti-Bot":"human-verified","Origin":"https://medkb.tinix.ai","Referer":"https://medkb.tinix.ai/search","User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
import urllib.request
def one(row):
    body={"query":row["query"],"top_k":20,"offset":0,"year":0,"year_from":0,"year_to":0,"lang":"","doc_type":""}
    req=urllib.request.Request(E,data=json.dumps(body,ensure_ascii=False).encode("utf-8"),headers=H)
    try:
        r=urllib.request.urlopen(req,timeout=90); d=json.loads(r.read().decode())
        res=d.get("results") or []
        urls=[x.get("url") for x in res]
        inlist=[u in corpus_urls for u in urls if u]
        tl=[len(x.get("text") or "") for x in res]
        return dict(qid=int(row["id"]),n=len(res),n_url=len([u for u in urls if u]),
                    n_in_corpus=int(sum(inlist)),langs=[x.get("lang") for x in res],
                    dtypes=[x.get("doc_type") for x in res],srcs=[x.get("source") for x in res],
                    textlen_med=float(np.median(tl)) if tl else 0,maxlen=max(tl) if tl else 0,
                    sect=sum(len(x.get("section_types") or []) for x in res),
                    tr=d.get("translated_query"))
    except Exception as e:
        return dict(qid=int(row["id"]),err=repr(e)[:120])
random.seed(20261018)
samp=qd.sample(60,random_state=20261018)
out=[]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    for r in ex.map(one,samp.itertuples(index=False)._asdict if False else [dict(id=x.id,query=x.query) for x in samp.itertuples()]):
        out.append(r)
json.dump(out,open("medkb_query_probe.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
ok=[o for o in out if "err" not in o]
print("queries probed:",len(out),"ok:",len(ok),"errors:",len(out)-len(ok))
print("total results:",sum(o["n"] for o in ok),"with url:",sum(o["n_url"] for o in ok),
      "URLs found in corpus list:",sum(o["n_in_corpus"] for o in ok))
print("overlap rate per query (median):",np.median([o["n_in_corpus"]/max(1,o["n_url"]) for o in ok]))
from collections import Counter
L=Counter(); D=Counter(); S=Counter(); T=Counter()
for o in ok:
    L.update(o["langs"]); D.update(o["dtypes"]); S.update(o["srcs"])
print("lang distribution of MedKB top-20 results:",L.most_common())
print("doc_type distribution:",D.most_common())
print("top sources:",S.most_common(15))
print("median text len:",np.median([o["textlen_med"] for o in ok]),"| median max len:",np.median([o["maxlen"] for o in ok]))
print("sample translated queries:",[o["tr"] for o in ok[:8]])
