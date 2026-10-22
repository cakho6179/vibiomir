import json,sys,time,threading,urllib.request,urllib.error,urllib.parse,collections,queue
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
def cdx(url,timeout=25):
    q="https://web.archive.org/cdx/search/cdx?url="+urllib.parse.quote(url,safe="")+"&output=json&limit=2&fl=timestamp,statuscode"
    try:
        r=urllib.request.urlopen(urllib.request.Request(q,headers=UA),timeout=timeout)
        b=r.read(4000).decode("utf-8","replace")
        if not b.strip() or b.strip()=="[]": return 200,0,None
        j=json.loads(b); return 200,len(j)-1,(j[1][0] if len(j)>1 else None)
    except urllib.error.HTTPError as e: return e.code,None,None
    except Exception as e: return None,None,type(e).__name__
c=pd.read_pickle("corpus_cat.pkl")
top=c.host.value_counts().head(10).index.tolist()
parts=[]
for h,g in c.groupby("host"):
    n=5 if h in top else 1
    parts.append(g.sample(min(n,len(g)),random_state=11))
samp=pd.concat(parts)
print("sample:",len(samp),"urls across",samp.host.nunique(),"hosts (5/top10, 1/each other)",flush=True)
todo=list(samp.itertuples(index=False)); res=[]; lock=threading.Lock()
def work(chunk):
    out=[]
    for row in chunk:
        u=row.url
        st,n,ts=cdx(u)
        out.append({"host":row.host,"url":u,"status":st,"caps":n,"first_ts":ts})
    with lock: res.extend(out)
chunks=[todo[i::6] for i in range(6)]
ths=[threading.Thread(target=work,args=(ch,)) for ch in chunks]
[t.start() for t in ths]; [t.join() for t in ths]
json.dump(res,open("wayback_cdx_exact.json","w",encoding="utf-8"),indent=1,ensure_ascii=False)
ok=[x for x in res if isinstance(x["caps"],int) and x["caps"]>0]
err=[x for x in res if x["status"]!=200]
print(f"EXACT-URL ARCHIVED: {len(ok)}/{len(res)} ({len(ok)/len(res)*100:.1f}%) non-200:{len(err)}")
ph=collections.defaultdict(lambda:[0,0])
for x in res:
    ph[x["host"]][1]+=1
    if isinstance(x["caps"],int) and x["caps"]>0: ph[x["host"]][0]+=1
for h,(a,n) in sorted(ph.items(),key=lambda kv:-kv[1][0]):
    if a>0: print(f"   {str(h)[:34]:<35} {a}/{n}")
print("--- 0-cap hosts ---")
print(", ".join(sorted(str(h) for h,(a,n) in ph.items() if a==0)))
print("DONE")
