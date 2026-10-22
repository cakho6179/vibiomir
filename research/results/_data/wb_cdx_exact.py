import json,sys,time,urllib.request,urllib.error,urllib.parse,collections
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
def cdx(url,timeout=30):
    q="https://web.archive.org/cdx/search/cdx?url="+urllib.parse.quote(url,safe="")+"&output=json&limit=2&fl=timestamp,statuscode,digest"
    try:
        r=urllib.request.urlopen(urllib.request.Request(q,headers=UA),timeout=timeout)
        b=r.read(4000).decode("utf-8","replace")
        if not b.strip() or b.strip()=="[]": return 200,0,None
        j=json.loads(b)
        return 200,len(j)-1,(j[1][0] if len(j)>1 else None)
    except urllib.error.HTTPError as e:
        return e.code,None,None
    except Exception as e:
        return None,None,type(e).__name__
c=pd.read_pickle("corpus_cat.pkl")
parts=[]
for h,g in c.groupby("host"):
    parts.append(g.sample(min(3,len(g)),random_state=11))
samp=pd.concat(parts)
print("sample:",len(samp),"urls across",samp.host.nunique(),"hosts")
res=[]
for _,r in samp.iterrows():
    st,n,ts=cdx(r["url"])
    res.append({"host":r["host"],"url":r["url"],"status":st,"caps":n,"first_ts":ts})
    time.sleep(0.3)
json.dump(res,open("wayback_cdx_exact.json","w",encoding="utf-8"),indent=1,ensure_ascii=False)
ok=[x for x in res if isinstance(x["caps"],int) and x["caps"]>0]
err=[x for x in res if x["status"]!=200]
print(f"EXACT-URL ARCHIVED: {len(ok)}/{len(res)} ({len(ok)/len(res)*100:.1f}%)   non-200: {len(err)}")
ph=collections.defaultdict(lambda:[0,0])
for x in res:
    ph[x["host"]][1]+=1
    if isinstance(x["caps"],int) and x["caps"]>0: ph[x["host"]][0]+=1
for h,(a,n) in sorted(ph.items(),key=lambda kv:-kv[1][0]):
    if a>0: print(f"   {str(h)[:32]:<33} {a}/{n}")
print("--- hosts with 0 exact captures (of 3) ---")
print(", ".join(sorted(str(h) for h,(a,n) in ph.items() if a==0)))
