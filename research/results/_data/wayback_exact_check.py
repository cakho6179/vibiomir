import json,sys,time,urllib.request,urllib.error,urllib.parse
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
def get(u,timeout=25):
    try:
        r=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=timeout)
        return r.status, r.read(4000)
    except urllib.error.HTTPError as e:
        b=b""
        try: b=e.read(400)
        except Exception: pass
        return e.code, b
    except Exception as e:
        return None, type(e).__name__.encode()

c=pd.read_pickle("corpus_cat.pkl")
parts=[]
for h,g in c.groupby("host"):
    parts.append(g.sample(min(3,len(g)),random_state=11))
samp=pd.concat(parts)
print("sample rows:",len(samp),"hosts:",samp.host.nunique())
print("### WAYBACK exact-URL availability (3 corpus URLs per host)")
wb=[]
for _,r in samp.iterrows():
    url=r["url"]; host=r["host"]
    st,body=get("https://archive.org/wayback/available?url="+urllib.parse.quote(url,safe=""),timeout=25)
    ok=False; ts=None
    if st==200:
        try:
            j=json.loads(body.decode("utf-8","replace"))
            snap=j.get("archived_snapshots",{}).get("closest")
            if snap: ok=bool(snap.get("available")); ts=snap.get("timestamp")
        except Exception: pass
    wb.append({"host":host,"url":url,"available":bool(ok),"ts":ts,"status":st})
    print(f"  {str(host)[:26]:<27} wb={ok} ts={ts} st={st}",flush=True)
    time.sleep(0.35)
json.dump(wb,open("wayback_exact.json","w",encoding="utf-8"),indent=1,ensure_ascii=False)
av=sum(1 for x in wb if x["available"])
tot=len(wb)
print(f"WAYBACK exact-URL: {av}/{tot} archived ({av/tot*100:.1f}%)")
# per host summary
import collections
ph=collections.defaultdict(lambda:[0,0])
for x in wb: ph[x["host"]][0]+= int(x["available"]); ph[x["host"]][1]+=1
for h,(a,n) in sorted(ph.items(), key=lambda kv:-kv[1][0]): print(f"   {str(h)[:30]:<31} {a}/{n}")
