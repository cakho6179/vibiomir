import json,sys,threading,urllib.request,urllib.error,collections
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
class NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**k): return None
OP=urllib.request.build_opener(NoRedir)
def probe(u,timeout=15):
    try:
        r=OP.open(urllib.request.Request(u,headers=UA),timeout=timeout)
        return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, (e.headers.get("Location") if 300<=e.code<400 else None)
    except Exception as e:
        return None, type(e).__name__
c=pd.read_pickle("corpus_cat.pkl")
# focus on hosts known to answer GET (skip Cloudflare/ESA-blocked to avoid noise)
hosts=["suckhoecongdongonline.vn","suckhoedoisong.vn","thanhnien.vn","vinmec.com","medlatec.vn","giadinhonline.vn","phunusuckhoe.giadinhonline.vn","baonghean.vn","vietnamnet.vn","hellobacsi.com","youmed.vn","vnexpress.net","suckhoeviet.org.vn","cnkang.com","ask.39.net","laodong.vn","tienphong.vn","dantri.com.vn","tuoitre.vn","sggp.org.vn"]
parts=[]
for h in hosts:
    g=c[c.host==h]
    if len(g)>=8: parts.append(g.sample(8,random_state=7))
samp=pd.concat(parts); todo=list(samp.itertuples(index=False))
print("redirect probe sample:",len(todo),"urls across",samp.host.nunique(),"hosts",flush=True)
res=[];lock=threading.Lock()
def work(chunk):
    out=[]
    for row in chunk:
        st,loc=probe(row.url)
        out.append({"host":row.host,"url":row.url,"status":st,"location":loc})
    with lock: res.extend(out)
chunks=[todo[i::6] for i in range(6)]
ths=[threading.Thread(target=work,args=(ch,)) for ch in chunks]
[t.start() for t in ths]; [t.join() for t in ths]
json.dump(res,open("redirect_probe.json","w",encoding="utf-8"),indent=1,ensure_ascii=False)
cnt=collections.Counter((x["status"] if x["status"] else "ERR") for x in res)
print("status distribution:",dict(sorted(cnt.items(),key=lambda kv:-kv[1])))
red=[x for x in res if x["status"] and 300<=x["status"]<400]
print(f"REDIRECTS: {len(red)}/{len(res)} ({len(red)/len(res)*100:.1f}%)")
for x in red[:12]: print("   ",x["status"],x["host"],"->",(x["location"] or "")[:90])
byhost=collections.defaultdict(collections.Counter)
for x in res: byhost[x["host"]][x["status"] if x["status"] else "ERR"]+=1
print("--- per host ---")
for h in hosts:
    if h in byhost: print(f"   {h[:30]:<31}",dict(byhost[h]))
print("DONE")
