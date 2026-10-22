import pandas as pd, numpy as np, sys, json, time, random, urllib.request, urllib.error, ssl
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
hosts=["pubmed.ncbi.nlm.nih.gov","suckhoecongdongonline.vn","120ask.com","suckhoedoisong.vn","nhathuoclongchau.com.vn","thanhnien.vn","cnkang.com","ask.39.net","laodong.vn","vinmec.com","medlatec.vn","familydoctor.com.cn","hellobacsi.com","youmed.vn","giadinhonline.vn"]
random.seed(7)
res=[]
for h in hosts:
    sub=c[c.host.str.endswith(h)]
    n=min(10,len(sub))
    for url in sub.sample(n,random_state=hash(h)%10000).url:
        t=time.time(); rec={"host":h,"url":url}
        try:
            req=urllib.request.Request(url,method="HEAD",headers={"User-Agent":UA,"Accept":"text/html,*/*"})
            r=urllib.request.urlopen(req,timeout=20,context=ctx)
            rec.update(status=r.status,final=r.geturl()[:180],ctype=r.headers.get("content-type"),clen=r.headers.get("content-length"),redir=int(r.geturl()!=url))
        except urllib.error.HTTPError as e: rec.update(status=e.code,err="HTTP")
        except Exception as e: rec.update(status=None,err=type(e).__name__+":"+str(e)[:60])
        rec["sec"]=round(time.time()-t,2); res.append(rec)
        time.sleep(0.7)
json.dump(res,open("liveness_head.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
df=pd.DataFrame(res)
print(df.groupby("host").status.apply(lambda s: dict(s.value_counts())).to_string())
print()
print("STATUS TOTAL:",df.status.value_counts(dropna=False).to_dict())
print("redirects to different URL:",int(df.redir.fillna(0).sum()),"/",len(df))
print("errors:",df[df.status.isna()].err.value_counts().to_dict())
print("http->https upgrades:", sum(1 for _,r in df.iterrows() if r.status and r.get("final") and r["url"].startswith("http://") and str(r["final"]).startswith("https://")))
print("median latency",df.sec.median())
