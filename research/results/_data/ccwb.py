import json,sys,time,random,urllib.request,urllib.error,urllib.parse
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (compatible; IR-research/1.0)"}
def try_get(u,timeout=90,tries=3):
    for i in range(tries):
        try:
            r=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=timeout)
            return r.status,r.read(4000).decode("utf-8","replace")
        except urllib.error.HTTPError as e:
            if e.code in (504,503,502,500) and i<tries-1: time.sleep(3); continue
            b=""
            try: b=e.read(300).decode("utf-8","replace")
            except Exception: pass
            return e.code,b
        except Exception as e:
            if i<tries-1: time.sleep(2); continue
            return None,type(e).__name__
    return None,"exhausted"
print("### COMMON CRAWL index coverage (retry)")
ccres={}
for h in ["pubmed.ncbi.nlm.nih.gov","suckhoedoisong.vn","nhathuoclongchau.com.vn","vinmec.com","medlatec.vn","thanhnien.vn","120ask.com","cnkang.com","ask.39.net","familydoctor.com.cn","laodong.vn","suckhoecongdongonline.vn"]:
    row={}
    for cc in ["CC-MAIN-2026-39","CC-MAIN-2026-34"]:
        st,body=try_get(f"https://index.commoncrawl.org/{cc}-index?url={urllib.parse.quote(h)}%2F*&output=json&limit=1")
        row[cc]=("HIT" if (st==200 and body.strip().startswith("{")) else f"{st}")
        time.sleep(1)
    ccres[h]=row; print(h,row,flush=True)
json.dump(ccres,open("cc_coverage.json","w"),indent=1)
print()
print("### WAYBACK exact-URL availability for 45 sampled corpus URLs")
c=pd.read_pickle("corpus_cat.pkl")
random.seed(11)
samp=c.groupby("host",group_keys=False).apply(lambda d: d.sample(min(3,len(d)),random_state=11)).head(45)
wbres=[]
for _,r in samp.iterrows():
    u=r.url
    st,body=try_get("https://archive.org/wayback/available?url="+urllib.parse.quote(u,safe=""),timeout=40,tries=2)
    ok=False; ts=None
    if st==200:
        try:
            j=json.loads(body); snap=j.get("archived_snapshots",{}).get("closest")
            if snap: ok=bool(snap.get("available")); ts=snap.get("timestamp")
        except Exception: pass
    wbres.append({"host":r.host,"url":u,"available":ok,"ts":ts,"status":st})
    print(f"  {r.host[:24]:<25} wb={ok} ts={ts} st={st}",flush=True)
    time.sleep(0.5)
json.dump(wbres,open("wayback_exact.json","w"),indent=1,ensure_ascii=False)
avail=sum(1 for x in wbres if x["available"])
print(f"WAYBACK: {avail}/{len(wbres)} exact corpus URLs archived ({avail/len(wbres)*100:.1f}%)")
print("DONE")
