import pandas as pd,numpy as np,sys,json,urllib.request,time,collections,random
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus.pkl"); u=c.url.astype(str)
pm=c[u.str.contains("pubmed.ncbi.nlm.nih.gov",regex=False)]
pmid=pm.url.str.extract(r"/(\d+)/")[0].astype(np.int64).values
random.seed(1); samp=random.sample(list(pmid),600)
years=[]; jn=[]
for i in range(0,600,200):
    ids=",".join(map(str,samp[i:i+200]))
    url=f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={ids}"
    try:
        d=json.loads(urllib.request.urlopen(url,timeout=60).read().decode())
        r=d.get("result",{})
        for k,v in r.items():
            if k=="uids": continue
            years.append(v.get("pubdate","")[:4]); jn.append(v.get("source"))
    except Exception as e: print("err",repr(e)[:100])
    time.sleep(0.5)
print("fetched",len(years))
yc=collections.Counter([y for y in years if y.isdigit()])
print("year distribution (sample of 600 PMIDs):")
for y in sorted(yc): print("   ",y,yc[y])
print("journal top:",collections.Counter(jn).most_common(12))
json.dump({"years":years,"journals":jn},open("pubmed_meta_sample.json","w"))
# also get max/min year
