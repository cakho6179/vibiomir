import json,sys,time,urllib.request,urllib.parse,random,re
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (compatible; IR-research/1.0)"}
def get(u,t=45):
    return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t).read().decode("utf-8","replace")
hosts=["pubmed.ncbi.nlm.nih.gov","suckhoedoisong.vn","nhathuoclongchau.com.vn","vinmec.com","medlatec.vn","thanhnien.vn","laodong.vn","suckhoecongdongonline.vn","120ask.com","cnkang.com","ask.39.net","familydoctor.com.cn","vietnamnet.vn","tuoitre.vn","vnexpress.net","hellobacsi.com","youmed.vn","giadinhonline.vn"]
print("=== COMMON CRAWL coverage (CC-MAIN-2026-39 + 2026-34), per host ===")
for h in hosts:
    row=[h]
    for cc in ["CC-MAIN-2026-39","CC-MAIN-2026-34"]:
        q=f"https://index.commoncrawl.org/{cc}-index?url={urllib.parse.quote(h)}%2F*&output=json&limit=1"
        try:
            r=get(q,30)
            row.append("yes" if r.strip().startswith("{") else "no-records")
        except Exception as e: row.append("err:"+type(e).__name__)
    print("   "," | ".join(row))
    time.sleep(0.3)
