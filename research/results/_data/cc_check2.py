import json,sys,time,urllib.request,urllib.error,urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
UA={"User-Agent":"Mozilla/5.0 (compatible; IR-research/1.0)"}
hosts=["pubmed.ncbi.nlm.nih.gov","suckhoedoisong.vn","nhathuoclongchau.com.vn","vinmec.com","medlatec.vn","thanhnien.vn","laodong.vn","suckhoecongdongonline.vn","120ask.com","cnkang.com","ask.39.net","familydoctor.com.cn","vietnamnet.vn","tuoitre.vn","vnexpress.net","hellobacsi.com","youmed.vn","giadinhonline.vn","phunusuckhoe.giadinhonline.vn","baonghean.vn"]
res={}
for h in hosts:
    row={"host":h}
    for cc in ["CC-MAIN-2026-39","CC-MAIN-2026-34","CC-MAIN-2026-21","CC-MAIN-2025-30"]:
        q=f"https://index.commoncrawl.org/{cc}-index?url={urllib.parse.quote(h)}%2F*&output=json&limit=1"
        try:
            r=urllib.request.urlopen(urllib.request.Request(q,headers=UA),timeout=40)
            body=r.read(3000).decode("utf-8","replace")
            row[cc]="HIT" if body.strip().startswith("{") else "?"
        except urllib.error.HTTPError as e:
            body=""
            try: body=e.read(300).decode("utf-8","replace")
            except Exception: pass
            row[cc]=f"{e.code}" + ("(no-captures)" if e.code==404 else "")
        except Exception as e: row[cc]="err"
        time.sleep(0.25)
    res[h]=row; print(h,row,flush=True)
json.dump(res,open("cc_coverage.json","w",encoding="utf-8"),indent=1)
print("DONE")
