import json,sys,urllib.request,time
sys.stdout.reconfigure(encoding="utf-8")
E="https://medkb.tinix.ai/api/search"
H={"Content-Type":"application/json","X-Medical-Anti-Bot":"human-verified","Origin":"https://medkb.tinix.ai","Referer":"https://medkb.tinix.ai/search","User-Agent":"Mozilla/5.0 Chrome/126"}
body=json.dumps({"query":"benh tieu duong","top_k":5,"offset":0,"year":0,"year_from":0,"year_to":0,"lang":"","doc_type":""}).encode()
# rapid-fire test to find rate limit
for i in range(12):
    t=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(E,data=body,headers=H),timeout=60)
        d=json.loads(r.read().decode())
        print(i,"status",r.status,"n",len(d.get("results") or []),"%.1fs"%(time.time()-t))
    except urllib.error.HTTPError as e:
        print(i,"HTTPError",e.code,e.reason,"%.1fs"%(time.time()-t), e.read()[:200])
    except Exception as e:
        print(i,"ERR",repr(e)[:120],"%.1fs"%(time.time()-t))
