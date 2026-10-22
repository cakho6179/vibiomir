import json,sys,time,urllib.request
sys.stdout.reconfigure(encoding="utf-8")
E="https://medkb.tinix.ai/api/search"
H={"Content-Type":"application/json","X-Medical-Anti-Bot":"human-verified","Origin":"https://medkb.tinix.ai","Referer":"https://medkb.tinix.ai/search","User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
def call(body):
    req=urllib.request.Request(E,data=json.dumps(body).encode(),headers=H)
    t=time.time()
    try:
        r=urllib.request.urlopen(req,timeout=60); raw=r.read().decode(); dt=time.time()-t
        d=json.loads(raw)
        return {"status":r.status,"bytes":len(raw),"sec":round(dt,2),"total":d.get("total"),"n_results":len(d.get("results") or []),"translated":d.get("translated_query")}
    except Exception as e:
        return {"err":repr(e)[:200],"sec":round(time.time()-t,2)}
base=dict(query="viem phoi",top_k=10,offset=0,year=0,year_from=0,year_to=0,lang="",doc_type="")
tests=[
  ("default",base),
  ("top_k=50",{**base,"top_k":50}),
  ("top_k=200",{**base,"top_k":200}),
  ("offset=10",{**base,"offset":10}),
  ("offset=100",{**base,"offset":100}),
  ("empty query",{**base,"query":""}),
  ("lang=vi",{**base,"lang":"vi"}),
  ("lang=en",{**base,"lang":"en"}),
  ("lang=zh",{**base,"lang":"zh"}),
  ("doc_type=news",{**base,"doc_type":"news"}),
  ("doc_type=pubmed",{**base,"doc_type":"pubmed"}),
  ("no antibot hdr x",None),
]
for name,b in tests:
    if b is None:
        req=urllib.request.Request(E,data=json.dumps(base).encode(),headers={k:v for k,v in H.items() if k!="X-Medical-Anti-Bot"})
        try:
            r=urllib.request.urlopen(req,timeout=30); print(name,"->",r.status,len(r.read()))
        except Exception as e: print(name,"-> ERR",repr(e)[:160])
        continue
    print(name,"->",call(b))
