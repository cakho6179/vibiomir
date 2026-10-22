import json,sys,urllib.request,time
sys.stdout.reconfigure(encoding="utf-8")
E="https://medkb.tinix.ai/api/search"
H={"Content-Type":"application/json","Origin":"https://medkb.tinix.ai","Referer":"https://medkb.tinix.ai/search","User-Agent":"Mozilla/5.0 Chrome/126"}
body=json.dumps({"query":"Em bị viêm xung huyết 1/3 dưới thực quản, viêm dạ dày trợt xung huyết mạn tính","top_k":3,"offset":0,"year":0,"year_from":0,"year_to":0,"lang":"","doc_type":""},ensure_ascii=False).encode()
for i in range(3):
    try:
        r=urllib.request.urlopen(urllib.request.Request(E,data=body,headers=H),timeout=120)
        d=json.loads(r.read().decode())
        json.dump(d,open("medkb_raw_probe.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
        print("OK status",r.status,"n",len(d["results"]))
        for x in d["results"]:
            t=x.get("text") or ""
            print("-"*80)
            print("url:",x.get("url"))
            print("keys:",sorted(x.keys()))
            print("id:",x.get("id"),"| source_id:",x.get("source_id"),"| doc_type:",x.get("doc_type"),"| lang:",x.get("lang"),"| source:",x.get("source"),"| year:",x.get("year"),"| credibility:",x.get("credibility"))
            print("title:",(x.get("title") or "")[:100])
            print("section_types:",x.get("section_types"),"| text len:",len(t))
            print("head:",repr(t[:180]))
            print("tail:",repr(t[-200:]))
            print("similar_queries:",(x.get("similar_queries") or [])[:6])
        break
    except Exception as e:
        print("attempt",i,repr(e)[:110]); time.sleep(20)
