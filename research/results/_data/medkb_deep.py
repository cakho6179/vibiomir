import json,sys,urllib.request,time
sys.stdout.reconfigure(encoding="utf-8")
E="https://medkb.tinix.ai/api/search"
H={"Content-Type":"application/json","X-Medical-Anti-Bot":"human-verified","Origin":"https://medkb.tinix.ai","Referer":"https://medkb.tinix.ai/search","User-Agent":"Mozilla/5.0 Chrome/126"}
for attempt in range(6):
    try:
        body=json.dumps({"query":"Em bị viêm xung huyết 1/3 dưới thực quản, viêm dạ dày trợt xung huyết mạn tính","top_k":3,"offset":0,"year":0,"year_from":0,"year_to":0,"lang":"","doc_type":""},ensure_ascii=False).encode()
        r=urllib.request.urlopen(urllib.request.Request(E,data=body,headers=H),timeout=90)
        d=json.loads(r.read().decode()); break
    except Exception as e:
        print("attempt",attempt,repr(e)[:80]); time.sleep(25)
else: raise SystemExit("fail")
json.dump(d,open("medkb_raw_probe.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("keys per result:",list(d["results"][0].keys()))
for i,x in enumerate(d["results"]):
    t=x.get("text") or ""
    print("="*90)
    print(i,"url:",x.get("url"))
    print("   id:",x.get("id"),"| source_id:",x.get("source_id"),"| doc_type:",x.get("doc_type"),"| lang:",x.get("lang"),"| source:",x.get("source"),"| year:",x.get("year"))
    print("   title:",(x.get("title") or "")[:110])
    print("   section_types:",x.get("section_types"),"| credibility:",x.get("credibility"),"| score:",x.get("score"))
    print("   text len:",len(t),"| head:",repr(t[:150]))
    print("   tail:",repr(t[-220:]))
    print("   similar_queries:",(x.get("similar_queries") or [])[:5])
