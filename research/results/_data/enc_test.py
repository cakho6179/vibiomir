import sys,urllib.request,ssl,gzip,re,json
sys.stdout.reconfigure(encoding="utf-8")
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
HDR={"User-Agent":UA,"Accept":"text/html,application/xhtml+xml,*/*;q=0.8","Accept-Language":"en-US,en;q=0.9","Accept-Encoding":"gzip"}
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
print("### PubMed article page raw")
r=urllib.request.urlopen(urllib.request.Request("https://pubmed.ncbi.nlm.nih.gov/35288655/",headers=HDR),timeout=30,context=ctx)
raw=r.read()
print("status",r.status,"server",r.headers.get("server"),"len",len(raw),"ct",r.headers.get("content-type"),"encoding",r.headers.get("content-encoding"))
if r.headers.get("Content-Encoding")=="gzip":
    raw=gzip.decompress(raw)
print("---- body ----")
print(raw.decode("utf-8","replace")[:600])
print()
print("### cnkang encoding test")
rn=urllib.request.urlopen(urllib.request.Request("https://www.cnkang.com/dise/5971/5971.html",headers={**HDR,"Accept-Language":"zh-CN,zh;q=0.9"},timeout=30,context=ctx),timeout=30)
raw2=rn.read()
print("status",rn.status,"ct",rn.headers.get("content-type"),"len",len(raw2))
m=re.search(rb'charset=["\']?([\w-]+)',raw2[:3000],re.I)
print("declared charset:",m.group(1) if m else None)
for enc in ["gb18030","utf-8","gbk"]:
    try:
        t=re.sub(r"<[^>]+>"," ",raw2.decode(enc,"replace"))
        t=" ".join(t.split())
        print(f"  [{enc}] first 180 chars: {t[100:280]}")
    except Exception as e: print(f"  [{enc}] fail {e}")
