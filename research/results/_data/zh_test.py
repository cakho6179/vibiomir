import pandas as pd,urllib.request,ssl,gzip,re,sys,random
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus_cat.pkl")
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url,al="zh-CN,zh;q=0.9"):
    h={"User-Agent":UA,"Accept":"text/html,*/*;q=0.8","Accept-Language":al,"Accept-Encoding":"gzip"}
    r=urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=30,context=ctx); raw=r.read()
    if r.headers.get("Content-Encoding")=="gzip":
        try: raw=gzip.decompress(raw)
        except Exception: pass
    return r,raw
for h in ["cnkang.com","ask.39.net"]:
    sub=c[c.host==h]
    print("="*80); print("HOST",h,"n",len(sub))
    for url in sub.sample(4,random_state=3).url:
        try:
            r,raw=get(url)
            m=re.search(rb'charset=["\']?([\w-]+)',raw[:4000],re.I); dec=m.group(1).decode() if m else "?"
            if dec.lower() in ("gb2312","gbk"): dec="gb18030"
            if dec=="?": dec="gb18030"
            t=" ".join(re.sub(r"\s+"," ",re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>"," ",raw.decode(dec,"replace"))).split())
            tt=re.sub(r"<[^>]+>"," ",raw.decode(dec,"replace"))
            tt=" ".join(tt.split())
            print(f"  {r.status} len={len(raw):>7} decl={dec:<8} title~{(re.search(r'<title[^>]*>(.*?)</title>',t,re.S).group(1)[:70] if re.search(r'<title[^>]*>(.*?)</title>',t,re.S) else '')!r}")
            print(f"      text: {tt[:200]}")
        except Exception as e:
            print(f"  ERR {type(e).__name__} {str(e)[:70]} for {url[:110]}")
