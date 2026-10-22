import sys,urllib.request,ssl,gzip,re
sys.stdout.reconfigure(encoding="utf-8")
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url,accept_lang="zh-CN,zh;q=0.9"):
    h={"User-Agent":UA,"Accept":"text/html,*/*;q=0.8","Accept-Language":accept_lang,"Accept-Encoding":"gzip"}
    r=urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=30,context=ctx)
    raw=r.read()
    if r.headers.get("Content-Encoding")=="gzip":
        try: raw=gzip.decompress(raw)
        except Exception: pass
    return r,raw
for name,url in [("cnkang","https://www.cnkang.com/dise/5971/5971.html"),
                 ("ask39","https://ask.39.net/question/41031471.html"),
                 ("familydoc","https://www.familydoctor.com.cn/fuke/a/201707/2122841.html"),
                 ("120ask","https://www.120ask.com/question/828643.htm")]:
    try:
        r,raw=get(url)
        m=re.search(rb'charset=["\']?([\w-]+)',raw[:4000],re.I)
        dec=m.group(1).decode() if m else "?"
        print(f"== {name} status={r.status} ct={r.headers.get('content-type')} len={len(raw)} declared={dec}")
        for enc in [dec,"gb18030","utf-8"]:
            if enc=="?": continue
            try:
                t=" ".join(re.sub(r"<[^>]+>"," ",raw.decode(enc,"replace")).split())
                print(f"   [{enc}] {t[:220]}")
                break
            except Exception as e: print("   fail",enc,e)
    except Exception as e:
        print(f"== {name} ERR {type(e).__name__} {str(e)[:90]}")
