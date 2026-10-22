import pandas as pd,numpy as np,sys,json,time,random,urllib.request,urllib.error,ssl,gzip,re,socket
sys.stdout.reconfigure(encoding="utf-8")
socket.setdefaulttimeout(25)
c=pd.read_pickle("corpus.pkl"); u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
HDR={"User-Agent":UA,"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language":"vi-VN,vi;q=0.9,en;q=0.8,zh-CN;q=0.7","Accept-Encoding":"gzip","Connection":"keep-alive","Upgrade-Insecure-Requests":"1"}
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
hosts=["pubmed.ncbi.nlm.nih.gov","suckhoecongdongonline.vn","120ask.com","suckhoedoisong.vn","nhathuoclongchau.com.vn","thanhnien.vn","cnkang.com","ask.39.net","laodong.vn","phunusuckhoe.giadinhonline.vn","familydoctor.com.cn","vinmec.com","medlatec.vn","giadinhonline.vn","baonghean.vn","vietnamnet.vn","hellobacsi.com","youmed.vn","vnexpress.net","suckhoeviet.org.vn"]
def text_of(html):
    h=re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>"," ",html)
    h=re.sub(r"(?s)<!--.*?-->"," ",h)
    t=" ".join(re.sub(r"<[^>]+>"," ",h).split())
    return t
out=[]
for h in hosts:
    sub=c[c.host==h]
    if len(sub)==0: continue
    for url in sub.sample(min(6,len(sub)),random_state=42).url:
        t=time.time(); rec={"host":h,"url":url}
        try:
            r=urllib.request.urlopen(urllib.request.Request(url,headers=HDR),timeout=25,context=ctx)
            raw=r.read(3000000)
            if r.headers.get("Content-Encoding")=="gzip":
                try: raw=gzip.decompress(raw)
                except Exception: pass
            html=raw.decode("utf-8","replace")
            tx=text_of(html)
            rec.update(status=r.status,bytes=len(raw),final=r.geturl()[:200],redir=int(r.geturl()!=url),
                       title=(" ".join(re.search(r"<title[^>]*>(.*?)</title>",html,re.S|re.I).group(1).split())[:100] if re.search(r"<title[^>]*>(.*?)</title>",html,re.S|re.I) else ""),
                       text_chars=len(tx),text_words=len(tx.split()))
        except urllib.error.HTTPError as e: rec.update(status=e.code,err="HTTP")
        except Exception as e: rec.update(status=None,err=type(e).__name__)
        rec["sec"]=round(time.time()-t,1); out.append(rec)
        print(f"{h[:22]:<23} {rec.get('status')} chars={rec.get('text_chars',0):>7} words={rec.get('text_words',0):>6} {rec['sec']}s {str(rec.get('title',''))[:60]}",flush=True)
        time.sleep(0.4)
json.dump(out,open("liveness_get.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("DONE")
