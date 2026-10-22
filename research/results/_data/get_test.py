import json,sys,urllib.request,urllib.error,ssl,time,gzip,io
sys.stdout.reconfigure(encoding="utf-8")
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
HDR={"User-Agent":UA,"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language":"vi-VN,vi;q=0.9,en;q=0.8,zh-CN;q=0.7","Accept-Encoding":"gzip, deflate","Connection":"keep-alive","Upgrade-Insecure-Requests":"1"}
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
tests=[
 ("nhathuoclongchau.com.vn","https://nhathuoclongchau.com.vn/bai-viet/nhung-cach-chong-lao-hoa-bang-dau-olive-50731.html"),
 ("nhathuoclongchau.com.vn","https://nhathuoclongchau.com.vn/bai-viet/viem-phoi-do-virus-gia-tang-vao-mua-lanh.html"),
 ("familydoctor.com.cn","https://www.familydoctor.com.cn/fuke/a/201707/2122841.html"),
 ("medlatec.vn","https://medlatec.vn/tin-tuc/dau-than-kinh-toa-nen-an-gi-top-5-thuc-pham-nen-bo-sung-hang-ngay-s65-n27701"),
 ("medlatec.vn","https://medlatec.vn/tin-tuc/6-thoi-quen-ngan-ngua-tram-cam-tao-dung-loi-song-tich-cuc-s195-n22534"),
 ("120ask.com","https://www.120ask.com/question/828643.htm"),
 ("cnkang.com","https://www.cnkang.com/dise/1145800.html"),
 ("ask.39.net","https://ask.39.net/question/41031471.html"),
 ("laodong.vn","https://laodong.vn/ban-tin/ha-duong-huyet-hieu-qua-nho-loai-la-quen-thuoc-1.ldo"),
 ("suckhoe"+"congdongonline.vn","https://suckhoecongdongonline.vn/tu-vi-ngay-1-1-2024-cung-hoang-dao-1-1-2024-xem-tu-vi-hom-nay-1-1-2024-d226137.html"),
 ("vinmec.com","https://www.vinmec.com/vie/bai-viet/su-phat-trien-cua-tre-1-tuan-tuoi-sau-sinh-vi"),
 ("thanhnien.vn","https://thanhnien.vn/ban-co-biet-1.htm"),
 ("youmed.vn","https://youmed.vn/tin-tuc/mang-thai-tuan-20/"),
 ("giadinhonline.vn","https://giadinhonline.vn/tu-vi-ngay-1-1-2024-cua-12-cung-hoang-dao-d198945.html"),
 ("hellobacsi.com","https://hellobacsi.com/mang-thai/thai-ky/tam-ca-nguyet-1/thai-nhi-20-tuan/"),
]
for name,url in tests:
    t=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(url,headers=HDR),timeout=25,context=ctx)
        b=r.read(400000)
        txt=""
        try:
            txt=gzip.decompress(b).decode("utf-8","replace") if r.headers.get("Content-Encoding")=="gzip" else b.decode("utf-8","replace")
        except Exception: txt=b.decode("utf-8","replace")
        title=""
        import re
        m=re.search(r"<title[^>]*>(.*?)</title>",txt,re.S|re.I)
        if m: title=" ".join(m.group(1).split())[:90]
        print(f"{name:<26} {r.status} len={len(b):>7} ct={str(r.headers.get('content-type'))[:22]:<22} title='{title}' {time.time()-t:.1f}s")
    except urllib.error.HTTPError as e:
        body=b""
        try: body=e.read(300)
        except Exception: pass
        print(f"{name:<26} HTTP {e.code} server={e.headers.get('server')} cf-mitigated={e.headers.get('cf-mitigated')} body={body[:110]!r} {time.time()-t:.1f}s")
    except Exception as e:
        print(f"{name:<26} ERR {type(e).__name__}: {str(e)[:80]} {time.time()-t:.1f}s")
    time.sleep(0.6)
