import pandas as pd,numpy as np,sys,re,json
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus_cat.pkl")
CJK=r"\u4e00-\u9fff"
def slug(u):
    p=re.sub(r"^[a-z]+://[^/]+","",u); p=p.split("?")[0].split("#")[0]
    p=re.sub(r"\.(html?|htm|ldo|vnp|aspx?|php)$","",p,flags=re.I); p=p.strip("/")
    p=re.sub(r"[-_/]+"," ",p); p=re.sub(r"\b\d{3,}\b"," ",p)
    p=re.sub(r"[^0-9a-zA-Z\u00C0-\u024F\u1E00-\u1EFF"+CJK+r" ]+"," ",p)
    return " ".join(p.lower().split())
c["slug"]=c.url.map(slug)
c["nslug"]=c.slug.map(lambda s: re.sub(r"[^a-z0-9"+CJK+r"]","",s))
vi=c[(c.lang=="vi")&(c.slug.str.count(" ")>=3)]
print("VI docs with usable slugs:",len(vi))
vc=vi.slug.value_counts(); print("VI dup slug values:",int((vc>1).sum()),"covering",int(vc[vc>1].sum()),"docs")
for k,v in vc.head(8).items(): print(f"   {v:>4}  {k[:100]}")
zh=c[(c.lang=="zh")&(c.nslug.str.len()>=6)]
zc=zh.nslug.value_counts(); print("\nZH docs w/ usable slugs:",len(zh),"dup:",int((zc>1).sum()),"covering",int(zc[zc>1].sum()))
sub=vi[vi.slug.str.count(" ")>=5]
g=sub.groupby("slug").host.nunique()
print("\nVI long slugs (>=6 words) on >1 host:",int((g>1).sum()),"covering",int(sub[sub.slug.isin(g[g>1].index)].shape[0]),"docs of",len(sub))
dups=sub[sub.duplicated("slug",keep=False)]; pairs=Counter()
for s,grp in dups.groupby("slug"):
    hs=sorted(set(grp.host))
    for i in range(len(hs)):
        for j in range(i+1,len(hs)): pairs[(hs[i],hs[j])]+=1
print("top host pairs sharing identical long slugs:")
for k,v in pairs.most_common(10): print(f"   {v:>4}  {k[0]} <-> {k[1]}")
MED=re.compile(r"benh|suc-khoe|thuoc|y-te|dieu-tri|trieu-chung|vac-xin|vaccine|ung-thu|huyet-ap|tieu-duong|covid|dinh-duong|kham|bac-si|benh-vien|xet-nghiem|tim|gan|phoi|da-day|than|sinh-san|nhieu-khoa")
NOISE=re.compile(r"tu-vi|xo-so|lich-thi-dau|bong-da|the-thao|chuyen-nhuong|gia-vang|ty-gia|thoi-tiet|du-bao|sao-viet|showbiz|phim-anh|nhac|lich-am|diem-chuan|tuyen-sinh")
vi2=c[c.lang=="vi"]; m=vi2.url.str.lower()
nz=m.str.contains(NOISE); md=m.str.contains(MED)
print("\nVI slug MEDICAL-ish: %d (%.1f%%) | NOISE-ish: %d (%.1f%%) | neither: %d (%.1f%%)"%(md.sum(),md.mean()*100,nz.sum(),nz.mean()*100,(~md&~nz).sum(),(~md&~nz).mean()*100))
per=vi2.assign(nz=nz,md=md).groupby("host")[["nz","md"]].mean().mul(100).round(1).join(vi2.groupby("host").size().rename("n"))
print(per.sort_values("nz",ascending=False).head(20).to_string())
print()
en=c[c.lang=="en"]
print("EN(PubMed) sample slugs:",[s for s in en.slug.head(3)])
