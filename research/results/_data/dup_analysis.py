import pandas as pd,numpy as np,sys,re,unicodedata,json
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus_cat.pkl")
def slug(u):
    p=re.sub(r"^[a-z]+://[^/]+","",u); p=p.split("?")[0].split("#")[0]
    p=re.sub(r"\.(html?|htm|ldo|vnp|aspx?|php)$","",p,flags=re.I)
    p=p.strip("/")
    p=re.sub(r"[-_/]+"," ",p)
    p=re.sub(r"\b\d{3,}\b"," ",p)
    p=re.sub(r"[^0-9a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\u4e00-\u9fff ]+"," ",p)
    return " ".join(p.lower().split())
c["slug"]=c.url.map(slug)
c["nslug"]=c.slug.str.replace(r"[^a-z0-9\u4e00-\u9fff]","",regex=True)
# VI: only consider slugs with >=4 token-units
vi=c[(c.lang=="vi")&(c.slug.str.count(" ")>=3)]
print("VI docs with usable slugs:",len(vi))
vc=vi.slug.value_counts()
print("VI duplicate slug values (across corpus):",(vc>1).sum(),"slugs covering",vc[vc>1].sum(),"docs")
print("top duplicated VI slugs:")
for k,v in vc.head(10).items(): print(f"   {v:>4}  {k[:100]}")
print()
zh=c[(c.lang=="zh")&(c.nslug.str.len()>=6)]
zc=zh.nslug.value_counts(); print("ZH docs with usable slugs:",len(zh),"| dup slug values:",(zc>1).sum(),"covering",zc[zc>1].sum())
for k,v in zc.head(8).items(): print(f"   {v:>4}  {k[:60]}")
print()
# cross-host identical slug text (different hosts, >=6 words)
g=vi[vi.slug.str.count(" ")>=5].groupby("slug").host.nunique()
print("VI slugs (>=6 words) appearing on >1 distinct host:",(g>1).sum(),"covering",vi[vi.slug.isin(g[g>1].index)].shape[0],"docs")
# host pair co-occurrence
sub=vi[vi.slug.str.count(" ")>=5]
dups=sub[sub.duplicated("slug",keep=False)]
pairs=Counter()
for s,grp in dups.groupby("slug"):
    hs=sorted(set(grp.host))
    for i in range(len(hs)):
        for j in range(i+1,len(hs)): pairs[(hs[i],hs[j])]+=1
print("top host pairs sharing identical long slugs:")
for k,v in pairs.most_common(12): print(f"   {v:>4}  {k[0]}  <->  {k[1]}")
print()
# split context: how much VI content is NON-medical (heuristic keyword)
MED=re.compile(r"benh|suc-khoe|suc-khoe|thuoc|y-te|y-te|dieu-tri|trieu-chung|vaccine|vac-xin|ung-thu|tim|gan|than|phoi|dai|huyet-ap|tieu-duong|covid|dinh-duong|kham|bac-si|benh-vien|xet-nghiem")
NOISE=re.compile(r"tu-vi|xo-so|ket-qua-xo-so|lich-thi-dau|bong-da|the-thao|chuyen-nhuong|gia-vang|ty-gia|thoi-tiet|du-bao|sao-viet|showbiz|phim|nhac|nhan-vat|lich-am")
vi2=c[c.lang=="vi"]
m=vi2.url.str.lower()
print("VI URLs matching NOISE-ish slug patterns: %d (%.2f%%)"%(m.str.contains(NOISE).sum(),m.str.contains(NOISE).mean()*100))
print("VI URLs matching MEDICAL-ish slug patterns: %d (%.2f%%)"%(m.str.contains(MED).sum(),m.str.contains(MED).mean()*100))
print("VI URLs matching neither: %d (%.2f%%)"%((~m.str.contains(MED)&~m.str.contains(NOISE)).sum(),(~m.str.contains(MED)&~m.str.contains(NOISE)).mean()*100))
print()
per=vi2.assign(nz=m.str.contains(NOISE),md=m.str.contains(MED)).groupby("host")[["nz","md"]].mean().mul(100).round(1).join(vi2.groupby("host").size().rename("n"))
print("per-host noise/medical slug share (%):")
print(per.sort_values("nz",ascending=False).head(22).to_string())
