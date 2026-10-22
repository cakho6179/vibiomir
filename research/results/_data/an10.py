import pandas as pd,numpy as np,sys,json,re
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
c=pd.read_pickle("corpus.pkl"); u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
ZH_EXTRA=set(h for h in c.host.unique() if h.endswith(".39.net"))
ZH=set(["120ask.com","cnkang.com","familydoctor.com.cn","ask.39.net","youlai.cn","zhongyibaodian.net","zysjonline.com","msdmanuals.cn","a-hospital.com","wujue.com","zydcd.com","article.iiyi.com","jb39.com","jbk.39.net","test.pmphai.com","baidu.com","familydoctor.cn","qihuangzhishu.com","health.people.com.cn","qy.familydoctor.com.cn","ask.familydoctor.com.cn","bingli.iiyi.com","baidianfeng.familydoctor.com.cn"]) | ZH_EXTRA
c["lang"]=np.where(c.host=="pubmed.ncbi.nlm.nih.gov","en",np.where(c.host.isin(ZH),"zh","vi"))
CAT={
 "pubmed.ncbi.nlm.nih.gov":"academic (PubMed)",
 "bachmai.gov.vn":"hospital","benhviennhitrunguong.gov.vn":"hospital","benhvienvietduc.org":"hospital",
 "tamanhhospital.vn":"hospital","vinmec.com":"hospital","medlatec.vn":"hospital",
 "nhathuoclongchau.com.vn":"pharmacy/health portal","tiemchunglongchau.com.vn":"pharmacy/vaccine",
 "pharmacity.vn":"pharmacy","pmc-ecm-healthblog.beta.pharmacity.io":"pharmacy blog",
 "hellobacsi.com":"health portal","youmed.vn":"health portal","thaythuocvietnam.vn":"drug reference",
 "suckhoeviet.org.vn":"health portal","khoahocphothong.vn":"science news",
 "msdmanuals.cn":"clinical manual","a-hospital.com":"medical encyclopedia",
 "zysjonline.com":"TCM portal","zhongyibaodian.net":"TCM portal","zydcd.com":"TCM portal","wujue.com":"TCM portal","qihuangzhishu.com":"TCM portal",
 "120ask.com":"medical Q&A","ask.39.net":"medical Q&A","jbk.39.net":"medical Q&A","jb39.com":"medical Q&A","baby.39.net":"health portal","byby.39.net":"health portal","cancer.39.net":"health portal","care.39.net":"health portal","ek.39.net":"health portal","fitness.39.net":"health portal","fk.39.net":"health portal","food.39.net":"health portal","gan.39.net":"health portal","gc.39.net":"health portal","gk.39.net":"health portal","heart.39.net":"health portal","man.39.net":"health portal","nk.39.net":"health portal","pf.39.net":"health portal","shen.39.net":"health portal","sj.39.net":"health portal","tnb.39.net":"health portal","wei.39.net":"health portal","woman.39.net":"health portal",
 "cnkang.com":"health portal","familydoctor.com.cn":"health portal","familydoctor.cn":"health portal","baidianfeng.familydoctor.com.cn":"health portal","qy.familydoctor.com.cn":"health portal","ask.familydoctor.com.cn":"health portal",
 "youlai.cn":"medical Q&A","test.pmphai.com":"medical textbook/quiz","article.iiyi.com":"medical Q&A","bingli.iiyi.com":"medical Q&A","health.people.com.cn":"news",
 "baidu.com":"encyclopedia",
}
def cat(h):
    if h in CAT: return CAT[h]
    if h.endswith(".vn") or h.endswith(".io") or h.endswith(".org"): return "news/media"
    return "other"
c["cat"]=c.host.map(cat)
print("=== CORPUS BY LANGUAGE AND CATEGORY ===")
t=c.groupby(["lang","cat"]).size().rename("docs").reset_index().sort_values(["lang","docs"],ascending=[True,False])
print(t.to_string(index=False))
print()
print("=== VI host blocks: id contiguity proof (block boundaries) ===")
vi=c[c.lang=="vi"].sort_values("id")
g=vi.groupby("host").id.agg(["min","max","count"])
g["span"]=g["max"]-g["min"]+1; g["contiguous"]=g["span"]==g["count"]
print("VI hosts fully contiguous:",int(g.contiguous.sum()),"/",len(g))
print(g[~g.contiguous].to_string())
print()
print("=== ID SPAN SUMMARY ===")
for lg in ["vi","en","zh"]:
    s=np.sort(c[c.lang==lg].id.values); print(f"  {lg}: ids {s[0]}..{s[-1]}  n={len(s)}  span={s[-1]-s[0]+1}  density={len(s)/(s[-1]-s[0]+1):.3f}")
print("  overall span 583..7466161 =",7466161-583+1,"| exposed",len(c),"| implied master size >=1466161".replace("1466161","7466161"))
c.to_pickle("corpus_cat.pkl")
