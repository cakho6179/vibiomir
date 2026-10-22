import pandas as pd, sys, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",250); pd.set_option("display.max_rows",300)
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
# language classification
ZH_HOSTS=set(["120ask.com","cnkang.com","familydoctor.com.cn","ask.39.net","youlai.cn","zhongyibaodian.net","zysjonline.com","msdmanuals.cn","a-hospital.com","wujue.com","zydcd.com","article.iiyi.com","jb39.com","jbk.39.net","test.pmphai.com","baidu.com","familydoctor.cn","qihuangzhishu.com","health.people.com.cn","qy.familydoctor.com.cn","ask.familydoctor.com.cn","bingli.iiyi.com","baidianfeng.familydoctor.com.cn"])
c["lang"]=np.where(c.host=="pubmed.ncbi.nlm.nih.gov","en",np.where(c.host.isin(ZH_HOSTS),"zh","vi"))
print("=== LANG SPLIT (by URL, vs README claim VI 740598 / EN 240000 / ZH 240000) ===")
print(c.lang.value_counts().to_string())
print()
# check the .39.net subhosts not in ZH list
mask=c.host.str.endswith(".39.net") & ~c.host.isin(ZH_HOSTS)
print("unclassified .39.net hosts:",c[mask].host.unique())
print()
# contiguous runs
s=np.sort(c.id.values)
d=np.diff(s); brk=np.where(d>1)[0]
runs_start=np.concatenate([[0],brk+1]); runs_end=np.concatenate([brk,[len(s)-1]])
runlen=s[runs_end]-s[runs_start]+1
print("=== ID GAP STRUCTURE ===")
print("num contiguous runs:",len(runlen))
print("total ids present:",len(s))
print("total span:",s[-1]-s[0]+1)
print("missing ids in span:",(s[-1]-s[0]+1)-len(s))
print("run length stats: max",runlen.max(),"median",int(np.median(runlen)),"mean %.1f"%runlen.mean())
big=np.argsort(runlen)[::-1][:15]
print("largest gaps (missing block between consecutive present-ids):")
for i in np.argsort(-(s[runs_start[1:]]-s[runs_end[:-1]]-1))[:15]:
    gap=s[runs_start[i+1]]-s[runs_end[i]]-1
    hprev=c.loc[c.id==s[runs_end[i]],"host"].iloc[0]; hnext=c.loc[c.id==s[runs_start[i+1]],"host"].iloc[0]
    print(f"   gap {gap:>9} between id {s[runs_end[i]]} ({hprev}) and {s[runs_start[i+1]]} ({hnext})")
