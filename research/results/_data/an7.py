import pandas as pd, sys, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width",250)
c=pd.read_pickle("corpus.pkl")
u=c.url.astype(str)
c["host"]=u.str.extract(r"^[a-z]+://([^/]+)")[0].str.lower().str.replace(r"^www\.","",regex=True)
ZH=set(["120ask.com","cnkang.com","familydoctor.com.cn","ask.39.net","youlai.cn","zhongyibaodian.net","zysjonline.com","msdmanuals.cn","a-hospital.com","wujue.com","zydcd.com","article.iiyi.com","jb39.com","jbk.39.net","test.pmphai.com","baidu.com","familydoctor.cn","qihuangzhishu.com","health.people.com.cn","qy.familydoctor.com.cn","ask.familydoctor.com.cn","bingli.iiyi.com","baidianfeng.familydoctor.com.cn"]) | set(h for h in c.host.unique() if h.endswith(".39.net"))
c["lang"]=np.where(c.host=="pubmed.ncbi.nlm.nih.gov","en",np.where(c.host.isin(ZH),"zh","vi"))
for lg in ["vi","en","zh"]:
    s=np.sort(c[c.lang==lg].id.values)
    span=s[-1]-s[0]+1; present=len(s)
    gaps=np.diff(s)-1
    print(f"=== {lg}: n={present} span={span} density={present/span:.3f} numGaps={(gaps>0).sum()} maxGap={gaps.max()} meanGapWhenGap={gaps[gaps>0].mean() if (gaps>0).any() else 0:.1f}")
    if lg=="vi":
        # per VI host check contiguity
        bad=[]
        for h,sub in c[c.lang=="vi"].groupby("host"):
            ss=np.sort(sub.id.values)
            if ((np.diff(ss)>1).sum()>0): bad.append((h,len(ss),(np.diff(ss)>1).sum(),ss[0],ss[-1]))
        print("   VI hosts with internal id gaps:",len(bad))
        for b in bad: print("      ",b)
print()
# gap-in-id vs host for zh hosts
print("ZH host id contiguity:")
for h,sub in c[c.lang=="zh"].groupby("host"):
    ss=np.sort(sub.id.values)
    print(f"   {h:<40} n={len(ss):>7} span={ss[-1]-ss[0]+1:>9} gaps={(np.diff(ss)>1).sum():>7}")
