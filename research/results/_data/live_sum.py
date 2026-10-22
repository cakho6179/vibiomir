import json,sys,pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
d=json.load(open("liveness_head.json",encoding="utf-8"))
df=pd.DataFrame(d)
piv=df.pivot_table(index="host",columns="status",values="url",aggfunc="count").fillna(0).astype(int)
print(piv.to_string()); print()
print("errs:",df[df.status.isna()][["host","url","err"]].to_string())
print()
print("=== 403/404/521 samples with URLs ===")
for h in ["nhathuoclongchau.com.vn","familydoctor.com.cn","medlatec.vn","120ask.com"]:
    s=df[df.host==h]
    print("--",h)
    for _,r in s.iterrows(): print("      ",r.status,str(r.url)[:120])
