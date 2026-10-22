import pandas as pd,numpy as np,sys,json
sys.stdout.reconfigure(encoding="utf-8")
c=pd.read_pickle("corpus_cat.pkl")
lg=pd.DataFrame(json.load(open("liveness_get.json",encoding="utf-8")))
lgok=lg[(lg.status==200)&(lg.text_chars>500)]
host_stats=lgok.groupby("host").agg(words=("text_words","median"),chars=("text_chars","median"),bytes=("bytes","median"))
cnt=c.groupby("host").size().rename("n")
cntlang=c.groupby("host").lang.first().rename("lang")
t=pd.concat([cnt,host_stats,cntlang],axis=1)
# manuscript/fallback medians
med_words_vi=t[(t.lang=="vi")]["words"].median(); med_words_zh=t[(t.lang=="zh")]["words"].median()
print("VI median words among measured hosts:",med_words_vi,"| ZH median words:",med_words_zh)
print()
# token heuristic
def tok(r):
    if r.lang=="en": return 300          # PubMed abstract ~250-300 words -> tokens
    if r.lang=="zh": return (r.chars if not np.isnan(r.chars) else 3000)/1.5   # CJK chars ~0.67 tok/char
    w=r.words if not np.isnan(r.words) else med_words_vi
    return w*1.35                        # Vietnamese ~1.35 tokens/word (multilingual BPE)
t["tok_per_doc"]=t.apply(tok,axis=1)
t["tokens"]=t.n*t.tok_per_doc
print(t.sort_values("n",ascending=False).to_string())
print()
tot=t.tokens.sum()
print("ESTIMATED TOTAL TOKENS for exposed 1.22M corpus: %.2f B"%(tot/1e9))
for l in ["vi","en","zh"]:
    s=t[t.lang==l]; print(f"   {l}: {s.n.sum():>8} docs, {s.tokens.sum()/1e9:.2f} B tokens")
print()
print("Storage estimates:")
print("   raw HTML @ median %d KB -> %.0f GB"%(lgok.bytes.median()/1024, 1220598*lgok.bytes.median()/1e9))
print("   extracted text @ median %d chars (~%.1f KB utf8) -> %.1f GB"%(lgok.text_chars.median(), lgok.text_chars.median()*1.35/1024, 1220598*lgok.text_chars.median()*1.35/1e9))
print("   chunks @512tok, ~%.1f chunks/doc -> %.1f M chunks; parquet int32 idx ~ %.1f GB"%( (tot/1220598)/512, 1220598*(tot/1220598)/512/1e6, 1220598*(tot/1220598)/512*1e6*1000/1e9))
print()
# priority tiers
t2=t.dropna(subset=["n"]).copy()
print("=== CRAWL PRIORITY TIERS (by doc count & reachability) ===")
reach={"120ask.com":"down(521)","nhathuoclongchau.com.vn":"blocked(403 CF)","familydoctor.com.cn":"blocked(403 ESA)","laodong.vn":"JS-cookie gate","pubmed.ncbi.nlm.nih.gov":"JS gate -> use E-utilities"}
for h,r in reach.items():
    if h in t2.index: t2.loc[h,"note"]=r
print(t2.sort_values("n",ascending=False).head(30).to_string())
t.to_csv("host_token_estimate.csv")
