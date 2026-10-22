import pandas as pd, re
from urllib.parse import urlparse
L=pd.read_parquet('data/raw/btc/links_corpus_vi_en_zh.parquet'); L=L[L.id<=766441]
L['host']=L.url.map(lambda u: urlparse(u).netloc.lower().removeprefix('www.'))
host=dict(zip(L.id,L.host))
JUNK=re.compile(r'xo-so|bong-da|ket-qua|thoi-tiet|gia-.*hom-nay|tu-vi|cung-hoang-dao|tuyen-sinh|lich-thi-dau|chung-khoan|gia-vang|truc-tiep')
junk=set(L.id[L.url.str.contains(JUNK)])
print('corpus junk-url share', round(len(junk)/len(L),3))
def mix(path,k,name):
    r=pd.read_parquet(path); r=r[r.doc_id<=766441].sort_values(['qid','score'],ascending=[True,False])
    r=r.groupby('qid').head(k)
    h=r.doc_id.map(host).value_counts(normalize=True).head(8)
    print(f'{name} @{k}: junk {r.doc_id.isin(junk).mean():.3f} |', ' '.join(f'{a}:{b:.2f}' for a,b in h.items()))
corp=L.host.value_counts(normalize=True).head(8); print('corpus |',' '.join(f'{a}:{b:.2f}' for a,b in corp.items()))
mix('kaggle/outputs/vibiomir-bm25-e1v2/ranking.parquet',28,'BM25 e1v2 (P.178)')
mix('kaggle/outputs/vibiomir-build-depth-a/w12_VI_ranking.parquet',28,'w12 (P.354)')
mix('kaggle/outputs/vibiomir-build-depth-a/w12_VI_ranking.parquet',150,'w12')
mix('kaggle/outputs/vibiomir-build-depth-c/rd_VI_ranking.parquet',28,'rd')
mix('kaggle/outputs/vibiomir-build-depth-c/rd_VI_ranking.parquet',150,'rd')
