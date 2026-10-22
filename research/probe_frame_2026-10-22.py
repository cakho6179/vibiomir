"""Label-free probe: entity (ICD 3-char) and intent<->section agreement across systems with known LB precision."""
import json, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'scripts'))
import r2ai2026_icd_systems as icd

def code_lexicon():
    lex = {}
    for e in json.loads(icd.ONTOLOGY.read_text(encoding='utf-8'))['entries']:
        c = e['icd10_code'][:3]
        if icd.chapter_of(c) is None: continue
        for name in (e.get('name_vn'),):
            if not name: continue
            toks = tuple(icd.norm(re.sub(r"\(.*?\)", " ", name).split(",")[0]))
            if 2 <= len(toks) <= 8: lex.setdefault(toks, c)
    return lex

INTENTS = {
 'cause': r'nguyên nhân|do đâu|vì sao|tại sao|bị làm sao|do bệnh gì',
 'symptom': r'triệu chứng|dấu hiệu|biểu hiện|cảnh báo bệnh',
 'diagnosis': r'chẩn đoán|xét nghiệm|siêu âm|chụp|kết quả|chỉ số',
 'treatment': r'điều trị|chữa|làm gì|xử lý|khắc phục|cải thiện|phẫu thuật|mổ',
 'drug': r'thuốc|uống|liều|kháng sinh|tiêm',
 'prevent': r'phòng ngừa|phòng tránh|phòng bệnh|tiêm phòng|vắc xin|vaccine',
 'risk': r'nguy hiểm|biến chứng|ảnh hưởng|có sao không|tiến triển|lây',
 'diet': r'ăn|uống gì|dinh dưỡng|thực phẩm|kiêng',
}
INT_RE = {k: re.compile(v) for k, v in INTENTS.items()}
def intents(text): t = text.lower(); return {k for k, r in INT_RE.items() if r.search(t)}

def headings(text):
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if 3 <= len(s) <= 110 and not s.endswith(('.', ',', ';')): out.append(s)
    return ' \n '.join(out)

lex = code_lexicon()
q = pd.read_parquet('data/raw/btc/query.parquet')
qc = {int(r.id): set(icd.tag(r.query, lex)) for r in q.itertuples()}
qi = {int(r.id): intents(r.query) for r in q.itertuples()}
print('queries with code', sum(bool(v) for v in qc.values()), 'with intent', sum(bool(v) for v in qi.values()))

def top(path, lo, hi):
    r = pd.read_parquet(path); r = r[r.doc_id <= 766441].sort_values(['qid','score'], ascending=[True,False])
    r['rk'] = r.groupby('qid').cumcount(); return r[(r.rk >= lo) & (r.rk < hi)][['qid','doc_id']]
O = 'kaggle/outputs/'
sets = {
 'BM25 e1v2 @1-28 (P.178)': top(O+'vibiomir-bm25-e1v2/ranking.parquet', 0, 28),
 'w12 @1-28 (P.354)': top(O+'vibiomir-build-depth-a/w12_VI_ranking.parquet', 0, 28),
 'rd @1-28': top(O+'vibiomir-build-depth-c/rd_VI_ranking.parquet', 0, 28),
 'w12 @29-90': top(O+'vibiomir-build-depth-a/w12_VI_ranking.parquet', 28, 90),
 'w12 @91-150': top(O+'vibiomir-build-depth-a/w12_VI_ranking.parquet', 90, 150),
}
need = set().union(*(set(s.doc_id) for s in sets.values()))
dc, ds = {}, {}
for f in sorted(Path('data/processed_v3').glob('docs-*.parquet')):
    t = pq.read_table(f, columns=['id','title','text']).to_pandas(); t = t[t.id.isin(need)]
    for i, ti, tx in zip(t.id, t.title, t.text):
        lead = f"{ti}\n{headings(tx)}\n{tx[:600]}"
        dc[int(i)] = set(icd.tag(lead, lex)); ds[int(i)] = intents(f"{ti}\n{headings(tx)}")
print('docs', len(dc))
rows = []
for name, s in sets.items():
    ent_both = ent_ok = int_both = int_ok = 0
    for qid, d in zip(s.qid, s.doc_id):
        a, b = qc.get(int(qid), set()), dc.get(int(d), set())
        if a and b: ent_both += 1; ent_ok += bool(a & b)
        a, b = qi.get(int(qid), set()), ds.get(int(d), set())
        if a: int_both += 1; int_ok += bool(a & b)
    rows.append((name, len(s), ent_both, round(ent_ok/max(ent_both,1),3), int_both, round(int_ok/max(int_both,1),3)))
print(pd.DataFrame(rows, columns=['set','pairs','ent_pairs','ent_match','int_pairs','int_match']).to_string(index=False))
