import re,sys
sys.stdout.reconfigure(encoding="utf-8")
s=open("medkb_index.js",encoding="utf-8",errors="replace").read()
for kw in ["section_types","credibility","similar_queries","translated_query","doc_type"]:
    for m in re.finditer(re.escape(kw),s):
        a=max(0,m.start()-500); b=min(len(s),m.end()+420)
        print("="*40,kw)
        print(s[a:b].replace("\n"," ")[:900])
        print()
        break
