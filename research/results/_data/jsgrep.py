import re,sys
sys.stdout.reconfigure(encoding="utf-8")
s=open("medkb_index.js",encoding="utf-8",errors="replace").read()
for kw in ["trích đoạn","trich doan","đoạn trích","chunk","Đoạn","phân đoạn","section","Nguồn","nguồn tài liệu","Xem thêm","Thu gọn","similar_queries","translated_query"]:
    hits=[m.start() for m in re.finditer(re.escape(kw),s)]
    print(f"{kw!r:26} occurrences={len(hits)}")
print()
# look at the render path for result text (the component that shows r.text)
for m in re.finditer(r"r\.text",s):
    a=max(0,m.start()-350); b=min(len(s),m.end()+250)
    print("---"); print(s[a:b].replace("\n"," ")[:600])
