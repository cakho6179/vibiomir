import re,sys
sys.stdout.reconfigure(encoding="utf-8")
s=open("medkb_index.js",encoding="utf-8",errors="replace").read()
for key in ["/api/search","/api/chat","/api/translate"]:
    print("="*100); print("KEY",key)
    for m in re.finditer(re.escape(key),s):
        a=max(0,m.start()-1400); b=min(len(s),m.end()+1400)
        print("---- window at",m.start()); print(s[a:b]); print()
