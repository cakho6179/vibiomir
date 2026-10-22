import sys,re,glob,os
sys.stdout.reconfigure(encoding="utf-8")
for f in sorted(glob.glob("robots_*.txt")):
    try: s=open(f,encoding="utf-8",errors="replace").read()
    except: continue
    if "<html" in s.lower()[:200] or s.strip()=="": print("=== ",f," (HTML/blocked or empty), len",len(s)); continue
    print("=== ",f,"len",len(s))
    for line in s.splitlines():
        l=line.strip()
        if re.search(r"(?i)bot|sitemap|crawl-delay|disallow: /$|allow: /$|disallow:$",l): print("    ",l)
