import json,sys,pandas as pd,numpy as np
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
o=json.load(open("medkb_query_probe.json",encoding="utf-8"))
errs=[x.get("err") for x in o if "err" in x]
print("error types:",Counter([e[:70] for e in errs]).most_common(10))
ok=[x for x in o if "err" not in x]
print("ok qids:",[x["qid"] for x in ok])
tl=[x["textlen_med"] for x in ok]
print("per-query median text len stats: min %.0f med %.0f max %.0f"%(min(tl),np.median(tl),max(tl)))
print("total URLs found in corpus:",sum(x["n_in_corpus"] for x in ok),"/",sum(x["n_url"] for x in ok))
