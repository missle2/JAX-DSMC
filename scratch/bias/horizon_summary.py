"""Summary of horizon.py rows: per variant (with the timing term) the gradient at each
captured horizon and the paired change.  python3 scratch/bias/horizon_summary.py FILE"""

import json
import sys

import numpy as np


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


r = [json.loads(line) for line in open(sys.argv[1])]
h = list(r[0]["horizons"])
for n in r[0]["horizons"][h[0]]["variants"]:
    g = {m: np.array([x["horizons"][m]["variants"][n] + x["horizons"][m].get("timing", 0.0)
                      for x in r]) for m in h}
    d = g[h[1]] - g[h[0]]
    print(f"{n:18s} " + "  ".join(f"M{m} {g[m].mean():.5f}+-{se(g[m]):.5f}" for m in h)
          + f"  diff {d.mean():+.2e}+-{se(d):.1e} ({d.mean() / g[h[0]].mean() * 100:+.1f}%)")
f = np.array([np.mean(x["fd"]) for x in r])
print(f"n={len(r)}  FD {f.mean():.5f}+-{se(f):.5f}  seconds {[round(x['seconds']) for x in r]}")
