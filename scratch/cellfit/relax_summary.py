"""Relax gate summary: per adjoint key, mean +- se, and the paired gap to the per-seed FD
(mean over the four h; the CRN FD at M40 is linear in h to its noise)."""

import json
import sys

import numpy as np

rows = [json.loads(line) for f in sys.argv[1:] for line in open(f)]
keys = sorted({k for r in rows for k in r["adjoint"] if not k.endswith("seconds")})
for k in keys:
    rs = [r for r in rows if k in r["adjoint"] and r["fd"]]
    g = np.array([r["adjoint"][k] for r in rs])
    f = np.array([np.mean(r["fd"]) for r in rs])
    d = g - f
    secs = [r["adjoint"].get(k.rsplit("_k", 1)[0] + "_seconds", np.nan) for r in rs]
    print(f"{k:24s} n={len(rs):2d} adjoint {g.mean():.4f} FD {f.mean():.4f}  "
          f"gap {d.mean()/f.mean()*100:+5.1f}% +- {d.std(ddof=1)/np.sqrt(len(d))/f.mean()*100:.1f}%"
          f"  backward {np.nanmedian(secs):.1f} s")
