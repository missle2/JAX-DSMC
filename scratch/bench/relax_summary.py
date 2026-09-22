"""Summarise relaxation artifacts (shipped, archived) as AD/FD means, sd, relative bias."""
import glob
import json
import sys

import numpy as np

R = "examples/gradient_bench/results/"
pats = sys.argv[1:] or ["linearised_step/ensemble_relax*.json", "hybrid_ghosts/*.json", "ensemble_relax*.json"]
for pat in pats:
    for f in sorted(glob.glob(R + pat)):
        d = json.load(open(f))
        rows = d["rows"]
        ad = np.array([r["gradient"] for r in rows])
        fd = np.array([r["fd"][0] for r in rows])
        n = len(ad)
        print(
            f"{f[len(R):]:60s} n={n:4d} AD {ad.mean():+.4f} sd {ad.std(ddof=1):.3f} se {ad.std(ddof=1)/n**0.5:.4f}"
            f"  FD {fd.mean():+.4f} se {fd.std(ddof=1)/n**0.5:.4f}"
            f"  rel {100*(ad.mean()-fd.mean())/abs(fd.mean()):+.1f}%  z {(ad.mean()-fd.mean())/np.hypot(ad.std(ddof=1), fd.std(ddof=1))*n**0.5:+.1f}"
        )
