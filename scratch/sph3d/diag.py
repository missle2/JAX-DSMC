"""Body statistics of the sph3d runs at a few iterates (2026-10-26)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/sph3d")
import shape3d as S  # noqa: E402
from shape3d import O  # noqa: E402

ROOT = Path(O.RESULTS)
for tag in sys.argv[1:]:
    out = ROOT / tag
    hist = json.loads((out / "history.json").read_text())
    cfg = dict(O.default_config(), **hist["config"])
    grid, ls0 = O.initial_levelset(cfg)
    g = S.PhiGrid(ls0, [(-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0)], 0.05)
    n = len(hist["rows"])
    print(tag, "area0", hist.get("area_target"))
    for k in sorted({0, n // 4, n // 2, 3 * n // 4, n}):
        p = out / f"coef_{k}.npy"
        if p.exists():
            st = S.body_stats(g.phi(np.load(p)), g.lo, g.h)
            j = hist["rows"][min(k, n - 1)]["objective"]
            print(f"  k {k:3d} J {j:.4g} {st}", flush=True)
