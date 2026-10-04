"""Is the windowed drag steady?  Per-step body drag over the captured 300 steps (window =
last 200), mean over seeds, specular and diffuse, s = 1, fnum x2: the slope over the
window is what makes the window-edge timing term nonzero in expectation.

    PYTHONPATH=. python3 scratch/bias/drag_trend.py SEEDS
"""

import json
import sys

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

out = {}
for acc in (0.0, 1.0):
    cfg = dict(OS.default_config(), squash=1.0, fnum_scale=2, acc=acc)
    per = []
    for i in range(int(sys.argv[1])):
        _s0, _f, _d, _r, tallies, st = EM.forward(cfg, "squash_y", 1.0,
                                                   jax.random.key(32 + i), False)
        dim = st["static"].D
        scale = float(st["params"].fnum[0]) / float(st["params"].dt)
        per.append(np.asarray(tallies.momentum[:, 2 * dim :, 0].sum(axis=1)) * scale)
    per = np.array(per)  # (seeds, M) drag per step [N/m]
    m = per.mean(0)
    w = m[-200:]
    k = np.arange(200)
    slope = np.polyfit(k, w, 1)[0]
    first, last = per[:, 100:120].mean(), per[:, -20:].mean()
    out[acc] = {"mean": float(w.mean()), "slope_per_step": float(slope),
                "rel_change_over_window": float(slope * 200 / w.mean()),
                "first20": float(first), "last20": float(last),
                "se20": float(per[:, -20:].mean(1).std(ddof=1) / np.sqrt(len(per)))}
    print(acc, json.dumps(out[acc]), flush=True)
