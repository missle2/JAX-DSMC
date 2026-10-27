"""Does reinitialisation drive the 3-D L / D bodies' bubbly roughness too (2026-10-26)?  As
reinit_change.py, over the whole surface (|phi| < 0.3 at the knot): one extra
``levelset.reinitialise`` pass's coefficient change (rms, knot-scale share) against the run's
per-iterate change, and the body's half-thickness in knots (the median of -phi's local maxima,
the medial surface)."""

import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

L = O.L
for tag in sys.argv[1:]:
    out = Path(O.RESULTS) / tag
    hist = json.loads((out / "history.json").read_text())
    rows, cfg = hist["rows"], dict(O.default_config(), **hist["config"])
    _g, ls0 = O.initial_levelset(cfg)
    h = float(ls0.h)
    print(tag)
    for k in sorted({10, 30, 45, 60, 75, len(rows) - 2}):
        if k + 1 >= len(rows) + 1 or not (out / f"coef_{k + 1}.npy").exists():
            continue
        c0 = np.load(out / f"coef_{k}.npy")
        ls = dataclasses.replace(ls0, coef=jnp.asarray(c0))
        phi = np.asarray(L.value(ls, L.knots(ls))).reshape(c0.shape)
        sel = np.abs(phi) < 0.3
        r = np.asarray(L.reinitialise(ls).coef) - c0
        hp = r - gaussian_filter(r, 1.0, mode="nearest")
        run = np.load(out / f"coef_{k + 1}.npy") - c0
        inner = -phi
        ridge = (inner > 0) & (inner == maximum_filter(inner, size=3))
        half = np.median(inner[ridge]) / h if ridge.any() else float("nan")
        print(f"  k {k:3d}: reinit alone rms {np.sqrt(np.mean(r[sel] ** 2)):.4f} (knot-scale "
              f"{np.linalg.norm(hp[sel]) / max(np.linalg.norm(r[sel]), 1e-30):.2f})  run's "
              f"change rms {np.sqrt(np.mean(run[sel] ** 2)):.4f}  half-thickness "
              f"{half:.2f} knots", flush=True)
