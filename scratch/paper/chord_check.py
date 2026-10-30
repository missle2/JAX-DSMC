"""The airfoil chord strip on the start ellipse and on a shortened body: chord before / after."""

import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from examples.mesh import optimize as O

tag, strip = sys.argv[1], sys.argv[2]
run = Path(O.RESULTS) / tag
cfg = dict(O.default_config(), **json.loads((run / "history.json").read_text())["config"])
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
centre = O.F.CASES[cfg["case"]]["centre"]
design = O.parse_region(cfg["design_region"], centre)
solid = O.parse_region(strip, centre)
for k in sys.argv[3:]:
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(run / f"coef_{k}.npy")))
    held, moved = O.hold_regions(ls, design, solid)
    for name, b in (("as run", ls), ("held", held)):
        segs = [s for s in O.surface_polyline(b, lo, hi) if len(s)]
        pts = np.concatenate(segs)
        print(f"it {k} {name}: chord x {pts[:, 0].min():.3f}..{pts[:, 0].max():.3f}, "
              f"segments {len(segs)}, knots moved {moved if name == 'held' else 0}")
