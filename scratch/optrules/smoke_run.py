"""optimize.run end to end for each rule with the DSMC evaluation replaced by smoke_rules'
synthetic objective (seconds on CPU): history rows, opt_state.npz, resume."""

import json
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import optimize as O  # noqa: E402

from smoke_rules import objective  # noqa: E402

GRID, _LS0 = O.initial_levelset(dict(O.default_config(), case="circle_long", fnum_scale=2.0))
LO, HI = tuple(np.asarray(GRID.lo)), tuple(np.asarray(GRID.hi))
F = jax.jit(jax.value_and_grad(objective), static_argnums=(2, 3))


def fake_terms(cfg, ls, seeds, out=None):
    j, g = F(ls.coef, ls, LO, HI)
    return {"frames": None, "gradient": np.asarray(g), "gradient_seed_sd": 0.0,
            "objective": float(j), "objective_se": 1e-6, "hits": 0.0, "stuck": 0.0,
            "seconds": 0.0, "phase_seconds": {}}  # fmt: skip


O.averaged_terms = fake_terms
for rule, extra in (("tr", {}), ("lbfgs", {}), ("mma", {}), ("tr", {"perimeter": 0.05})):
    cfg = dict(O.default_config(), case="circle_long", fnum_scale=2.0, smooth=8.0, band=1.0,
               step=0.05, area_mode="tangent", keepout=1.0, optimizer=rule, iterations=3,
               frames_every=0, **extra)  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        O.run(cfg, out)
        O.run(dict(cfg, iterations=5), out)  # resume
        rows = json.loads((out / "history.json").read_text())["rows"]
        keys = ("objective", "rho", "radius", "accepted", "lbfgs_pairs", "step_max",
                "perimeter", "mma_multipliers")  # fmt: skip
        for r in rows:
            print(rule, {k: r[k] for k in keys if r.get(k) is not None})
        assert len(rows) == 5 and rows[-1]["objective"] < rows[0]["objective"], rule
print("ok")
