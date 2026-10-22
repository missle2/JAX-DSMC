"""optimize.run end to end for every optax arm (examples/mesh/optax_rules.py) with the DSMC
evaluation replaced by scratch/optrules/smoke_rules' synthetic objective (2026-10-21): ox_adam
reproduces the shipped adam row for row; every arm lowers the objective and resumes (3 iterates,
then resumed to 6) with the same result as an uninterrupted 6-iterate run.

    PYTHONPATH=. python3 scratch/optax/smoke.py [arm ...]
"""

import json
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optrules")
import optax_rules as OX  # noqa: E402
import optimize as O  # noqa: E402

from smoke_rules import objective  # noqa: E402

GRID, _LS0 = O.initial_levelset(dict(O.default_config(), case="circle_long", fnum_scale=2.0))
LO, HI = tuple(np.asarray(GRID.lo)), tuple(np.asarray(GRID.hi))
F = jax.jit(jax.value_and_grad(objective), static_argnums=(2, 3))


def fake_terms(cfg, ls, seeds, out=None):
    j, g = F(ls.coef, ls, LO, HI)
    objs = float(j) + 1e-6 * np.asarray(seeds, float)
    return {"frames": None, "gradient": np.asarray(g), "gradient_seed_sd": 0.0,
            "objective": float(objs.mean()), "objectives": objs.tolist(), "objective_se": 1e-6,
            "hits": 0.0, "stuck": 0.0, "seconds": 0.0, "phase_seconds": {}}  # fmt: skip


O.averaged_terms = fake_terms


def history(optimizer, iterations, resume_at=None):
    cfg = dict(O.default_config(), case="circle_long", fnum_scale=2.0, smooth=8.0, band=1.0,
               step=0.05, area_mode="tangent", keepout=1.0, optimizer=optimizer,
               iterations=iterations, frames_every=0, seeds_per_iterate=1)  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        if resume_at:
            O.run(dict(cfg, iterations=resume_at), out)
        O.run(cfg, out)
        rows = json.loads((out / "history.json").read_text())["rows"]
    return np.array([r["objective"] for r in rows]), [r.get("step_max") for r in rows]


def main():
    ref, _ = history("adam", 6)
    for arm in sys.argv[1:] or OX.ARMS:
        whole, steps = history(arm, 6)
        resumed, _ = history(arm, 6, resume_at=3)
        same = np.max(np.abs(whole - resumed) / np.abs(whole))
        note = ""
        if arm == "ox_adam":
            note = f"  vs shipped adam: max rel diff {np.max(np.abs(whole - ref) / ref):.1e}"
        steps = ", ".join(f"{s:.3f}" for s in steps if s is not None)
        print(f"{arm:12s} J {whole[0]:.3f} -> {whole[-1]:.3f}  resume rel diff {same:.1e}  "
              f"steps [{steps}]{note}", flush=True)
        assert whole[-1] < whole[0] and same < 1e-9, arm


if __name__ == "__main__":
    main()
