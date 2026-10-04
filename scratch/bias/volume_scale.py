"""Gate of the cut-volume term's machinery: scale the NTC flow volume of a set of cells by
(1 + eps) -- the only change -- and compare the CRN FD of the windowed drag in eps with the
adjoint's prediction  dJ/deps = sum_{c in set} g_c V_c  (g_c = shape.volume_weights from
the virtual pull-back's cell collision increments).  Sets: every cell ("all") and the
body's cut cells ("cut").  A large eps keeps the FD's chaos noise small against the signal.

    PYTHONPATH=. python3 scratch/bias/volume_scale.py OUT.jsonl S SEED0 SEED1 '{overrides}'
"""

import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402

_build = EM.build
SCALE = {}


def build_scaled(cfg, parameter, value, steps, body=None):
    out = _build(cfg, parameter, value, steps, body)
    if SCALE:
        grid = out[0]
        vol = grid.volume * jnp.where(SCALE["mask"], 1.0 + SCALE["eps"], 1.0)
        out = (dataclasses.replace(grid, volume=vol), *out[1:])
    return out


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    seeds = range(int(sys.argv[3]), int(sys.argv[4]))
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    eps = over.pop("eps", 0.2)
    sk = over.pop("virtual", [2, 4])
    cfg = dict(OS.default_config(), squash=s, **over)
    ensemble_backward.configure("virtual", sk[0], sk[1], "staged")
    EM.build = build_scaled
    grid, mesh, *_ = _build(cfg, "squash_y", s, 1)
    masks = {"all": jnp.asarray(grid.volume > 0.0),
             "cut": jnp.zeros(grid.volume.shape, bool).at[: mesh.cell_type.shape[0]].set(
                 jnp.asarray(mesh.cell_type) == cut.CUT)}
    for i in seeds:
        key = jax.random.key(cfg["seed_start"] + i)
        t0 = time.time()
        captured = {}
        orig = EM._volume_term

        def grab(dev, rec, grid, mesh, captured=captured, orig=orig):
            captured["w"] = np.asarray(shape.volume_weights(dev.cell_collision, grid))
            captured["v"] = np.asarray(grid.volume)
            return orig(dev, rec, grid, mesh)

        EM._volume_term = grab
        try:
            _t, info, _st = EM.shape_terms(cfg, key, None, "squash_y", 32, 0.2, ("one",))
        finally:
            EM._volume_term = orig
        row = {"s": s, "seed": i, "eps": eps, "config": over, "virtual": sk,
               "objective": info["objective"]}
        for name, mask in masks.items():
            m = np.asarray(mask)
            row[f"pred_{name}"] = float(np.sum(captured["w"][m] * captured["v"][m]))
            vals = []
            for sign in (1.0, -1.0):
                SCALE.update({"mask": mask, "eps": sign * eps})
                _a, _b, diag, _r, tallies, st = EM.forward(cfg, "squash_y", s, key, False)
                SCALE.clear()
                EM._check(diag)
                scale_r = float(st["params"].fnum[0]) / (cfg["window"] * float(st["params"].dt))
                vals.append(EM.objective_value(cfg, tallies, st["static"].D, scale_r))
            row[f"fd_{name}"] = (vals[0] - vals[1]) / (2 * eps)
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
