"""The collision pull-back in situ on the fitgate gates (2026-11-01): scale the NTC flow volume
of a set of collision cells by (1 + eps) -- nothing else changes -- and compare the CRN central
difference of the windowed objective in eps with the backward's prediction

    dJ/deps = sum_{c in set} g_c V_c,   g_c = shape.volume_weights (the cells' collision
                                        increments C_kc of the virtual pull-back, -sum_k C_kc / V_c)

(the squash version: scratch/bias/volume_scale.py, 2026-10-02).  Every NTC probability of a cell
scales as 1 / V_c, so this is the adjoint's sensitivity to the collision rates themselves: the
weights of the cut-volume term (the M 10 bias's suspects, with the slope-fit terms fed by the
same pull-back), free of the geometry, the fits of grad lambda and the wall terms.  Sets:
``all`` (every collision cell: the whole shock layer) and ``cut`` (the body's cut cells: what the
cut-volume term weighs); two eps per set (linearity: the FD's own curvature).

    PYTHONPATH=. python3 scratch/fitgate/volume_gate.py --gate c10_spec --seeds 0 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path("examples/mesh/results/fitgate")
EPS = (0.2, 0.1)
_BUILD = EM.build
_VOLUME_TERM = EM._volume_term
SCALE = {}
CAPTURED = {}


def build_scaled(cfg, parameter, value, steps, body=None):
    """``EM.build`` with the flow volume of ``SCALE["mask"]``'s cells times ``1 + eps``."""
    out = _BUILD(cfg, parameter, value, steps, body)
    if SCALE:
        grid = out[0]
        vol = grid.volume * jnp.where(SCALE["mask"], 1.0 + SCALE["eps"], 1.0)
        out = (dataclasses.replace(grid, volume=vol), *out[1:])
    return out


def grab_volume_term(dev, rec, grid, mesh):
    """``EM._volume_term``, keeping the weights and the volumes it saw."""
    CAPTURED["w"] = np.asarray(shape.volume_weights(dev.cell_collision, grid))
    CAPTURED["v"] = np.asarray(grid.volume)
    return _VOLUME_TERM(dev, rec, grid, mesh)


def objective(cfg, parameter, value, key):
    _a, _b, diag, _r, tallies, st = EM.forward(cfg, parameter, value, key, False)
    EM._check(diag)
    scale_r = float(st["params"].fnum[0]) / (cfg["window"] * float(st["params"].dt))
    return EM.objective_value(cfg, tallies, st["static"].D, scale_r)


def masks(cfg, parameter, value):
    grid, mesh, *_ = _BUILD(cfg, parameter, value, 1)
    cells = jnp.zeros(grid.volume.shape, bool)
    cells = cells.at[: mesh.cell_type.shape[0]].set(jnp.asarray(mesh.cell_type) == cut.CUT)
    return {"all": jnp.asarray(grid.volume > 0.0), "cut": cells}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first, end (exclusive)")
    args = ap.parse_args(argv)
    cfg, parameter, _steps, seed_start, _rows = V.gate_config(args.gate)
    value = float(EM.nominal(cfg, parameter))
    out = OUT_DIR / f"volume_{args.gate}_s{args.seeds[0]}-{args.seeds[1]}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines()}
    ensemble_backward.configure("virtual", 4, 8, "staged", V.BASES["base"])
    EM.build = build_scaled
    EM._volume_term = grab_volume_term
    sets = masks(cfg, parameter, value)
    for seed in range(*args.seeds):
        if seed in done:
            continue
        key = jax.random.key(seed_start + seed)
        t0 = time.time()
        CAPTURED.clear()
        _t, info, _s = EM.shape_terms(
            cfg, key, None, parameter, 32, cfg.get("timing_band", 10.0), ("cell_pool_plane",)
        )
        row = {"gate": args.gate, "seed": seed, "objective": info["objective"], "eps": list(EPS)}
        for name, mask in sets.items():
            m = np.asarray(mask)
            row[f"pred_{name}"] = float(np.sum(CAPTURED["w"][m] * CAPTURED["v"][m]))
            fds = []
            for eps in EPS:
                vals = []
                for sign in (1.0, -1.0):
                    SCALE.update(mask=mask, eps=sign * eps)
                    vals.append(objective(cfg, parameter, value, key))
                    SCALE.clear()
                fds.append((vals[0] - vals[1]) / (2 * eps))
            row[f"fd_{name}"] = fds
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
