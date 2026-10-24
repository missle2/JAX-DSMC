"""Does the centroid constraint itself roughen a body (2026-10-24, the optct break-up)?  The
real ox_adam chain, constraint and restoration (examples/mesh/optimize.py) for ITERS iterates
from the optct M 0.3 Adam body at iterate 70 (its nose already a sharp tip), on a synthetic
gradient: smoothed noise on the surface plus a steady push downstream (the drift tendency),
no DSMC.  Variants: area only, the centroid rows (optcm), raw translation modes (optct),
translation modes smoothed like the step, and "restore": the area-only step with the body
translated back after it (no projection of the gradient or the step).  Roughness (total turning) every 10 iterates.

    PYTHONPATH=. python3 scratch/optax/constraint_replay.py [variant ...]
"""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
sys.path.append("scratch/optrules")
import optax_rules as OX  # noqa: E402
import optimize as O  # noqa: E402
from analysis import ROOT, tag  # noqa: E402
from roughness import turning  # noqa: E402

ITERS = 100
START = 70
PUSH = 0.3  # the translation push relative to the noise's size


def constraint(variant, ls, lo, hi, g_area, cfg):
    if variant in ("area", "restore"):
        return g_area
    rows = np.concatenate((g_area[None], O.centroid_and_gradient(ls, lo, hi)[1]))
    if variant == "rows":
        return rows
    modes = O.translation_modes(ls, cfg["band"] + ls.h)
    if variant == "modes_smooth":
        modes = np.stack([O.smoothed_direction(m, ls, cfg["smooth"], cfg["band"] + ls.h)
                          for m in modes])  # fmt: skip
    return np.stack((rows, np.concatenate((g_area[None], modes))))


def run(variant, cfg, ls, lo, hi):
    area0 = float(O.L.body_volume(ls, lo, hi, 4))
    centroid0 = np.asarray(O._centroid_of_coef(ls.coef, ls, lo, hi))
    ctx = OX.Context()
    tx = OX.build("ox_adam", ctx, cfg["step"])
    state = tx.init(jnp.asarray(ls.coef))
    rng = np.random.default_rng(0)
    out = []
    for k in range(ITERS + 1):
        if k % 10 == 0:
            segs = O.surface_polyline(ls, lo, hi, n=1500)
            c = np.asarray(O._centroid_of_coef(ls.coef, ls, lo, hi))
            out.append((k, round(turning(max(segs, key=len))[0], 2), len(segs), round(float(c[0]), 2)))
        if k == ITERS:
            break
        g_area = O.area_and_gradient(ls, lo, hi)[1]
        near = np.abs(g_area) > 1e-3 * np.abs(g_area).max()
        noise = gaussian_filter(rng.normal(size=ls.coef.shape), 1.0) * near
        noise /= np.abs(noise).max()
        push = O.translation_modes(ls, cfg["band"] + ls.h)[0] * near  # d phi = -d_x phi: downstream
        g = -(noise + PUSH * push / np.abs(push).max()) * np.abs(g_area).max()
        cons = constraint(variant, ls, lo, hi, g_area, cfg)
        g, _ = O.constrained_gradient(g, cons)
        delta, state = OX.step(tx, state, ctx, g, ls, cons, cfg)
        stepped = dataclasses.replace(ls, coef=ls.coef + jnp.asarray(delta))
        ls, _, _ = O.after_step(stepped, "tangent", area0, lo, hi, cfg["keepout"],
                                None if variant == "area" else centroid0)  # fmt: skip
        # "restore": the area-only step, the body moved back rigidly after it
    return out


def main():
    regime, arm = "kn0.5", "ox_adam_ct"
    cfg = dict(O.default_config(), **json.load(open(f"{ROOT}/{tag(regime, arm)}/history.json"))["config"])  # fmt: skip
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{ROOT}/{tag(regime, arm)}/coef_{START}.npy")))
    for variant in sys.argv[1:] or ("area", "rows", "modes", "modes_smooth", "restore"):
        print(variant, run(variant, cfg, ls, lo, hi), flush=True)


if __name__ == "__main__":
    main()
