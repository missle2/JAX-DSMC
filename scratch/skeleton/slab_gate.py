"""Gate for jax_dsmc/mesh/skeleton.py (2026-11-27): slabs of thickness t = r_s, 1.5 r_s, 2 r_s,
3 r_s, 4 r_s (2-D, spanning the box in x: no tips) and a thick disc; the surface mean of S_c
(P at threshold 1) should fall to ~0 at t >= 2 r_s (Hermann & Noel Fig. 4) and be clearly
positive below.  Also prints the voxel profile of |eta| and S_c across the thinnest slab.

    PYTHONPATH=. python3 scratch/skeleton/slab_gate.py [--r-s 0.2] [--h 0.1] [--eta-c 0.937]
"""

import argparse
import dataclasses

import jax.numpy as jnp
import numpy as np

from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh import skeleton as K


def slab(t, lo, hi, h):
    origin, shape = L.grid_for_box(lo, hi, h)
    template = L.LevelSet(coef=jnp.zeros(shape), lo=jnp.asarray(origin), h=h)
    x = L.knots(template)
    sdf = jnp.abs(x[:, 1]) - 0.5 * t
    return L.from_values(sdf.reshape(shape), origin, h)


def disc(radius, lo, hi, h):
    origin, shape = L.grid_for_box(lo, hi, h)
    return L.from_analytic(np.zeros(2), radius, origin, h, shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-s", type=float, default=0.2)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--eta-c", type=float, default=0.937)
    ap.add_argument("--heat-scale", type=float, default=2.0)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()
    s = K.SkeletonSettings(r_s=args.r_s, eta_c=args.eta_c, heat_scale=args.heat_scale,
                           width=args.width)  # fmt: skip
    lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
    for mult in (1.0, 1.5, 2.0, 3.0, 4.0):
        ls = slab(mult * args.r_s, lo, hi, args.h)
        p, _g = K.penalty_and_gradient(ls, lo, hi, 1.0, s)
        f = K.surface_fields(ls, lo, hi, s)
        col = f["counts"][0] // 2
        y = f["origin"][1] + f["hv"] * (np.arange(f["counts"][1]) + 0.5)
        inside = f["phi"][col] < 0
        eta = np.sqrt(np.sum(f["eta"][:, col] ** 2, axis=0))
        print(f"t = {mult:g} r_s: P {p:.4f}   |eta| inside max {eta[inside].max():.3f} "
              f"min {eta[inside].min():.3f}; S_c at the outermost solid voxel "
              f"{f['s_c'][col][inside][[0, -1]].round(3)}", flush=True)  # fmt: skip
        if mult == 1.0:
            near = np.abs(y) < 2 * args.r_s
            for yy, e, sc, ph in zip(y[near], eta[near], f["s_c"][col][near], f["phi"][col][near],
                                     strict=True):  # fmt: skip
                print(f"    y {yy:+.3f} phi {ph:+.3f} |eta| {e:.3f} S_c {sc:.3f}")
    ls = disc(1.0, lo, hi, args.h)
    p, g = K.penalty_and_gradient(ls, lo, hi, 1.0, s)
    print(f"disc R = 1: P {p:.5f}, |dP/dcoef| {float(jnp.abs(g).max()):.3g}")
    s2 = dataclasses.replace(s)
    del s2


if __name__ == "__main__":
    main()
