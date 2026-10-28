"""Pieces (knot lattice / half-knot sampling) and thin share T (depth 2 knots) along the 2-D
M10 lsfilter runs pulled 2026-10-28: diamond and plate starts, circle-start L / D (Adam, MMA).

    PYTHONPATH=. python3 scratch/topo/sharp_pieces.py
"""

import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/topo")
import optimize as O  # noqa: E402
import single as S  # noqa: E402

from jax_dsmc.mesh import topology as T  # noqa: E402

C, W = "circle_long_kn0.5_M10_S1", "sm8_free_g32_w500"
TAGS = [f"{C}_ox_adam_dmhz0_tan_{W}", f"{C}_ox_adam_plhz0_floor_{W}",
        f"{C}_ox_adam_ldhz0_floor_{W}", f"{C}_mma_s_ldhz0_floor_{W}"]  # fmt: skip


def main():
    jax.config.update("jax_enable_x64", True)
    for tag in TAGS:
        out = S.ROOT / tag
        hist = json.loads((out / "history.json").read_text())
        cfg = dict(O.default_config(), **hist["config"])
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        ks = sorted(int(p.stem[5:]) for p in out.glob("coef_*.npy"))
        print(tag, flush=True)
        multi = []
        for k in ks:
            ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(out / f"coef_{k}.npy")))
            kp, fp = T.pieces(O.knot_values(ls)), S.fine_pieces(ls, lo, hi)
            if kp > 1 or fp > 1:
                multi.append((k, kp, fp))
            if k % 40 == 0 or k == ks[-1]:
                t = O.thinness_and_gradient(ls, lo, hi, 2.0 * float(ls.h))[0]
                print(f"  {k:4d}  pieces {kp}/{fp}  T {t:.3f}", flush=True)
        print(f"  multi-piece iterates ({len(multi)} of {len(ks)}): {multi[:30]}", flush=True)


if __name__ == "__main__":
    main()
