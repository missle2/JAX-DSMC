"""Knot / half-knot pieces and thin share T of the 3-D Newtonian battery's bodies at a few
iterates (2026-10-28).

    PYTHONPATH=. python3 scratch/topo/final_state.py
"""

import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/topo")
import single as S  # noqa: E402

O, T = S.O, S.T
TAGS = ["n3d_ld_hz_re0", "n3d_ld_hz_re0_g", "n3d_ld_hz_re0_t", "n3d_ld_hz_re1", "n3d_ld_hz_re1_g"]


def main():
    jax.config.update("jax_enable_x64", True)
    for tag in TAGS:
        out = S.ROOT / tag
        hist = json.loads((out / "history.json").read_text())
        cfg = dict(O.default_config(), **hist["config"])
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        cells = []
        for k in (50, 100, 149):
            ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(out / f"coef_{k}.npy")))
            t = O.thinness_and_gradient(ls, lo, hi, 2.0 * float(ls.h))[0]
            cells.append(f"@{k}: knots {T.pieces(O.knot_values(ls))} fine "
                         f"{S.fine_pieces(ls, lo, hi)} T {t:.3f}")  # fmt: skip
        print(f"{tag}: " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
