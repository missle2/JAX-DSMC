"""Seconds per optimiser step: optimize.py's host Adam step against the optax chain of
scratch/optax/chain.py, on the circle_long level set's coefficients (2026-10-21)."""

import functools
import sys
import time

import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from chain import _max_norm_update, _smooth_update, _tangent_update, transform  # noqa: E402


def main(n=20):
    cfg = dict(O.default_config(), case="circle_long", fnum_scale=2.0, smooth=8.0, band=1.0,
               step=0.05, levelset_h=0.0625)  # fmt: skip
    grid, ls = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    band = cfg["band"] + ls.h
    t0 = time.perf_counter()
    _a, g_area = O.area_and_gradient(ls, lo, hi)
    t_area = time.perf_counter() - t0
    shape = np.asarray(ls.coef).shape
    rng = np.random.default_rng(0)
    gs = [rng.normal(size=shape) * 1e-6 for _ in range(n)]
    adam = (np.zeros(shape), np.zeros(shape))
    t0 = time.perf_counter()
    for k, g in enumerate(gs):
        d = O.smoothed_direction(g, ls, cfg["smooth"], band)
        d, adam = O.adam_direction(d, adam, k, 0.9, 0.999)
        d = O.smoothed_direction(d, ls, cfg["smooth"], band)
        d, _ = O.tangent_component(d, g_area)
        _ = -cfg["step"] / np.abs(d).max() * d
    t_ours = (time.perf_counter() - t0) / n
    smooth = functools.partial(_smooth_update, sigma=cfg["smooth"], band=band)
    chain = optax.chain(transform(smooth), optax.scale_by_adam(eps=1e-30, eps_root=0.0),
                        transform(smooth), transform(_tangent_update),
                        transform(_max_norm_update))  # fmt: skip
    state = chain.init(jnp.zeros(shape))
    upd, state = chain.update(jnp.asarray(gs[0]), state, None, ls=ls, g_area=g_area, step=0.05)
    t0 = time.perf_counter()
    for g in gs:
        upd, state = chain.update(jnp.asarray(g), state, None, ls=ls, g_area=g_area, step=0.05)
    upd.block_until_ready()
    t_optax = (time.perf_counter() - t0) / n
    print(f"coefficients {np.prod(shape)}; area gradient (both paths, once per iterate) "
          f"{t_area:.3f} s incl. compile")
    print(f"our Adam step {t_ours * 1e3:.1f} ms; optax chain {t_optax * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
