"""Our whole Adam step as an optax chain (2026-10-21): smooth -> scale_by_adam -> smooth ->
area-tangent -> max-norm step, against optimize.py's own sequence of calls, on the circle_long
level set with synthetic band gradients.  The smoothing and the tangent projection are host
(numpy / scipy) transforms -- optax does not need jit -- with the level set and the area
gradient passed as extra args each iterate.

    PYTHONPATH=. python3 scratch/optax/chain.py
"""

import functools
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

jax.config.update("jax_enable_x64", True)


class EmptyState(NamedTuple):
    pass


def _init_empty(params):
    del params
    return EmptyState()


def _smooth_update(updates, state, params=None, *, ls, sigma, band, **extra):
    del params, extra
    return jnp.asarray(O.smoothed_direction(np.asarray(updates), ls, sigma, band)), state


def _tangent_update(updates, state, params=None, *, g_area, **extra):
    del params, extra
    return jnp.asarray(O.tangent_component(np.asarray(updates), g_area)[0]), state


def _max_norm_update(updates, state, params=None, *, step, **extra):
    del params, extra
    return -step * updates / jnp.maximum(jnp.abs(updates).max(), 1e-300), state


def transform(update):
    return optax.GradientTransformationExtraArgs(_init_empty, update)


def main():
    cfg = dict(O.default_config(), case="circle_long", fnum_scale=2.0, smooth=8.0, band=1.0,
               step=0.05, levelset_h=0.0625)  # fmt: skip
    grid, ls = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    band = cfg["band"] + ls.h
    _a, g_area = O.area_and_gradient(ls, lo, hi)
    smooth = functools.partial(_smooth_update, sigma=cfg["smooth"], band=band)
    chain = optax.chain(
        transform(smooth),
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-30, eps_root=0.0),
        transform(smooth),
        transform(_tangent_update),
        transform(_max_norm_update),
    )
    rng = np.random.default_rng(3)
    shape = np.asarray(ls.coef).shape
    state = chain.init(jnp.zeros(shape))
    adam = (np.zeros(shape), np.zeros(shape))
    worst = 0.0
    for k in range(8):
        g = rng.normal(size=shape) * 1e-6
        g, _nu = O.tangent_component(g, g_area)
        # optimize.run's Adam branch
        d = O.smoothed_direction(g, ls, cfg["smooth"], band)
        d, adam = O.adam_direction(d, adam, k, 0.9, 0.999)
        d = O.smoothed_direction(d, ls, cfg["smooth"], band)
        d, _ = O.tangent_component(d, g_area)
        ours = -cfg["step"] / np.abs(d).max() * d
        upd, state = chain.update(jnp.asarray(g), state, None, ls=ls, g_area=g_area,
                                  step=cfg["step"])  # fmt: skip
        worst = max(worst, float(np.abs(ours - np.asarray(upd)).max() / cfg["step"]))
    print(f"max |our step - optax chain step| / step over 8 iterates: {worst:.2e}")


if __name__ == "__main__":
    main()
