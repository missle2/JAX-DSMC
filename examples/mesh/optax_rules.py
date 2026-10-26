"""optax step rules for the level-set optimiser (:mod:`examples.mesh.optimize`, 2026-10-21).

Each arm is an ``optax`` chain from the area-tangent gradient ``g`` (``dJ/dcoef`` without its
component along ``grad A``) to the coefficient step, built from the same pieces as the shipped
Adam step: the band-restricted H1 smoothing (:func:`optimize.smoothed_direction`) before and
after the per-coefficient rule (the elementwise normalisation undoes the smoothing), the
first-order area-tangent projection, and the step size.  ``ox_adam`` is the shipped Adam step
exactly (``scratch/optax/chain.py``: 4e-16 of the step over 8 iterates).

The shipped step rescales every step to the largest normal displacement ``step``; that erases
whatever a rule does to the step's MAGNITUDE (schedules, learning-rate-free estimates, AMSGrad
-type denominators) and keeps only its DIRECTION.  So the arms split:

* direction rules, rescaled to ``step``: ``ox_adam`` (control), ``ox_ademamix`` (AdEMAMix: a
  slow momentum, ``b3 = 0.995`` -- a ~200-iterate horizon for runs of a few hundred iterates
  -- mixed in with ``alpha = 5``: noise-averaged direction);
* magnitude rules: ``ox_adamcos`` (Adam with the rescaled step following a cosine from
  ``2 step`` to ``step / 4`` over :data:`COSINE_ITERATES`, then held: the trust region showed
  larger early steps pay -- 0.35 J0 at iterate 25 against Adam's 0.61 -- and Adam overshoots
  late at M 10), ``ox_sfadam`` (schedule-free: the base Adam step without momentum, rescaled to
  ``step``, iterate-averaged; the gradient is taken at the interpolated point -- an
  average that should damp the late overshoot and the noise-driven roughness without a
  schedule), ``ox_prodigy`` (Prodigy, learning-rate free: its own distance estimate sets
  the step, capped at the trust region's largest radius ``4 step``), ``ox_prodigycos``
  (2026-10-23: Prodigy with that cap following a cosine from ``4 step`` to ``step / 4`` over
  :data:`COSINE_ITERATES`: plain Prodigy's estimate only grows, its step sat at the cap from
  iterate ~25 on in both optax1seed regimes -- the fastest descent, then 0.2 m steps on a
  noisy gradient near the optimum).

The level set and the area gradient of the iterate reach the host transforms (smoothing,
tangent projection) through :class:`Context`, set before every update -- extra arguments do not
pass through every optax wrapper.  The optimiser state is saved as its flattened leaves
(:func:`save_state` / :func:`load_state`) so a resumed run continues the rule.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import contrib

COSINE_ITERATES = 300
CAP = 4.0  # the magnitude rules' largest normal displacement, in units of ``step``
EPS = 1e-30  # the shipped Adam's; optax's 1e-8 is ~1 % of a Kn 0.5 coefficient gradient


class Context:
    """The iterate's level set, area gradient, smoothing settings and nominal step, read by the
    host transforms; :func:`step` sets it before each update."""

    def __init__(self):
        self.ls = None
        self.g_area = None
        self.sigma = 0.0
        self.band = 0.0
        self.step = 0.0


class _Empty(NamedTuple):
    pass


class _Count(NamedTuple):
    count: Any


def _init_empty(params):
    del params
    return _Empty()


def _init_count(params):
    del params
    return _Count(jnp.zeros([], jnp.int32))


def _smooth(updates, state, params=None, *, ctx, **extra):
    from optimize import smoothed_direction  # noqa: PLC0415 -- optimize imports this module

    del params, extra
    return jnp.asarray(smoothed_direction(np.asarray(updates), ctx.ls, ctx.sigma, ctx.band)), state


def _tangent(updates, state, params=None, *, ctx, **extra):
    from optimize import tangent_component  # noqa: PLC0415

    del params, extra
    if ctx.g_area is None:  # area_mode none
        return updates, state
    return jnp.asarray(tangent_component(np.asarray(updates), ctx.g_area)[0]), state


def _rescale(updates, state, params=None, *, ctx, **extra):
    """A descent step whose largest entry is ``ctx.step``."""
    del params, extra
    return -ctx.step * updates / jnp.maximum(jnp.abs(updates).max(), 1e-300), state


def _rescale_scheduled(updates, state, params=None, *, ctx, schedule, **extra):
    """:func:`_rescale` to ``schedule(count) * ctx.step``."""
    del params, extra
    size = schedule(state.count) * ctx.step
    out = -size * updates / jnp.maximum(jnp.abs(updates).max(), 1e-300)
    return out, _Count(state.count + 1)


def _cap(updates, state, params=None, *, ctx, **extra):
    """A descent step (already signed) scaled down to at most ``CAP * ctx.step``."""
    del params, extra
    big = jnp.abs(updates).max()
    return updates * jnp.minimum(1.0, CAP * ctx.step / jnp.maximum(big, 1e-300)), state


def _cap_scheduled(updates, state, params=None, *, ctx, schedule, **extra):
    """:func:`_cap` to ``schedule(count) * ctx.step``."""
    del params, extra
    big = jnp.abs(updates).max()
    cap = schedule(state.count) * ctx.step
    return updates * jnp.minimum(1.0, cap / jnp.maximum(big, 1e-300)), _Count(state.count + 1)


def _cap_cosine(count):
    return optax.cosine_decay_schedule(CAP, COSINE_ITERATES, alpha=1.0 / 16.0)(count)


def _host(update, ctx, **kw):
    return optax.GradientTransformationExtraArgs(
        _init_empty, functools.partial(update, ctx=ctx, **kw)
    )


def _cosine(count):
    return optax.cosine_decay_schedule(2.0, COSINE_ITERATES, alpha=0.125)(count)


def _direction_chain(rule, ctx):
    return optax.chain(_host(_smooth, ctx), rule, _host(_smooth, ctx), _host(_tangent, ctx))


def build(name: str, ctx: Context, step: float) -> optax.GradientTransformationExtraArgs:
    """The arm ``name`` (:data:`ARMS`) on ``ctx``; ``step`` the nominal step (``cfg["step"]``),
    Prodigy's initial distance scale."""
    adam = optax.scale_by_adam(b1=0.9, b2=0.999, eps=EPS, eps_root=0.0)
    if name == "ox_adam":
        return optax.chain(_direction_chain(adam, ctx), _host(_rescale, ctx))
    if name == "ox_ademamix":
        rule = contrib.scale_by_ademamix(b1=0.9, b2=0.999, b3=0.995, alpha=5.0, eps=EPS)
        return optax.chain(_direction_chain(rule, ctx), _host(_rescale, ctx))
    if name == "ox_adamcos":
        scheduled = optax.GradientTransformationExtraArgs(
            _init_count, functools.partial(_rescale_scheduled, ctx=ctx, schedule=_cosine)
        )
        return optax.chain(_direction_chain(adam, ctx), scheduled)
    if name == "ox_sfadam":
        base = optax.chain(
            _direction_chain(optax.scale_by_adam(b1=0.0, b2=0.999, eps=EPS), ctx),
            _host(_rescale, ctx),
        )
        return contrib.schedule_free(base, learning_rate=1.0, b1=0.9)
    if name in ("ox_prodigy", "ox_prodigycos"):
        # its distance estimate starts at a tenth of the nominal step (optax's 1e-6 needed
        # far more iterates than a run has to grow to the displacement scale: zero motion in a
        # 6-iterate smoke, 2026-10-21) and adapts from there
        rule = contrib.prodigy(learning_rate=1.0, eps=EPS, estim_lr0=0.1 * step)
        if name == "ox_prodigy":
            return optax.chain(_direction_chain(rule, ctx), _host(_cap, ctx))
        capped = optax.GradientTransformationExtraArgs(
            _init_count, functools.partial(_cap_scheduled, ctx=ctx, schedule=_cap_cosine)
        )
        return optax.chain(_direction_chain(rule, ctx), capped)
    raise ValueError(f"unknown optax arm {name!r} (one of {ARMS})")


ARMS = ("ox_adam", "ox_ademamix", "ox_adamcos", "ox_sfadam", "ox_prodigy", "ox_prodigycos")


def step(tx, state, ctx, g, ls, g_area, cfg):
    """One update: ``(coefficient step, new state)`` from the area-tangent gradient ``g`` of the
    level set ``ls`` (whose coefficients are the parameters)."""
    ctx.ls, ctx.g_area = ls, g_area
    ctx.sigma, ctx.band, ctx.step = cfg["smooth"], cfg["band"] + ls.h, cfg["step"]
    updates, state = tx.update(jnp.asarray(g), state, jnp.asarray(ls.coef))
    return np.asarray(updates), state


def save_state(path: Path, state) -> None:
    leaves = jax.tree_util.tree_leaves(state)
    np.savez(path, **{f"leaf{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)})


def load_state(path: Path, fresh):
    """The saved leaves in the structure of ``fresh`` (``tx.init`` of the same arm), or
    ``fresh`` if nothing was saved."""
    if not path.exists():
        return fresh
    z = np.load(path)
    leaves, treedef = jax.tree_util.tree_flatten(fresh)
    saved = [jnp.asarray(z[f"leaf{i}"], dtype=jnp.asarray(leaf).dtype)
             for i, leaf in enumerate(leaves)]  # fmt: skip
    return jax.tree_util.tree_unflatten(treedef, saved)
