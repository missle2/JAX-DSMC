"""The shape VJP's batched-steps axis (:func:`jax_dsmc.shape._batched_scan`).

The move's runtime branches -- a pass skipped when no particle has an event, a compacted
path unless it overflowed, the smallest compaction tier that holds the candidates -- are
``lax.cond`` / ``lax.switch`` on a value of the step's own particles.  Under ``vmap`` over
captured steps such a predicate is batched, and JAX evaluates every branch for every step
and selects (no skip, the dense fallbacks' memory on every call).  The predicates below are
reduced over the steps' named axis instead, so the batch takes ONE branch: the most
general that any of its steps needs (a pass runs if any step has an event, the dense
fallback if any overflowed, the largest tier any needs).  Every branch of these choices
gives the same results on the rows it is given (the cheaper ones are shortcuts valid
when they apply), so a step's results do not depend on its batch-mates.  Outside the
batched axis (the forward, the sequential replay) they are the plain reductions.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

STEPS = "captured_steps"
"""The ``vmap`` axis name of the batched shape VJP."""


def _over_steps(value, collective):
    try:
        return collective(value, STEPS)
    except NameError:  # not under the batched steps: the step's own value
        return value


def any_(x):
    """``jnp.any(x)``, or over every step of the batch."""
    return _over_steps(jnp.any(x).astype(jnp.int32), lax.pmax) > 0


def all_(x):
    """``jnp.all(x)``, or over every step of the batch."""
    return _over_steps(jnp.all(x).astype(jnp.int32), lax.pmin) > 0


def tier(branch):
    """A ``lax.switch`` index of ordered tiers (each holds what the lower ones do): the
    batch's largest."""
    return _over_steps(branch, lax.pmax)
