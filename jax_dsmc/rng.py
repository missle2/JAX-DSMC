"""The RNG contract (spec section 1).

**This is the single most important contract in the system.** Get it wrong and
gradients are silently incorrect, finite-difference checks become pure noise, and
``jax.checkpoint`` produces a different trajectory on the backward pass than the
forward pass took.

The whole contract is one function::

    key_for(base_key, step, stage_id) = fold_in(fold_in(base_key, step), stage_id)

and four rules:

1. **Keys depend only on** ``base_key``, the loop index, and a static stage id --
   never on a traced state value.  A key that depended on state would open a
   gradient path through the RNG, which is meaningless.
2. **Keys are never carried in the state.**  They are recomputed from
   ``(base_key, step, stage)`` every time, which is what makes rematerialisation
   under ``jax.checkpoint`` reproduce the *same* draws on the backward pass.
3. **Each stage draws all its randomness in one batched call**, not per-element
   splits.
4. **Common random numbers**: two runs with different parameters but the same
   ``base_key`` consume an identical stream.  Without this the finite-difference
   gradient check is pure noise (section 13.4).

Contrast [DF], which seeds from ``clock::getTime() + const * Pstream::myProcNo()``
so runs are not reproducible unless ``seedNumber`` is pinned.  Our contract has to
be stronger: the FD gradient checks are meaningless without bit-identical common
random numbers across perturbed runs.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import random

# --- stage ids (section 1).  Module-level Python ints, never traced. --------
STAGE_INIT = 0
"""Particle initialisation."""
STAGE_MOVE = 1
"""Advection sub-steps / randomised ``tau`` (M7)."""
STAGE_WALL = 2
"""Diffuse re-emission draws (M4+)."""
STAGE_NTC_ROUND = 3
"""Stochastic rounding of the attempt count (section 8.1)."""
STAGE_PAIR = 4
"""Collision partner selection (section 8.3)."""
STAGE_ACCEPT = 5
"""NTC accept/reject uniform (section 7.2)."""
STAGE_SCATTER = 6
"""VSS deflection angle and azimuth (section 7.4)."""
STAGE_INFLOW = 7
"""Inflow flux sampling (M5+)."""

N_STAGES = 8
"""Number of declared stages.  Used by the contract tests to sweep all of them."""


def key_for(base_key: jax.Array, step, stage_id: int) -> jax.Array:
    """Derive the key for one stage of one timestep.

    Args:
        base_key: the run's fixed ``PRNGKey``.
        step: the ``lax.scan`` loop index.  An integer -- traced is fine (it is a
            loop counter, not a state value), a float is not.
        stage_id: a module-level ``STAGE_*`` constant.  STATIC.

    Returns:
        A ``PRNGKey``.

    Two folds, not one, so that step and stage are independent axes: stage 3 of
    step 7 is unrelated to stage 7 of step 3, which a single additive fold would
    not guarantee.
    """
    return random.fold_in(random.fold_in(base_key, step), stage_id)


def key_for_slot(stage_key: jax.Array, slot) -> jax.Array:
    """Fold the attempt-slot index into a stage key (section 1, final paragraph).

    The attempt-slot ``scan`` of section 8.4 needs a fresh stream per slot ``a``.
    ``a`` is a ``scan`` loop index, so folding it in preserves rule 1.

    Kept as a named function rather than an inline ``fold_in`` so that the
    contract test can assert every per-slot stream in the codebase goes through
    exactly this path.
    """
    return random.fold_in(stage_key, slot)


def _uniform_at(key, i, shape):
    return random.uniform(random.fold_in(key, i), shape, dtype=jnp.float64)


def _normal_at(key, i, shape):
    return random.normal(random.fold_in(key, i), shape, dtype=jnp.float64)


def _randint_at(key, i, shape, maxval):
    return random.randint(random.fold_in(key, i), shape, 0, maxval)


def indexed_uniform(key, n, trailing_shape=()):
    """Capacity-independent candidate draws, batched into one compiled operation.

    Draw ``i`` depends only on ``(key, i)``, so growing ``n`` appends draws and
    leaves every earlier one unchanged (section 1, rule 3).  Bodies are module-level
    and bound with ``partial`` (decision 14).
    """
    return jax.vmap(functools.partial(_uniform_at, key, shape=trailing_shape))(jnp.arange(n))


def indexed_normal(key, n, trailing_shape=()):
    """Normal counterpart of :func:`indexed_uniform`, stable under capacity growth."""
    return jax.vmap(functools.partial(_normal_at, key, shape=trailing_shape))(jnp.arange(n))


def indexed_randint(key, n, trailing_shape, maxval):
    """``randint`` counterpart; ``maxval`` may be an array of ``trailing_shape``."""
    return jax.vmap(functools.partial(_randint_at, key, shape=trailing_shape, maxval=maxval))(
        jnp.arange(n)
    )
