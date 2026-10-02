"""The RNG contract (spec section 1).

These are cheap tests guarding the most important contract in the system: if the
key derivation is wrong, gradients are silently incorrect, finite-difference
checks become pure noise, and ``jax.checkpoint`` replays a different trajectory
than the forward pass took.  None of those failures announces itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_dsmc.rng import (  # noqa: E402
    N_STAGES,
    STAGE_ACCEPT,
    STAGE_INIT,
    STAGE_NTC_ROUND,
    STAGE_PAIR,
    STAGE_SCATTER,
    key_for,
    key_for_slot,
)

BASE = jax.random.PRNGKey(20260904)
STEPS = (0, 1, 2, 7, 31)


def test_stage_ids_are_distinct_and_contiguous():
    ids = [STAGE_INIT, STAGE_NTC_ROUND, STAGE_PAIR, STAGE_ACCEPT, STAGE_SCATTER]
    assert len(set(ids)) == len(ids)
    assert all(0 <= i < N_STAGES for i in ids)


@pytest.mark.parametrize("step", STEPS)
def test_rule_4_common_random_numbers(step):
    """Same base key and step => bit-identical stream, which FD checks depend on."""
    for stage in range(N_STAGES):
        a = key_for(BASE, step, stage)
        b = key_for(BASE, step, stage)
        assert jnp.array_equal(a, b)


def test_stages_within_a_step_are_independent():
    keys = [key_for(BASE, 3, s) for s in range(N_STAGES)]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not jnp.array_equal(keys[i], keys[j])


def test_steps_within_a_stage_are_independent():
    keys = [key_for(BASE, s, STAGE_PAIR) for s in STEPS]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not jnp.array_equal(keys[i], keys[j])


def test_step_and_stage_are_independent_axes():
    """Two folds, not one: stage 3 of step 7 must not collide with stage 7 of step 3.

    A single additive fold (``fold_in(base, step + stage)``) would alias these,
    which is exactly the kind of correlation that makes two supposedly independent
    stages share a stream.
    """
    assert not jnp.array_equal(key_for(BASE, 7, 3), key_for(BASE, 3, 7))


def test_different_base_keys_give_different_streams():
    other = jax.random.PRNGKey(1)
    assert not jnp.array_equal(key_for(BASE, 0, STAGE_PAIR), key_for(other, 0, STAGE_PAIR))


@pytest.mark.parametrize("step", STEPS)
def test_rule_1_keys_are_jit_stable_under_a_traced_step(step):
    """``step`` is a loop counter, so tracing it must not change the key."""
    eager = key_for(BASE, step, STAGE_ACCEPT)
    traced = jax.jit(lambda k, s: key_for(k, s, STAGE_ACCEPT))(BASE, step)
    assert jnp.array_equal(eager, traced)


def test_slot_folding_is_independent_across_slots():
    """The attempt-slot scan needs a fresh stream per slot (section 1, last para)."""
    stage = key_for(BASE, 5, STAGE_ACCEPT)
    keys = [key_for_slot(stage, a) for a in range(8)]
    for i in range(len(keys)):
        assert not jnp.array_equal(keys[i], stage)
        for j in range(i + 1, len(keys)):
            assert not jnp.array_equal(keys[i], keys[j])


def test_slot_streams_do_not_alias_across_stages():
    """Slot ``a`` of ACCEPT must not equal slot ``a`` of SCATTER."""
    ka = key_for(BASE, 2, STAGE_ACCEPT)
    ks = key_for(BASE, 2, STAGE_SCATTER)
    for a in range(4):
        assert not jnp.array_equal(key_for_slot(ka, a), key_for_slot(ks, a))


def test_rule_2_keys_are_recomputable_not_carried():
    """Recomputing a key long after the fact reproduces it exactly.

    This is what makes ``jax.checkpoint`` rematerialisation replay the *same*
    draws on the backward pass rather than a fresh stream.
    """
    first = [key_for(BASE, s, STAGE_SCATTER) for s in range(6)]
    _burn = [jax.random.uniform(key_for(BASE, s, STAGE_PAIR), (100,)) for s in range(6)]
    again = [key_for(BASE, s, STAGE_SCATTER) for s in range(6)]
    assert all(jnp.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_split_diffuse_draws_are_the_draws():
    """``standard_diffuse_uniforms`` + ``diffuse_draws_from_uniforms`` (the wall passes
    transform only the rows that re-emit) are bit for bit the original construction:
    ``jax.random.normal`` and the Rayleigh of ``jax.random.uniform`` on the same keys.
    Guards the reliance on ``jax.random.normal``'s internal transform."""
    from jax_dsmc import sampling  # noqa: PLC0415
    from jax_dsmc.safe import safe_sqrt  # noqa: PLC0415

    for seed, n in ((0, 1), (3, 17), (11, 4096)):
        key = jax.random.PRNGKey(seed)
        k_perp, k_tan = jax.random.split(key)
        z_ref = jax.random.normal(k_tan, (n, 3), dtype=jnp.float64)
        u = jax.random.uniform(k_perp, (n,), dtype=jnp.float64)
        r_ref = safe_sqrt(-2.0 * jnp.log1p(-u))
        z, r = sampling.standard_diffuse_draws(key, n)
        np.testing.assert_array_equal(np.asarray(z), np.asarray(z_ref))
        np.testing.assert_array_equal(np.asarray(r), np.asarray(r_ref))
        u_z, u_r = sampling.standard_diffuse_uniforms(key, n)
        take = np.arange(n)[::3]
        z_rows, r_rows = sampling.diffuse_draws_from_uniforms(u_z[take], u_r[take])
        np.testing.assert_array_equal(np.asarray(z_rows), np.asarray(z_ref)[take])
        np.testing.assert_array_equal(np.asarray(r_rows), np.asarray(r_ref)[take])
