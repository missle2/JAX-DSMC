"""``jax_dsmc.compaction``: ``ranked_slots`` against numpy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_dsmc import compaction  # noqa: E402


@pytest.mark.parametrize("first, size", [(0, 50), (37, 400), (990, 64)])
def test_ranked_slots_are_the_set_slots_in_order(first, size):
    rng = np.random.default_rng(size)
    mask = rng.random(3000) < 0.35
    rank = compaction.prefix_count(jnp.asarray(mask)) - 1
    got = np.asarray(compaction.ranked_slots(rank, first, size))
    slots = np.flatnonzero(mask)[first : first + size]
    np.testing.assert_array_equal(got[: len(slots)], slots)
    assert (got[len(slots) :] == -1).all()
