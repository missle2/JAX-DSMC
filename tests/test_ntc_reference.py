"""Count-law and conservative-envelope references, independent of trajectory AD."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.collide import ntc


def test_remainder_carry_is_chunk_invariant():
    means = jnp.array([0.3, 1.7, 2.0, 0.8, 1.2])

    def run(xs, carry):
        return jax.lax.scan(lambda r, f: ntc.remainder_count(f, r), carry, xs)

    end, counts = run(means, jnp.float64(0))
    mid, first = run(means[:2], jnp.float64(0))
    end2, second = run(means[2:], mid)
    np.testing.assert_array_equal(counts, jnp.concatenate((first, second)))
    assert end == end2
    assert counts.sum() == jnp.floor(means.sum())


def test_rounded_weak_derivative_has_distinct_integer_limits():
    # E[N^2] is piecewise linear: slopes 2n+1 and 2n-1 around integer n.
    def fn(n):
        return n.astype(jnp.float64) ** 2

    assert ntc.rounded_weak_derivative(jnp.float64(2), fn, side="right") == 5
    assert ntc.rounded_weak_derivative(jnp.float64(2), fn, side="left") == 3
    assert ntc.rounded_weak_derivative(jnp.float64(2.3), fn) == 5
    assert ntc.rounded_weak_derivative(jnp.float64(2), lambda n: jnp.float64(1)) == 0
