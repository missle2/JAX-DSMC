"""Counter-based uniform random numbers that fuse into the surrounding XLA kernel.

``uniform(key, counter)`` is SplitMix64 evaluated at position ``counter`` of the stream ``key``:
``z = mix64(key + (counter + 1) * golden)``, ``u = (z >> 11) * 2^-53`` in [0, 1).  It is a pure
elementwise function of integers, so XLA fuses it into whichever loop consumes it -- there is no
separate RNG kernel and no key splitting per call (``jax.random``'s threefry costs ~20 integer ops
per 32-bit word plus its own kernels, which dominated small per-step draws).

SPARTA draws from one sequential Park-Miller / Knuth stream per process; the order of draws cannot
be reproduced by a data-parallel code in any case, so only the distribution is matched.

Streams: ``step_key(seed, step, stream)`` derives an independent 64-bit key per (step, stage);
within a stage the counter enumerates the draws (slot * stride + draw index).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
TWO_M53 = 1.0 / float(2**53)


def mix64(z):
    z = (z ^ (z >> np.uint64(30))) * M1
    z = (z ^ (z >> np.uint64(27))) * M2
    return z ^ (z >> np.uint64(31))


def step_key(seed, step, stream):
    """64-bit key of (seed, step, stream); all traced or python ints."""
    z = mix64(jnp.asarray(seed, jnp.uint64) * GOLDEN + np.uint64(0x632BE59BD9B4E019))
    z = mix64(z + jnp.asarray(step, jnp.uint64) * GOLDEN)
    return mix64(z + jnp.asarray(stream, jnp.uint64) * M1)


def uniform(key, counter):
    """U[0, 1) at ``counter`` (int array) of stream ``key`` (uint64 scalar)."""
    c = jnp.asarray(counter).astype(jnp.uint64)
    z = mix64(key + (c + np.uint64(1)) * GOLDEN)
    return (z >> np.uint64(11)).astype(jnp.float64) * TWO_M53
