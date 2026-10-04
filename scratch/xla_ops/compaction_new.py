"""Stream compaction: the slots of the set entries of a mask, in slot order.

Every compaction in the tracker (the lanes, the culled searches, the compacted inside
check) is a prefix count of the mask (:func:`prefix_count`) followed by a lookup
(:func:`ranked_slots`).  Two exact forms, chosen per
backend at trace time -- they return the same integers:

* **binary search** (XLA:CPU): the slot of rank ``r`` is the first index whose inclusive
  count reaches ``r`` -- ``O(size log N)`` work instead of ``N`` serial scatter writes;
* **scatter** (GPU): every set slot writes its index at its rank -- one kernel, where the
  binary search's ``log N`` dependent rounds are as many launches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def ranked_slots(rank: jax.Array, first: int | jax.Array, size: int) -> jax.Array:
    """The slots of rank ``first .. first + size - 1``, ``i32[size]`` (``-1`` past the
    last): ``rank`` is the inclusive cumulative count of the set slots minus one, so a set
    slot's rank is its position among them and the slot of rank ``r`` is the first index
    where ``rank >= r``."""
    want = first + jnp.arange(size, dtype=rank.dtype)
    if jax.default_backend() == "cpu":
        return jnp.where(want <= rank[-1], lower_bound(rank, want), -1)
    n = rank.shape[0]
    is_set = jnp.diff(rank, prepend=jnp.asarray([-1], rank.dtype)) > 0
    target = jnp.where(is_set & (rank >= first) & (rank < first + size), rank - first, size)
    return (
        jnp.full((size,), -1, jnp.int32).at[target].set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )


def lower_bound(sorted_: jax.Array, want: jax.Array) -> jax.Array:
    """The first index ``i`` with ``sorted_[i] >= want`` (``len(sorted_)`` if none),
    ``i32`` shaped like ``want`` -- ``jnp.searchsorted(sorted_, want, side="left")``.

    A branch-free binary search of static depth: ``pos`` counts the entries below
    ``want`` and grows by the halving powers of two whose probe is still below it.  The
    levels unroll into straight-line XLA ops; ``searchsorted``'s ``while`` loop instead
    runs ~60 tiny ops and carry copies per call on XLA:CPU (2026-10-02 op census)."""
    n = sorted_.shape[0]
    pos = jnp.zeros(want.shape, jnp.int32)
    step = 1 << (n.bit_length() - 1)  # the largest power of two <= n
    while step:
        probe = jnp.minimum(pos + step, n) - 1
        below = (pos + step <= n) & (sorted_[probe] < want)
        pos = jnp.where(below, pos + step, pos)
        step >>= 1
    return pos


PREFIX_BLOCK = 8
"""Block width of :func:`prefix_count`'s two-level scan on XLA:CPU."""


def prefix_count(mask: jax.Array) -> jax.Array:
    """Inclusive ``int32`` prefix count of a 1-D boolean (or ``0/1`` integer) array --
    ``jnp.cumsum(mask.astype(int32))``, the same integers.  On XLA:CPU ``cumsum`` is a tree
    of reduce-windows over the whole array (~16 ns per element at ``N = 5e4``, measured by
    the SPARTA-JAX audit, 2026-10-01); prefix sums inside blocks of
    :data:`PREFIX_BLOCK`, the same recursively on the block totals, then one broadcast add
    cost about half."""
    m = mask.astype(jnp.int32)
    n = m.shape[0]
    block = PREFIX_BLOCK
    if jax.default_backend() != "cpu" or n <= 4 * block or n % block:
        return jnp.cumsum(m, dtype=jnp.int32)
    inner = jnp.cumsum(m.reshape(-1, block), axis=1, dtype=jnp.int32)
    total = inner[:, -1]
    offset = _materialise(prefix_count(total) - total)
    return (inner + offset[:, None]).reshape(n)


def _materialise(t: jax.Array) -> jax.Array:
    """``t`` stored as its own array: XLA:CPU fuses a small computed table into each
    consumer and an ``N``-sized consumer then recomputes it per element; a scatter is
    always the root of its fusion (``optimization_barrier`` is expanded away before
    fusion on XLA:CPU)."""
    return jnp.zeros_like(t).at[jnp.arange(t.shape[0], dtype=jnp.int32)].set(t)
