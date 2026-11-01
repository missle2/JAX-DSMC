"""Synthetic check of shape.membership_cotangent (2026-11-01): a 4 x 1 grid of unit cells, a
collision-increment field constant in each cell (c = 1, 3, 3, 10), hit slots placed near faces
and mid-cell; the cotangent must be +-(c_B - c_A) / (band dx) within band / 2 of a face (into a
flow cell), 0 elsewhere and at the box's faces.

    PYTHONPATH=. python3 scratch/fitgate/check_membership.py
"""

import types

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import shape
from jax_dsmc.state import GridSpec

jax.config.update("jax_enable_x64", True)
BAND = 0.1
FIELD = np.array([1.0, 3.0, 3.0, 10.0])
# (x, expected d value / dx_0): near the 0|1 face from the left (+(3 - 1)), from the right
# (lower face of cell 1: -(1 - 3) = +2), mid-cell (0), near the 1|2 face (equal cells: 0), near
# the 2|3 face (+7), near the box's lower face (skipped: 0), outside the band (0)
QUERIES = [(0.97, 2.0 / BAND), (1.03, 2.0 / BAND), (0.5, 0.0), (1.98, 0.0), (2.96, 7.0 / BAND),
           (0.02, 0.0), (0.90, 0.0)]  # fmt: skip


def hit_slots(events, survivors, dim, *, n_hits):
    del events, survivors, dim
    return jnp.arange(n_hits), jnp.int32(0)


def main():
    rng = np.random.default_rng(0)
    nodes = 64
    xs = np.concatenate([np.stack([c + rng.uniform(0.0, 1.0, nodes), rng.uniform(0, 1, nodes)], 1)
                         for c in range(4)])  # fmt: skip
    c_node = np.repeat(FIELD, nodes)
    q = np.array([[x, 0.5] for x, _ in QUERIES])
    x = np.concatenate([q, xs])
    n = x.shape[0]
    v = rng.normal(size=(n, 3))
    # the hit slots are nodes of their cells too, carrying their cell's value
    coll = np.concatenate([FIELD[np.floor(q[:, 0]).astype(int)], c_node])
    alive = np.ones(n, bool)
    rec = types.SimpleNamespace(x=jnp.asarray(x), v=jnp.asarray(v), alive=jnp.asarray(alive))
    grid = GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([4.0, 1.0]), n=(4, 1))
    orig = shape._body_hit_slots
    shape._body_hit_slots = lambda e, s, d: hit_slots(e, s, d, n_hits=len(q))  # noqa: E731
    try:
        cot = shape.membership_cotangent(
            None, jnp.asarray(alive), rec, jnp.asarray(coll, jnp.float32), jnp.zeros((n, 3)),
            grid, 128, jnp.ones(5), 4096, BAND,
        )  # fmt: skip
    finally:
        shape._body_hit_slots = orig
    got = np.asarray(cot[: len(q), 0])
    want = np.array([w for _, w in QUERIES])
    for (xq, w), g in zip(QUERIES, got, strict=True):
        print(f"x {xq:5.2f}: cotangent {g:9.4f}  expected {w:9.4f}")
    print("axis-1 (single row: box faces only) max |cot|", float(np.abs(cot[: len(q), 1]).max()))
    print("rest of the slots max |cot|", float(np.abs(cot[len(q):]).max()))
    assert np.allclose(got, want, atol=0.05 * np.abs(want).max()), (got, want)
    print("ok")


if __name__ == "__main__":
    main()
