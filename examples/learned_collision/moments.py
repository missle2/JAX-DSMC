"""Coarse-cell velocity moments from raw particle sums: the one estimator both the fine
reference data (:mod:`.reference`) and the coarse training loss read.

Per coarse cell the raw sums ``(N, S1 = sum v, S2 = sum v v^T, S3 = sum v |v|^2)`` are taken
once and the central moments formed from them with the unbiased small-sample factors, so a
fine reference (hundreds of particles per coarse cell) and a coarse run (tens) estimate the
same quantity:

* ``pm = N / (N - 1) * <c c^T>``                              (kinetic tensor per unit mass)
* ``qm = N^2 / ((N - 1)(N - 2)) * <c |c|^2> / 2``             (heat flux per unit mass density)

with ``c = v - u`` about the cell's sample mean ``u``.  ``sample.moments`` takes T, P and q
about the measured mean without these factors -- a ~5 % bias at 20 particles per cell, of
different size in the reference and the model (plan, "Moment estimators").

Everything is differentiable in ``v`` (the cell index is integer, a frozen decision).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

FIELDS = ("n", "ux", "uy", "uz", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz", "qx", "qy", "qz")
"""Moment fields, in the order of the last axis of :func:`coarse_moments`."""
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def coarse_cell(x0, lo, hi, n_coarse):
    """``i32[N]`` coarse cell of each particle along axis 0 (``n_coarse`` equal bins)."""
    frac = (x0 - lo) / (hi - lo)
    return jnp.clip(jnp.floor(frac * n_coarse).astype(jnp.int32), 0, n_coarse - 1)


def raw_sums(x0, v, alive, lo, hi, n_coarse):
    """``(N[C], S1[C, 3], S2[C, 3, 3], S3[C, 3])`` over the alive particles of each bin."""
    seg = jnp.where(alive, coarse_cell(x0, lo, hi, n_coarse), n_coarse)
    w = alive.astype(v.dtype)
    v = jnp.where(alive[:, None], v, 0.0)
    count = jax.ops.segment_sum(w, seg, n_coarse + 1)[:n_coarse]
    s1 = jax.ops.segment_sum(v, seg, n_coarse + 1)[:n_coarse]
    s2 = jax.ops.segment_sum(v[:, :, None] * v[:, None, :], seg, n_coarse + 1)[:n_coarse]
    s3 = jax.ops.segment_sum(v * jnp.sum(v**2, axis=1, keepdims=True), seg, n_coarse + 1)
    return count, s1, s2, s3[:n_coarse]


def central(count, s1, s2, s3, unbiased=True):
    """``f64[C, 13]`` (:data:`FIELDS`): the count, mean velocity, unbiased kinetic tensor
    and heat flux per unit mass density from the raw sums.  Cells with fewer than three
    particles get zero second and third moments (the factors are undefined there).
    ``unbiased=False``: the plain moments of the sums (a seed- and time-averaged reference
    profile's, whose sums are expectations already)."""
    n = jnp.maximum(count, 1.0)
    u = s1 / n[:, None]
    m2 = s2 / n[:, None, None] - u[:, :, None] * u[:, None, :]
    # sum c_i |c|^2 = S3 - 2 S2 u - u tr S2 + 2 N u |u|^2
    tr2 = jnp.trace(s2, axis1=1, axis2=2)
    uu = jnp.sum(u**2, axis=1)
    m3 = (s3 - 2.0 * jnp.einsum("cij,cj->ci", s2, u) - u * tr2[:, None]
          + 2.0 * count[:, None] * u * uu[:, None]) / n[:, None]  # fmt: skip
    ok = count >= 3.0
    f2 = jnp.where(ok, count / jnp.maximum(count - 1.0, 1.0), 0.0)
    f3 = jnp.where(ok, count**2 / jnp.maximum((count - 1.0) * (count - 2.0), 1.0), 0.0)
    if not unbiased:
        f2 = f3 = jnp.where(count > 0.0, 1.0, 0.0)
    pm = f2[:, None, None] * m2
    qm = 0.5 * f3[:, None] * m3
    pij = jnp.stack([pm[:, i, j] for i, j in _PAIRS], axis=1)
    return jnp.concatenate([count[:, None], u, pij, qm], axis=1)


def coarse_moments(state, lo, hi, n_coarse):
    """:func:`central` of :func:`raw_sums` along axis 0 of ``state.x``."""
    return central(*raw_sums(state.x[:, 0], state.v, state.alive, lo, hi, n_coarse))
