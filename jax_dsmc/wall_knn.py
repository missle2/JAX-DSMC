"""Time-pooled k-nearest-neighbour wall gradients (2026-11-05, the fit lab's gate 2).

The shape adjoint's pathwise term needs ``grad_x lambda`` at each body hit's post-move state.
The per-cell, per-step fits (``ensemble_device.node_gradients``) fit one step's few hundred
nodes per near-wall cell and return a cell-scale slope; next to the wall lambda varies on a
centimetre scale (the dense cold layer), and the fit lab (scratch/fitlab) measured the
resulting bias (drag cos 2 theta +24 %, heat +37 %).  The estimator here pools the near-wall
nodes of EVERY captured step (unbiased for the pathwise sum when the flow's statistics are
stationary over the window) and fits each hit from its ``k`` nearest nodes in a scaled
position-velocity metric, so the neighbourhood is centimetres wide in the dense layer and
velocity-conditioned (the velocity mix near a wall changes with the distance and otherwise
reads as a slope).

Generic in the coordinates: the caller supplies, per node and per query,

* ``z``: the metric coordinates (already divided by their scales), the first ``n_pos`` of
  them positional (wall distance, the wall foot point ...) -- the tiling uses only these;
* ``feat``: the regression features (physical units; the local fit is linear in them);
* per query, ``jac``: ``d feat / d x`` (``[F, D]``), mapping the fitted slopes to
  ``grad_x lambda``;

and ``key`` (int64: step x slot) so a query never uses its own node.

The search is exact kNN by tiles: the nodes are sorted by a tile of their positional
coordinates (``tile`` scaled units per side); the queries are sorted by tile and cut into
blocks of ``block_q`` from one tile; each block scans its own and the ``3^n_pos``
neighbouring tiles' nodes in chunks of ``block_n`` (dense block x chunk distances, a running
``lax.top_k``).  Exact whenever the k-th neighbour lies within one tile of the query in every
positional coordinate -- :func:`knn` returns the k-th distance, so the caller can check it.
The local fit: box weights over the k neighbours, each feature scaled by its range among
them, a ridge ``ridge`` relative to the design's mean diagonal, and a condition guard (above
``guard`` the fit drops to the intercept and the positional features: a slow particle's own
trajectory can fill a neighbourhood and make the velocity terms collinear with d).

No nested function definitions (module-level functions bound with ``functools.partial``).
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

_INF = jnp.inf


def tile_grid(z_pos_lo, z_pos_hi, tile):
    """Static tile grid over the positional box ``[lo, hi]`` (numpy, host): ``(lo, n)``."""
    lo = np.floor(np.asarray(z_pos_lo, float) / tile) * tile
    n = np.maximum(np.ceil((np.asarray(z_pos_hi, float) - lo) / tile).astype(int), 1)
    return lo, tuple(int(a) for a in n)


def tile_of(z_pos, lo, n, tile):
    """``(i32[P] linear tile, i32[P, n_pos] tile coordinates)`` of positional coordinates."""
    c = jnp.clip(jnp.floor((z_pos - jnp.asarray(lo)) / tile).astype(jnp.int32), 0,
                 jnp.asarray(n, jnp.int32) - 1)  # fmt: skip
    strides = np.cumprod((1, *tuple(n[:0:-1])))[::-1].astype(np.int32)
    return jnp.sum(c * jnp.asarray(strides), axis=1), c


def _tile_ranges(tiles_sorted, n_tiles):
    """``(start, count)`` per tile of a tile-sorted array (``i32[n_tiles]`` each)."""
    ids = jnp.arange(n_tiles, dtype=tiles_sorted.dtype)
    start = jnp.searchsorted(tiles_sorted, ids, side="left").astype(jnp.int32)
    end = jnp.searchsorted(tiles_sorted, ids, side="right").astype(jnp.int32)
    return start, end - start


def _merge(best_d, best_i, d, i, k):
    """The ``k`` smallest of ``(best_d, d)`` per row, with their indices."""
    cat_d = jnp.concatenate((best_d, d), axis=1)
    cat_i = jnp.concatenate((best_i, i), axis=1)
    neg, pos = lax.top_k(-cat_d, k)
    return -neg, jnp.take_along_axis(cat_i, pos, axis=1)


def _chunk_body(c, carry, zq, keyq, qmask, z_n, key_n, start, count, block_n, k):
    best_d, best_i = carry
    j = start + c * block_n + jnp.arange(block_n, dtype=jnp.int32)
    valid = (c * block_n + jnp.arange(block_n)) < count
    jj = jnp.clip(j, 0, z_n.shape[0] - 1)
    zn = z_n[jj]
    d = jnp.sum((zq[:, None, :] - zn[None, :, :]) ** 2, axis=-1)
    bad = (~valid)[None, :] | (key_n[jj][None, :] == keyq[:, None]) | (~qmask)[:, None]
    d = jnp.where(bad, _INF, d)
    return _merge(best_d, best_i, d, jnp.broadcast_to(jj[None, :], d.shape), k)


def _block_search(b, tables, z_n, key_n, zq_s, keyq_s, k, block_q, block_n, offsets, n):
    """The k nearest nodes of query block ``b`` (one tile's queries)."""
    (blk_tile, blk_start, blk_len, n_start, n_count) = tables
    tile = blk_tile[b]
    qi = blk_start[b] + jnp.arange(block_q, dtype=jnp.int32)
    qmask = jnp.arange(block_q) < blk_len[b]
    qi = jnp.clip(qi, 0, zq_s.shape[0] - 1)
    zq, keyq = zq_s[qi], keyq_s[qi]
    n_arr = jnp.asarray(n, jnp.int32)
    strides = jnp.asarray(np.cumprod((1, *tuple(n[:0:-1])))[::-1].astype(np.int32))
    coord = (tile // strides) % n_arr
    best_d = jnp.full((block_q, k), _INF)
    best_i = jnp.full((block_q, k), -1, jnp.int32)
    for off in offsets:  # static: the 3^n_pos neighbouring tiles
        nc = coord + jnp.asarray(off, jnp.int32)
        inside = jnp.all((nc >= 0) & (nc < n_arr))
        nt = jnp.sum(jnp.clip(nc, 0, n_arr - 1) * strides)
        start = n_start[nt]
        count = jnp.where(inside, n_count[nt], 0)
        trips = (count + block_n - 1) // block_n
        body = functools.partial(_chunk_body, zq=zq, keyq=keyq, qmask=qmask, z_n=z_n,
                                 key_n=key_n, start=start, count=count, block_n=block_n,
                                 k=k)  # fmt: skip
        best_d, best_i = lax.fori_loop(0, trips, body, (best_d, best_i))
    return best_d, best_i


@functools.partial(jax.jit, static_argnames=("n", "tile", "k", "block_q", "block_n", "n_pos"))
def knn(z_n, key_n, z_q, key_q, q_valid, lo, n, tile, k, block_q=128, block_n=2048, n_pos=2):
    """Exact (within one tile) k nearest nodes of every valid query: ``(i32[Q, k] node
    indices (-1: none), f64[Q, k] squared metric distances)``, in the queries' order."""
    n_tiles = int(np.prod(n))
    tn, _ = tile_of(z_n[:, :n_pos], lo, n, tile)
    order_n = jnp.argsort(tn)
    z_ns, key_ns, tn_s = z_n[order_n], key_n[order_n], tn[order_n]
    n_start, n_count = _tile_ranges(tn_s, n_tiles)
    tq, _ = tile_of(z_q[:, :n_pos], lo, n, tile)
    tq = jnp.where(q_valid, tq, n_tiles)  # invalid queries sort last, in no block
    order_q = jnp.argsort(tq)
    tq_s = tq[order_q]
    q_start, q_count = _tile_ranges(tq_s, n_tiles)
    # blocks: ceil(count / block_q) per tile, laid out by a cumulative sum
    n_blk = (q_count + block_q - 1) // block_q
    blk_end = jnp.cumsum(n_blk)
    max_blocks = z_q.shape[0] // block_q + n_tiles
    b = jnp.arange(max_blocks, dtype=jnp.int32)
    blk_tile = jnp.clip(jnp.searchsorted(blk_end, b, side="right"), 0, n_tiles - 1)
    j = b - (blk_end[blk_tile] - n_blk[blk_tile])
    blk_start = q_start[blk_tile] + j * block_q
    blk_len = jnp.where(b < blk_end[-1], jnp.clip(q_count[blk_tile] - j * block_q, 0, block_q), 0)
    offsets = tuple(itertools.product((-1, 0, 1), repeat=n_pos))
    tables = (blk_tile, blk_start, blk_len, n_start, n_count)
    search = functools.partial(_block_search, tables=tables, z_n=z_ns, key_n=key_ns,
                               zq_s=z_q[order_q], keyq_s=key_q[order_q], k=k, block_q=block_q,
                               block_n=block_n, offsets=offsets, n=n)  # fmt: skip
    d_blk, i_blk = lax.map(search, b)
    # back to the sorted-query order, then to the caller's
    qpos = blk_start[:, None] + jnp.arange(block_q)[None, :]
    live = jnp.arange(block_q)[None, :] < blk_len[:, None]
    qpos = jnp.where(live, qpos, z_q.shape[0])
    q = z_q.shape[0]
    d_sorted = jnp.full((q, k), _INF).at[qpos.ravel()].set(d_blk.reshape(-1, k), mode="drop")
    i_sorted = jnp.full((q, k), -1, jnp.int32).at[qpos.ravel()].set(i_blk.reshape(-1, k),
                                                                   mode="drop")  # fmt: skip
    inv = jnp.zeros(q, jnp.int32).at[order_q].set(jnp.arange(q, dtype=jnp.int32))
    i_out = i_sorted[inv]
    i_out = jnp.where(i_out >= 0, order_n[jnp.clip(i_out, 0, None)], -1)
    return i_out, d_sorted[inv]


def _solve_one(ata, aty, n_red, guard):
    """Full fit, or the reduced (intercept + positional) one when ill-conditioned."""
    beta = jnp.linalg.solve(ata, aty)
    f = ata.shape[0]
    red = jnp.arange(f) < n_red
    a_red = jnp.where(red[:, None] & red[None, :], ata, jnp.eye(f))
    beta_red = jnp.where(red, jnp.linalg.solve(a_red, jnp.where(red, aty, 0.0)), 0.0)
    bad = jnp.linalg.cond(ata) > guard
    return jnp.where(bad, beta_red, beta), bad


@functools.partial(jax.jit, static_argnames=("n_pos", "chunk"))
def local_gradients(idx, feat_n, lam_n, feat_q, jac_q, n_pos, ridge=1e-4, guard=1e3, chunk=4096):
    """``(f64[Q, D] grad_x lambda, bool[Q] guarded)`` from each query's neighbours ``idx``:
    lambda linear in the features about the query, box weights, each feature scaled by its
    range among the neighbours, ridge relative to the mean diagonal, the condition guard."""
    q = idx.shape[0]
    nf = feat_n.shape[1]
    out_g = []
    out_b = []
    for a in range(0, q, chunk):  # static chunks: (chunk, k, F) at a time
        ii = idx[a : a + chunk]
        live = ii >= 0
        jj = jnp.clip(ii, 0, None)
        rel = feat_n[jj] - feat_q[a : a + chunk, None, :]
        scale = jnp.maximum(jnp.max(jnp.abs(rel) * live[..., None], axis=1, keepdims=True),
                            1e-30)  # fmt: skip
        x = jnp.concatenate((jnp.ones((*rel.shape[:2], 1)), rel / scale), axis=2)
        x = x * live[..., None]
        y = jnp.where(live, lam_n[jj], 0.0)
        ata = jnp.einsum("qki,qkj->qij", x, x)
        diag = jnp.trace(ata, axis1=1, axis2=2) / (nf + 1)
        ata = ata + ridge * diag[:, None, None] * jnp.eye(nf + 1)[None]
        aty = jnp.einsum("qki,qk->qi", x, y)
        solve = functools.partial(_solve_one, n_red=1 + n_pos, guard=guard)
        beta, bad = jax.vmap(solve)(ata, aty)
        slopes = beta[:, 1:] / scale[:, 0, :]  # d lambda / d feature
        out_g.append(jnp.einsum("qf,qfd->qd", slopes, jac_q[a : a + chunk]))
        out_b.append(bad)
    return jnp.concatenate(out_g), jnp.concatenate(out_b)
