"""Array-only JIT implementation of the P3 retained-ensemble inlet adjoint.

This is the paper's kinetic ensemble adjoint, not a VJP of finite-population NTC.
Collision partners are uniform; nearest neighbors are used ONLY to reconstruct
its backward scalar field. Config and reward callables are static JIT arguments.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from jax_dsmc.state import static_field

jax.config.update("jax_enable_x64", True)


class Frame(NamedTuple):
    points: jax.Array
    valid: jax.Array
    counts: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trace:
    frames: Frame
    sources: jax.Array
    source_valid: jax.Array
    source_weights: jax.Array
    temperatures: jax.Array
    normalization: jax.Array
    objective: jax.Array
    invalid: jax.Array
    config: object = static_field(default=None)
    terminal_reward: object = static_field(default=None)


def reward(p):
    return jnp.sum(p[..., 1:] ** 2, axis=-1) * jnp.exp(-36 * (p[..., 0] - 0.5) ** 2)


def temperature_score(v, t):
    return v * v / (2 * t * t) - jnp.array([1.0, 0.5, 0.5]) / t


def local_affine(points, values, valid, query, neighbors, scale, tile=128):
    """Exact tiled phase-space kNN and weighted affine fit; no spatial tree.

    Scratch is bounded by query/reference tiles and k. Arithmetic remains
    O(number of queries * number of retained states); benchmark before scaling.
    Equal-distance ties follow original retained-array order.
    """
    points = points.reshape(-1, 4) / scale
    values, valid = values.reshape(-1), valid.reshape(-1)
    query = query.reshape(-1, 4) / scale
    n, nq = points.shape[0], query.shape[0]
    k = min(neighbors, n)
    if n == 0 or nq == 0:
        return jnp.zeros(nq)
    nr, nb = (n + tile - 1) // tile, (nq + tile - 1) // tile
    p = jnp.pad(points, ((0, nr * tile - n), (0, 0)))
    mask = jnp.pad(valid, (0, nr * tile - n))
    q = jnp.pad(query, ((0, nb * tile - nq), (0, 0))).reshape(nb, tile, 4)

    def batch(_, queries):
        dist = jnp.full((tile, k), jnp.inf)
        ids = jnp.zeros((tile, k), jnp.int32)

        def search(i, carry):
            best, index = carry
            candidates = lax.dynamic_slice(p, (i * tile, 0), (tile, 4))
            active = lax.dynamic_slice(mask, (i * tile,), (tile,))
            d = jnp.sum((queries[:, None] - candidates[None]) ** 2, axis=-1)
            d = jnp.where(active[None], d, jnp.inf)
            ix = jnp.broadcast_to(i * tile + jnp.arange(tile, dtype=jnp.int32), d.shape)
            distances = jnp.concatenate((best, d), axis=1)
            indices = jnp.concatenate((index, ix), axis=1)
            # top_k keeps the lower input index for equal values. Earlier tiles
            # precede later ones, so ties follow retained-array order.
            negative, order = lax.top_k(-distances, k)
            return -negative, jnp.take_along_axis(indices, order, 1)

        dist, ids = lax.fori_loop(0, nr, search, (dist, ids))
        good = jnp.isfinite(dist)
        far = jnp.maximum(jnp.max(jnp.where(good, dist, 0.0), axis=1, keepdims=True), 1e-24)
        weights = jnp.where(good, 1 / (1 + jnp.where(good, dist, 0.0) / far), 0.0)
        selected = p[ids]
        design = jnp.concatenate((jnp.ones((tile, k, 1)), selected - queries[:, None]), axis=2)
        matrix = jnp.einsum("nki,nk,nkj->nij", design, weights, design)
        empty = ~jnp.any(good, axis=1)
        matrix += jnp.diag(jnp.array([0.0, 1e-12, 1e-12, 1e-12, 1e-12]))
        matrix = matrix.at[:, 0, 0].add(empty.astype(jnp.float64))
        rhs = jnp.einsum("nki,nk,nk->ni", design, weights, values[jnp.minimum(ids, n - 1)])
        result = jnp.linalg.solve(matrix, rhs[..., None])[:, 0, 0]
        return None, jnp.where(empty, 0.0, result)

    return lax.scan(batch, None, q)[1].reshape(-1)[:nq]


def partner_indices(counts, uniform):
    """Uniform partner ranks among the OTHER valid retained states of each cell.

    ``counts`` is ``i32[C]`` valid states per cell and ``uniform`` is
    ``f64[C, R, P]``.  A self-pair has zero relative velocity and is an identically
    zero four-term sample, so admitting it biases the collision correction by
    ``O(1/R)``; ranks beyond ``counts`` are junk and must be masked by the caller.
    """
    own = jnp.arange(uniform.shape[1])[None, :, None]
    n = counts[:, None, None]
    return (own + 1 + jnp.floor(uniform * jnp.maximum(n - 1, 1)).astype(jnp.int32)) % jnp.maximum(
        n, 1
    )


def _key(key, stage, step=0, side=0):
    return random.fold_in(random.fold_in(random.fold_in(key, stage), step), side)


def _ordering(p, alive, cfg, key):
    cell = jnp.clip((p[:, 0] * cfg.cells).astype(jnp.int32), 0, cfg.cells - 1)
    cell = jnp.where(alive, cell, cfg.cells)
    counts = jnp.bincount(cell, length=cfg.cells + 1)[: cfg.cells]
    order = jnp.lexsort((random.uniform(key, alive.shape, dtype=jnp.float64), cell))
    starts = jnp.cumsum(counts) - counts
    return order, counts, starts


def _retain(p, alive, cfg, key):
    order, counts, starts = _ordering(p, alive, cfg, key)
    rank = jnp.arange(cfg.retained)[None]
    ids = order[jnp.minimum(starts[:, None] + rank, len(p) - 1)]
    valid = rank < counts[:, None]
    return Frame(jnp.where(valid[..., None], p[ids], 0.0), valid, counts)


def _scatter(v, w, key):
    direction = random.normal(key, v.shape, dtype=jnp.float64)
    direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True)
    rel = jnp.linalg.norm(v - w, axis=-1, keepdims=True) * direction / 2
    return (v + w) / 2 + rel, (v + w) / 2 - rel


@partial(jax.jit, static_argnames=("cfg", "injection_counts", "capacity", "terminal_reward"))
def forward(cfg, temperatures, key, injection_counts, capacity, terminal_reward=reward):  # noqa: PLR0915 -- keep the mathematical scan together
    """Fixed-count P3 forward scan. Invalid capacity/rates are returned, never hidden.

    All sizes/counts are static Python values. Callers must reject trace.invalid.
    Temperature values and PRNG keys may change without recompilation.
    """
    counts_birth = tuple(injection_counts)
    if capacity < cfg.n_initial or len(counts_birth) != 2 or min(counts_birth) < 0:
        raise ValueError("Invalid particle capacity or injection counts")
    t = jnp.asarray(temperatures, dtype=jnp.float64)
    if t.shape != (2, 3):
        raise ValueError("Expected (2, 3) source temperatures")
    initkey = random.key(0)  # Common initial population, independent of dynamics seeds.
    x = random.uniform(_key(initkey, 0), (cfg.n_initial, 1), dtype=jnp.float64)
    v = cfg.initial_std * random.normal(_key(initkey, 1), (cfg.n_initial, 3), dtype=jnp.float64)
    p = jnp.pad(jnp.concatenate((x, v), axis=1), ((0, capacity - cfg.n_initial), (0, 0)))
    alive = jnp.arange(capacity) < cfg.n_initial
    initial = _retain(p, alive, cfg, _key(key, 1))
    normalization = cfg.n_initial + cfg.steps * sum(counts_birth)

    def step(carry, stepno):
        p, alive, invalid = carry
        order, counts, starts = _ordering(p, alive, cfg, _key(key, 2, stepno))
        rate = counts * cfg.cells / cfg.n_initial * cfg.q0 * cfg.dt
        attempts = jnp.floor(
            counts * rate / 2 + random.uniform(_key(key, 3, stepno), (cfg.cells,))
        ).astype(jnp.int32)
        invalid |= jnp.any((rate > 1) | (2 * attempts > counts))
        ordered = p[order]
        cell = jnp.clip((ordered[:, 0] * cfg.cells).astype(jnp.int32), 0, cfg.cells - 1)
        rank = jnp.arange(capacity) - starts[cell]
        even = rank % 2 == 0
        mate = jnp.clip(jnp.arange(capacity) + jnp.where(even, 1, -1), 0, capacity - 1)
        # Each pair uses its even member's direction for both outgoing velocities.
        direction = random.normal(_key(key, 4, stepno), (capacity, 3), dtype=jnp.float64)
        direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True)
        direction = direction[jnp.where(even, jnp.arange(capacity), mate)]
        a, b = ordered[:, 1:], ordered[mate, 1:]
        out = (a + b) / 2 + jnp.where(even[:, None], 1.0, -1.0) * jnp.linalg.norm(
            a - b, axis=1, keepdims=True
        ) * direction / 2
        use = alive[order] & (rank < 2 * attempts[cell])
        ordered = ordered.at[:, 1:].set(jnp.where(use[:, None], out, a))
        p = p.at[order].set(ordered)
        p = p.at[:, 0].add(cfg.dt * p[:, 1])
        alive &= (p[:, 0] >= 0) & (p[:, 0] <= 1)
        reservoirs, masks, weights = [], [], []
        births, birth_alive = [], []
        for side, n in enumerate(counts_birth):
            vv = random.normal(_key(key, 5, stepno, side), (n, 3), dtype=jnp.float64) * jnp.sqrt(
                t[side]
            )
            u = random.uniform(_key(key, 6, stepno, side), (n,), dtype=jnp.float64)
            vv = vv.at[:, 0].set((1 - 2 * side) * jnp.sqrt(-2 * t[side, 0] * jnp.log1p(-u)))
            age = cfg.dt * random.uniform(_key(key, 7, stepno, side), (n,), dtype=jnp.float64)
            born = jnp.concatenate(((side + age * vv[:, 0])[:, None], vv), axis=1)
            m = min(n, cfg.source_samples)
            choice = jnp.argsort(random.uniform(_key(key, 8, stepno, side), (n,)))[:m]
            reservoirs.append(jnp.pad(born[choice], ((0, cfg.source_samples - m), (0, 0))))
            masks.append(jnp.arange(cfg.source_samples) < m)
            weights.append(n / m if m else 0.0)
            births.append(born)
            birth_alive.append((born[:, 0] >= 0) & (born[:, 0] <= 1))
        allp = jnp.concatenate((p, *births))
        alla = jnp.concatenate((alive, *birth_alive))
        invalid |= jnp.sum(alla) > capacity
        take = jnp.argsort(~alla, stable=True)[:capacity]
        p, alive = allp[take], alla[take]
        frame = _retain(p, alive, cfg, _key(key, 1, stepno + 1))
        return (p, alive, invalid), (
            frame,
            jnp.stack(reservoirs),
            jnp.stack(masks),
            jnp.asarray(weights),
        )

    (p, alive, invalid), (frames, sources, masks, weights) = lax.scan(
        step, (p, alive, jnp.any(~jnp.isfinite(t) | (t <= 0))), jnp.arange(cfg.steps)
    )
    frames = jax.tree.map(lambda a, b: jnp.concatenate((a[None], b)), initial, frames)
    objective = jnp.sum(jnp.where(alive, terminal_reward(p), 0.0)) / normalization
    return Trace(
        frames,
        sources,
        masks,
        weights,
        t,
        jnp.asarray(normalization),
        objective,
        invalid,
        cfg,
        terminal_reward,
    )


@partial(jax.jit, static_argnames=("cfg", "terminal_reward", "tile"))
def backward(trace, cfg, key, terminal_reward=None, tile=128):  # noqa: PLR0915 -- keep the mathematical scan together
    """Scalar P3 backward scan yielding six inlet-temperature sensitivities."""
    if trace.config is not None and trace.config != cfg:
        raise ValueError("Backward configuration must match the forward trace")
    if terminal_reward is None:
        terminal_reward = trace.terminal_reward or reward
    scale = jnp.array([1 / cfg.cells, cfg.initial_std, cfg.initial_std, cfg.initial_std])

    def ballistic(p, remaining):
        moved = p.at[..., 0].add(remaining * p[..., 1])
        return jnp.where(
            (moved[..., 0] >= 0) & (moved[..., 0] <= 1),
            terminal_reward(moved) / trace.normalization,
            0.0,
        )

    last = trace.frames.points[-1].reshape(-1, 4)
    terminal = terminal_reward(last) / trace.normalization

    def step(carry, k):
        nextvalues, gradient = carry
        nextp = trace.frames.points[k + 1].reshape(-1, 4)
        nextmask = trace.frames.valid[k + 1].reshape(-1)

        def field(p):
            shape = p.shape[:-1]
            flat = p.reshape(-1, 4)
            remaining = (cfg.steps - k - 1) * cfg.dt

            def reconstructed(_):
                if cfg.q0 == 0:
                    return ballistic(flat, remaining)
                val = nextvalues
                if cfg.reconstruction == "ballistic":
                    val = val - ballistic(nextp, remaining)
                result = local_affine(nextp, val, nextmask, flat, cfg.neighbors, scale, tile)
                if cfg.reconstruction == "ballistic":
                    result += ballistic(flat, remaining)
                return result

            result = lax.cond(
                k == cfg.steps - 1,
                lambda _: terminal_reward(flat) / trace.normalization,
                reconstructed,
                operand=None,
            )
            return jnp.where((flat[:, 0] >= 0) & (flat[:, 0] <= 1), result, 0.0).reshape(shape)

        source = trace.sources[k]
        score = temperature_score(source[..., 1:], trace.temperatures[:, None])
        gradient += trace.source_weights[k, :, None] * jnp.sum(
            jnp.where(trace.source_valid[k, ..., None], field(source)[..., None] * score, 0.0),
            axis=1,
        )

        def transported(p):
            return field(p.at[..., 0].add(cfg.dt * p[..., 1]))

        targets = trace.frames.points[k]
        valid = trace.frames.valid[k]
        base = transported(targets)
        values = base
        if cfg.q0:
            n = valid.sum(axis=1)
            u = random.uniform(_key(key, 0, k), (cfg.cells, cfg.retained, cfg.partners))
            ids = partner_indices(n, u)
            b = targets[jnp.arange(cfg.cells)[:, None, None], ids]
            a = jnp.broadcast_to(targets[:, :, None], b.shape)
            av, bv = _scatter(a[..., 1:], b[..., 1:], _key(key, 1, k))
            ap, bp = a.at[..., 1:].set(av), b.at[..., 1:].set(bv)
            correction = (
                transported(ap)
                + transported(bp)
                - base[..., None]
                - base[jnp.arange(cfg.cells)[:, None, None], ids]
            )
            pairable = (n >= 2)[:, None]
            values = base + jnp.where(
                pairable,
                cfg.dt
                * cfg.q0
                * trace.frames.counts[k, :, None]
                * cfg.cells
                / cfg.n_initial
                * correction.mean(axis=-1),
                0.0,
            )
        return (jnp.where(valid, values, 0.0).reshape(-1), gradient), None

    (_, gradient), _ = lax.scan(
        step, (terminal, jnp.zeros((2, 3))), jnp.arange(cfg.steps - 1, -1, -1)
    )
    return jnp.where(trace.invalid, jnp.nan, gradient)
