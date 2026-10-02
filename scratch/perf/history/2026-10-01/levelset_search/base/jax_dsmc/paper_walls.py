"""JAX reference driver for P3 closed-wall experiments, Algorithm 1 splitting.

Uses disjoint Maxwell pairs, collision before transport, and nondimensional
thermal variances. Hard cell assignments and collision counts are frozen, as in
P3's wall formulation; this is not an ensemble adjoint for moving cell interfaces.
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jax_dsmc.adjoint import surrogate, trajectory_surrogate
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import GridSpec
from jax_dsmc.wall_adjoint import advance


@dataclass(frozen=True)
class Config:
    particles: int = 2000
    cells: int = 20
    dt: float = 0.05
    steps: int = 10
    epsilon: float = 0.005
    initial_std: float = 1.0
    observable: str = "couette"
    score_mode: str = "paper"
    localize: bool = False


def collision(x, v, key, cfg):
    """Disjoint same-cell Maxwell collisions; no acceptance majorant."""
    n = len(v)
    cell = jnp.clip((x[:, 0] * cfg.cells).astype(jnp.int32), 0, cfg.cells - 1)
    counts = jnp.bincount(cell, length=cfg.cells)
    ku, kr, ks = jax.random.split(key, 3)
    order = jnp.lexsort((jax.random.uniform(ku, (n,)), cell))
    sorted_cell = cell[order]
    starts = jnp.cumsum(counts) - counts
    rank = jnp.arange(n) - starts[sorted_cell]
    rate = counts * cfg.cells / n * cfg.dt
    pairs = jnp.floor(counts * rate / 2 + jax.random.uniform(kr, (cfg.cells,))).astype(jnp.int32)
    selected = (rank % 2 == 0) & (rank < 2 * pairs[sorted_cell]) & (rank + 1 < counts[sorted_cell])
    a = jnp.where(selected, order, n)
    b = jnp.where(selected, order[jnp.minimum(jnp.arange(n) + 1, n - 1)], n)
    padded = jnp.concatenate((v, jnp.zeros((1, 3))))
    va, vb = padded[a], padded[b]
    direction = jax.random.normal(ks, (n, 3), dtype=jnp.float64)
    direction /= jnp.linalg.norm(direction, axis=1, keepdims=True)
    # Guard the norm at inactive scratch pairs, where its derivative is undefined.
    rel2 = jnp.sum((va - vb) ** 2, axis=1, keepdims=True)
    rel = 0.5 * jnp.sqrt(jnp.maximum(rel2, 1e-30)) * direction
    center = 0.5 * (va + vb)
    padded = padded.at[a].set(center + rel).at[b].set(center - rel)
    invalid = jnp.sum((rate > 1) | (2 * pairs > counts))
    return padded[:n], invalid


DEFAULT_CONFIG = Config()


@functools.partial(jax.jit, static_argnames=("cfg",))
def solve(temperatures, wall_speed, key, cfg=DEFAULT_CONFIG):
    """Returns scored objective and (plain objective, invalid-event count).

    By default every particle receives randomized time. ``localize=True`` uses
    the paper's hard three-sigma proximity selection, frozen under differentiation.
    """
    if cfg.score_mode not in ("paper", "trajectory"):
        raise ValueError("score_mode must be paper or trajectory")
    if cfg.observable not in ("couette", "heat"):
        raise ValueError("observable must be couette or heat")
    n = cfg.particles
    kx, kv, kt = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (n, 1), dtype=jnp.float64)
    v = cfg.initial_std * jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    grid = GridSpec(lo=jnp.array([0.0]), hi=jnp.array([1.0]), n=(cfg.cells,))
    wall = WallParams(
        temp=temperatures,
        vwall=jnp.array([[0.0, -wall_speed, 0.0], [0.0, wall_speed, 0.0]]),
        acc=jnp.ones(2),
    )
    faces = Faces.uniform(BC.DIFFUSE, 1)

    def step(carry, k):
        x, v, logp, failures = carry
        kc, kw = jax.random.split(jax.random.fold_in(kt, k))
        v, invalid = collision(x, v, kc, cfg)
        epsilon = cfg.epsilon
        if cfg.localize:
            near = jnp.minimum(x[:, 0], 1 - x[:, 0]) < (cfg.dt + 3 * cfg.epsilon) * jnp.abs(v[:, 0])
            epsilon = jnp.where(near, cfg.epsilon, 0.0)
        x, v, _, exhausted, lp = advance(
            x, v, jnp.ones(n, bool), grid, faces, wall, K_BOLTZ, cfg.dt, kw, epsilon
        )
        logp = logp + (jnp.sum(lp) if cfg.score_mode == "trajectory" else lp)
        return (x, v, logp, failures + invalid + exhausted), None

    (x, v, logp, failures), _ = jax.lax.scan(
        jax.checkpoint(step), (x, v, jnp.zeros(n), jnp.int64(0)), jnp.arange(cfg.steps)
    )
    phi = (
        v[:, 1] ** 2
        if cfg.observable == "couette"
        else jnp.sum(v * v, axis=1) * jnp.exp(-((x[:, 0] - 0.2) ** 2))
    )
    value = jnp.mean(phi)
    scored = (
        trajectory_surrogate(value, logp[0])
        if cfg.score_mode == "trajectory"
        else surrogate(phi, logp, jnp.ones(n, bool))
    )
    return scored, (value, failures)
