"""P3 section 5: host reference ensemble adjoint for 1D Maxwell inflow.

Temperatures are velocity variances, as in the nondimensional paper. This is
an optimize-then-discretize adjoint, not a custom VJP of a finite DSMC sample.
NumPy/SciPy make the retained-state reconstruction inspectable; the JAX NTC
solver is deliberately not used for the paper's disjoint-pair forward model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Config:
    n_initial: int = 2000
    cells: int = 40
    dt: float = 0.025
    steps: int = 20
    retained: int = 128
    source_samples: int = 512
    partners: int = 8
    neighbors: int = 32
    q0: float = 1.0
    source_density: float = 1000.0
    initial_std: float = 1.0
    reconstruction: str = "affine"

    def __post_init__(self):
        if self.reconstruction not in ("affine", "ballistic"):
            raise ValueError("Reconstruction must be affine or ballistic")
        if (
            min(
                self.n_initial,
                self.cells,
                self.retained,
                self.source_samples,
                self.partners,
                self.neighbors,
            )
            < 1
            or self.steps < 0
        ):
            raise ValueError("Population, mesh and sampling sizes must be positive")
        if self.dt <= 0 or self.q0 < 0 or self.source_density < 0 or self.initial_std <= 0:
            raise ValueError("Invalid physical parameters")


@dataclass
class Frame:
    points: np.ndarray
    valid: np.ndarray
    counts: np.ndarray


@dataclass
class Source:
    points: np.ndarray
    weight: float


@dataclass
class Trace:
    frames: list
    sources: list
    temperatures: np.ndarray
    injection_counts: tuple
    normalization: int
    objective: float
    config: Config
    terminal_reward: object


def reward(points):
    """P3 inflow test 3 terminal objective numerator."""
    return np.sum(points[:, 1:] ** 2, axis=1) * np.exp(-36 * (points[:, 0] - 0.5) ** 2)


def sample_flux(rng, n, temperature, side):
    temperature = np.asarray(temperature)
    if temperature.shape != (3,) or np.any(temperature <= 0) or side not in (0, 1):
        raise ValueError("Flux requires three positive variances and side 0 or 1")
    v = rng.normal(size=(n, 3)) * np.sqrt(temperature)
    v[:, 0] = (1 if side == 0 else -1) * np.sqrt(-2 * temperature[0] * np.log1p(-rng.random(n)))
    return v


def temperature_score(v, temperature):
    """Score of the NORMALIZED flux; injection counts are held fixed."""
    t = np.asarray(temperature)
    return v * v / (2 * t * t) - np.array([1.0, 0.5, 0.5]) / t


def scatter_maxwell(v, w, rng):
    direction = rng.normal(size=v.shape)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    rel = np.linalg.norm(v - w, axis=-1, keepdims=True) * direction / 2
    center = (v + w) / 2
    return center + rel, center - rel


def four_term(target_gain, partner_gain, target_loss, partner_loss):
    return target_gain + partner_gain - target_loss - partner_loss


class LocalAffine:
    """Weighted phase-space nearest-neighbor affine reconstruction.

    Coordinates use a fixed supplied scale. Fits are centered at the query;
    a tiny slope ridge handles locally rank-deficient clouds. Queries are batched
    to avoid a query-by-population distance matrix. No positivity clipping.
    """

    def __init__(self, points, values, neighbors=32, scale=None):
        self.scale = np.ones(4) if scale is None else np.asarray(scale)
        self.points = np.asarray(points) / self.scale
        self.values = np.asarray(values)
        self.k = min(neighbors, len(self.points))
        self.tree = cKDTree(self.points) if self.k else None

    def __call__(self, query):
        query = np.asarray(query) / self.scale
        result = np.zeros(len(query))
        if not self.k:
            return result
        for start in range(0, len(query), 1024):
            q = query[start : start + 1024]
            dist, index = self.tree.query(q, k=self.k)
            dist, index = dist.reshape(len(q), self.k), index.reshape(len(q), self.k)
            design = np.concatenate(
                (np.ones((len(q), self.k, 1)), self.points[index] - q[:, None]), axis=2
            )
            weights = 1 / (1 + (dist / np.maximum(dist[:, -1:], 1e-12)) ** 2)
            matrix = np.einsum("nki,nk,nkj->nij", design, weights, design)
            matrix += np.diag([0.0, 1e-12, 1e-12, 1e-12, 1e-12])
            rhs = np.einsum("nki,nk,nk->ni", design, weights, self.values[index])
            result[start : start + len(q)] = np.linalg.solve(matrix, rhs[..., None])[:, 0, 0]
        return result


def _cells(points, cfg):
    return np.clip((points[:, 0] * cfg.cells).astype(int), 0, cfg.cells - 1)


def _retain(points, cfg, rng):
    cell = _cells(points, cfg)
    counts = np.bincount(cell, minlength=cfg.cells)
    retained = np.zeros((cfg.cells, cfg.retained, 4))
    valid = np.zeros((cfg.cells, cfg.retained), bool)
    for j in range(cfg.cells):
        ids = np.flatnonzero(cell == j)
        chosen = rng.choice(ids, min(len(ids), cfg.retained), replace=False)
        retained[j, : len(chosen)] = points[chosen]
        valid[j, : len(chosen)] = True
    return Frame(retained, valid, counts)


def nominal_counts(cfg, temperatures):
    # Numerical test 3 uses dx * dt * (1000 N0) as its source prefactor.
    return tuple(
        map(
            int,
            np.rint(
                np.sqrt(np.asarray(temperatures)[:, 0] / (2 * np.pi))
                * cfg.dt
                / cfg.cells
                * cfg.source_density
                * cfg.n_initial
            ).astype(int),
        )
    )


def forward(
    cfg, temperatures, seed=0, injection_counts=None, terminal_reward=reward, initial_seed=0
):
    """Collision -> absorbing transport -> injection, with fixed birth counts.

    Pass the SAME ``injection_counts`` for all finite-difference perturbations.
    Separate per-step streams keep reservoir sampling out of forward dynamics.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.shape != (2, 3) or np.any(temperatures <= 0):
        raise ValueError("Expected positive (2, 3) boundary temperature variances")
    counts = (
        nominal_counts(cfg, temperatures) if injection_counts is None else tuple(injection_counts)
    )
    if len(counts) != 2 or any(int(n) != n or n < 0 for n in counts):
        raise ValueError("Injection counts must be two nonnegative integers")
    counts = tuple(map(int, counts))
    rng = np.random.default_rng(np.random.SeedSequence([initial_seed, 0]))
    points = np.column_stack(
        (rng.random(cfg.n_initial), rng.normal(size=(cfg.n_initial, 3)) * cfg.initial_std)
    )
    frames = [_retain(points, cfg, np.random.default_rng(np.random.SeedSequence([seed, 1, 0])))]
    sources = []
    for k in range(cfg.steps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, k]))
        cell = _cells(points, cfg)
        for j in range(cfg.cells):
            ids = rng.permutation(np.flatnonzero(cell == j))
            n = len(ids)
            rate = n * cfg.cells / cfg.n_initial * cfg.q0 * cfg.dt
            if rate > 1:
                raise ValueError("dt * rho * Q0 exceeds one; reduce dt for disjoint collisions")
            number = int(np.floor(n * rate / 2 + rng.random()))
            if 2 * number > n:
                raise ValueError(
                    "Rounded collision count exceeds disjoint-pair capacity; reduce dt"
                )
            a, b = ids[: 2 * number : 2], ids[1 : 2 * number : 2]
            points[a, 1:], points[b, 1:] = scatter_maxwell(points[a, 1:], points[b, 1:], rng)
        points[:, 0] += cfg.dt * points[:, 1]
        points = points[(points[:, 0] >= 0) & (points[:, 0] <= 1)]
        source = []
        for side, number in enumerate(counts):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3, k, side]))
            v = sample_flux(rng, number, temperatures[side], side)
            born = np.column_stack((side + cfg.dt * rng.random(number) * v[:, 0], v))
            # Retain from ALL births, including immediately absorbed births (lambda=0).
            sample_rng = np.random.default_rng(np.random.SeedSequence([seed, 4, k, side]))
            m = min(number, cfg.source_samples)
            chosen = sample_rng.choice(number, m, replace=False)
            source.append(Source(born[chosen], number / m if m else 0.0))
            points = np.concatenate((points, born[(born[:, 0] >= 0) & (born[:, 0] <= 1)]))
        sources.append(source)
        frames.append(
            _retain(points, cfg, np.random.default_rng(np.random.SeedSequence([seed, 1, k + 1])))
        )
    normalization = cfg.n_initial + cfg.steps * sum(counts)
    return Trace(
        frames,
        sources,
        temperatures.copy(),
        counts,
        normalization,
        float(np.sum(terminal_reward(points)) / normalization),
        cfg,
        terminal_reward,
    )


def _inside(field, points):
    result = np.zeros(len(points))
    mask = (points[:, 0] >= 0) & (points[:, 0] <= 1)
    result[mask] = field(points[mask])
    return result


def backward(trace, cfg=None, seed=0, terminal_reward=None):
    """One scalar backward solve yields all six source-temperature derivatives."""
    cfg = trace.config if cfg is None else cfg
    terminal_reward = trace.terminal_reward if terminal_reward is None else terminal_reward
    if cfg != trace.config:
        raise ValueError("Backward configuration must match the stored forward trace")
    rng = np.random.default_rng(seed)

    def field(p):
        return terminal_reward(p) / trace.normalization

    gradient = np.zeros((2, 3))
    scale = np.array([1 / cfg.cells, cfg.initial_std, cfg.initial_std, cfg.initial_std])
    for k in reversed(range(cfg.steps)):
        for side, source in enumerate(trace.sources[k]):
            values = _inside(field, source.points)
            gradient[side] += source.weight * np.sum(
                values[:, None] * temperature_score(source.points[:, 1:], trace.temperatures[side]),
                axis=0,
            )

        def transported(p):
            moved = p.copy()
            moved[:, 0] += cfg.dt * p[:, 1]
            return _inside(field, moved)

        frame = trace.frames[k]
        targets = frame.points[frame.valid]
        values = transported(targets)
        offset = 0
        if cfg.q0:
            for j in range(cfg.cells):
                pool = frame.points[j, frame.valid[j]]
                n = len(pool)
                if not n:
                    continue
                a = np.repeat(pool, cfg.partners, axis=0)
                partner_indices = rng.integers(n, size=n * cfg.partners)
                b = pool[partner_indices].copy()
                base = values[offset : offset + n].copy()
                ap, bp = a.copy(), b.copy()
                ap[:, 1:], bp[:, 1:] = scatter_maxwell(a[:, 1:], b[:, 1:], rng)
                correction = four_term(
                    transported(ap),
                    transported(bp),
                    np.repeat(base, cfg.partners),
                    base[partner_indices],
                )
                values[offset : offset + n] += (
                    cfg.dt
                    * cfg.q0
                    * frame.counts[j]
                    * cfg.cells
                    / cfg.n_initial
                    * correction.reshape(n, cfg.partners).mean(axis=1)
                )
                offset += n
        field = _reconstruct(
            targets,
            values,
            cfg,
            trace.normalization,
            (cfg.steps - k) * cfg.dt,
            terminal_reward,
            scale,
        )
    return gradient


def _reconstruct(targets, values, cfg, normalization, remaining, terminal_reward, scale):
    """Optionally remove the known ballistic field before the local affine fit."""

    def ballistic(p):
        moved = p.copy()
        moved[:, 0] += remaining * p[:, 1]
        return _inside(lambda q: terminal_reward(q) / normalization, moved)

    if cfg.q0 == 0:
        return ballistic
    if cfg.reconstruction == "ballistic":
        fit = LocalAffine(targets, values - ballistic(targets), cfg.neighbors, scale)

        def field(p):
            return ballistic(p) + fit(p)

        return field
    return LocalAffine(targets, values, cfg.neighbors, scale)
