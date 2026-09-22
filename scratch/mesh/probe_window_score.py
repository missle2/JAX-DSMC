"""G3a: is the epsilon hit-timing score a cancelling window-edge effect for steady windows?

1-D closed box [0, gap]: DIFFUSE(acc = 0) wall at x = 0 (its hit has the epsilon score),
specular far wall at x = gap (the control).  N particles, Maxwellian, M steps; J = mean
impulse on the x = 0 wall per step over the last W steps.  Analytic steady state:
J = m <|v|> N / (2 gap) per ... -> dJ/dgap = -J / gap.

Value-form epsilon score, as the shape adjoint would accumulate it:
    S = sum_k mean_i  d/dgap [log p_eps(step k, particle i)] * R_i(>= k)
with R the particle's realised impulse inside the window from step k on (the ensemble's
unbiased value sample) and the derivative taken at the frozen incoming state (the state
dependence belongs to the adjoint values).  If |mean S| << sd(FD) at W = 100 the shape runs
can set epsilon = 0 for steady windowed objectives.
"""

import functools
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import boundary  # noqa: E402
from jax_dsmc.adjoint import magic  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

N = 20000
MASS = 1.0
DT = 0.05
M = 200
GAP = 1.0
FACES = boundary.Faces((boundary.BC.DIFFUSE, boundary.BC.SPECULAR))
WALL = boundary.WallParams.uniform(1, 1.0, acc=0.0)


def step(x, v, gap, key, epsilon):
    grid = GridSpec(lo=jnp.zeros(1), hi=jnp.array([gap]), n=(1,))
    x1, v1, _alive, _, logp = boundary.apply_faces(
        x, v, jnp.ones(N, bool), grid, FACES, WALL, MASS, DT, key, epsilon=epsilon
    )
    hit0 = (v[:, 0] < 0) & (v1[:, 0] > 0)  # bounced off x = 0 this step
    impulse = jnp.where(hit0, 2.0 * MASS * jnp.abs(v[:, 0]), 0.0)
    return x1, v1, logp, impulse


def _scan_step(carry, k, gap, key, epsilon):
    x, v = carry
    xs, vs = jax.lax.stop_gradient(x), jax.lax.stop_gradient(v)
    # partial derivative of the step's log p wrt gap: the state is frozen, the state's own
    # dependence on gap belongs to the adjoint values
    x1, v1, logp, impulse = step(xs, vs, gap, jax.random.fold_in(key, k), epsilon)
    return (jax.lax.stop_gradient(x1), jax.lax.stop_gradient(v1)), (jax.lax.stop_gradient(impulse), logp)


def trajectory(gap, key, epsilon):
    """Per-step impulses and per-step eps scores (partial in gap at the frozen state)."""
    kx, kv = jax.random.split(key)
    x = jax.random.uniform(kx, (N, 1), dtype=jnp.float64) * jax.lax.stop_gradient(gap)
    v = jax.random.normal(kv, (N, 3), dtype=jnp.float64)
    body = functools.partial(_scan_step, gap=gap, key=key, epsilon=epsilon)
    _, (impulses, logp) = jax.lax.scan(body, (x, v), jnp.arange(M))
    return impulses, logp  # (M, N)


def score_objective(gap, key, epsilon, W):
    impulses, logp = trajectory(gap, key, epsilon)
    window = jnp.arange(M) >= M - W
    r_step = jnp.where(window[:, None], impulses, 0.0) / W
    # R(>= k): the particle's remaining windowed reward from step k on
    future = jnp.cumsum(r_step[::-1], axis=0)[::-1]
    return jnp.sum(jnp.mean(magic(logp) * future, axis=1))


def objective(gap, key, W):
    impulses, _ = trajectory(gap, key, 0.0)
    return jnp.mean(jnp.sum(impulses[M - W :], axis=0)) / W


def main():
    seeds = 12
    out = {}
    grad_score = jax.jit(jax.grad(score_objective), static_argnums=(3,))
    obj = jax.jit(objective, static_argnums=(2,))
    for W in (1, 10, 100):
        js, fds, scores = [], [], {}
        h = 0.05
        for s in range(seeds):
            key = jax.random.PRNGKey(s)
            js.append(float(obj(GAP, key, W)))
            jp = obj(GAP + h, jax.random.PRNGKey(1000 + s), W)
            jm = obj(GAP - h, jax.random.PRNGKey(2000 + s), W)
            fds.append(float((jp - jm) / (2 * h)))
            for eps in (0.02, 0.01):
                scores.setdefault(eps, []).append(float(grad_score(GAP, key, eps, W)))
        se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a)))  # noqa: E731
        sd_fd = float(np.std(fds, ddof=1))
        row = {"J": [float(np.mean(js)), se(js)], "fd": [float(np.mean(fds)), se(fds)], "sd_fd": sd_fd,
               "analytic": -float(np.mean(js)) / GAP,
               "score": {str(e): [float(np.mean(v)), se(v), float(np.std(v, ddof=1))] for e, v in scores.items()}}
        out[W] = row
        print(f"W={W:3d}: J={row['J'][0]:.4f}±{row['J'][1]:.4f}  FD={row['fd'][0]:+.4f}±{row['fd'][1]:.4f} (sd {sd_fd:.4f})  "
              f"analytic -J/gap={row['analytic']:+.4f}  " + "  ".join(
                  f"eps-score(eps={e})={v[0]:+.4f}±{v[1]:.4f} (sd {v[2]:.4f})" for e, v in row["score"].items()), flush=True)
    path = os.path.join(os.path.dirname(__file__), "probe_window_score.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
