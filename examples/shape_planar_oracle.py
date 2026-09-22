"""Shape-derivative oracle: the collisionless planar impulse (geometry.averaged_planar_impulse).

Particles at distance u * gap from a specular wall (DIFFUSE face, acc = 0) at x = 0, all
approaching it with speed s; the far face absorbs.  J = mean impulse on the wall per
particle over one step of length T.  Analytic: F = 2 m s min(T s / gap, 1); dF/dgap =
-2 m s (T s / gap^2) in the unsaturated branch.  Compares:
  (i)   jax.grad through the forward at frozen hits (epsilon = 0)      -> misses the hit set
  (ii)  jax.grad with the epsilon-regularised hit + branch score, reward-weighted (P3 s4)
  (iii) the ensemble form: value-weighted branch score (value = impulse here)
  (iv)  ensemble finite difference over independent keys
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import boundary, geometry  # noqa: E402
from jax_dsmc.adjoint import magic  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

N = 20000
T = 0.5
SPEED = 1.0
MASS = 1.0


def forward(gap, key, epsilon):
    """Returns (J, logp per particle, impulse per particle)."""
    u = jax.random.uniform(jax.random.fold_in(key, 1), (N, 1), dtype=jnp.float64)
    x = u * gap
    v = jnp.tile(jnp.array([[-SPEED, 0.0, 0.0]]), (N, 1))
    grid = GridSpec(lo=jnp.zeros(1), hi=jnp.array([gap]), n=(1,))
    faces = boundary.Faces((boundary.BC.DIFFUSE, boundary.BC.OUTFLOW))
    wall = boundary.WallParams.uniform(1, 1.0, acc=0.0)
    _x1, v1, _alive, _, logp = boundary.apply_faces(
        x, v, jnp.ones(N, bool), grid, faces, wall, MASS, T, key, epsilon=epsilon
    )
    impulse = MASS * (v[:, 0] - v1[:, 0])  # momentum given to the wall (positive: pushes -x)
    return jnp.mean(impulse), logp, impulse


def estimators(gap, key, epsilon):
    # (i) pathwise at frozen hits
    g_frozen = jax.grad(lambda g: forward(g, key, 0.0)[0])(gap)

    # (ii)/(iii): score-weighted -- reward-weighted DiCE (P3 s4 with terminal reward) and the
    # ensemble value-weighted form coincide on a single-step, single-hit oracle
    def scored(g):
        _j, logp, impulse = forward(g, key, epsilon)
        return jnp.mean(magic(logp) * impulse)

    g_scored = jax.grad(scored)(gap)
    return float(g_frozen), float(g_scored)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "gradient_bench", "results", "shape_planar_oracle.json"
        ),
    )
    args = ap.parse_args(argv)
    rows = {}
    gap = 1.3
    per_particle = geometry.averaged_planar_impulse(
        jnp.array([[-SPEED, 0.0, 0.0]]), jnp.array([1.0, 0.0, 0.0]), gap, T, MASS, 1.0
    )
    analytic_J = float(-per_particle[0, 0] * T)  # force * T = impulse per particle
    analytic_dJ = float(
        jax.grad(
            lambda g: (
                -geometry.averaged_planar_impulse(
                    jnp.array([[-SPEED, 0.0, 0.0]]), jnp.array([1.0, 0.0, 0.0]), g, T, MASS, 1.0
                )[0, 0]
                * T
            )
        )(gap)
    )
    print(
        f"analytic: J = {analytic_J:.5f}  dJ/dgap = {analytic_dJ:.5f}   (hit fraction {T * SPEED / gap:.3f})"
    )
    for epsilon in (0.02, 0.01, 0.005):
        fr, sc, fd, js = [], [], [], []
        for seed in range(args.seeds):
            key = jax.random.PRNGKey(seed)
            a, b = estimators(gap, key, epsilon)
            fr.append(a)
            sc.append(b)
            h = 0.05
            jp = forward(gap + h, jax.random.PRNGKey(1000 + seed), epsilon)[0]
            jm = forward(gap - h, jax.random.PRNGKey(2000 + seed), epsilon)[0]
            fd.append(float((jp - jm) / (2 * h)))
            js.append(float(forward(gap, key, epsilon)[0]))
        se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))  # noqa: E731
        print(
            f"eps={epsilon}: J={np.mean(js):.5f}±{se(js):.5f}  frozen-hit grad={np.mean(fr):+.5f}±{se(fr):.5f}  "
            f"score-weighted={np.mean(sc):+.5f}±{se(sc):.5f}  ensemble FD={np.mean(fd):+.5f}±{se(fd):.5f}"
        )
        rows[str(epsilon)] = {
            "J": [np.mean(js), se(js)],
            "frozen": [np.mean(fr), se(fr)],
            "scored": [np.mean(sc), se(sc)],
            "fd": [np.mean(fd), se(fd)],
        }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "analytic_J": analytic_J,
                "analytic_dJ_dgap": analytic_dJ,
                "seeds": args.seeds,
                "rows": rows,
            },
            fh,
            indent=1,
        )


if __name__ == "__main__":
    main()
