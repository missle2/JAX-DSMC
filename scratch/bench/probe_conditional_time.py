"""Does the conditional-time tangent matter?  One particle, position observable.

x0 = 0.89, v = 1, dt = 0.1, eps = 0.02, specular-in-effect wall (acc = 0) at x = 1.
Exact: d/dx0 E[x_final] = 2 Phi(0.5) - 1 = 0.3829.  Three estimators, 12 keys each:
  conditional : branch as shipped (interval score + tau differentiated on its interval)
  unconditional: interval score + the raw tau (its tangent is zero: parameter-free draw)
  none        : no score at all (pathwise only)
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

jax.config.update("jax_enable_x64", True)

from jax_dsmc import adjoint, boundary, wall_adjoint  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

n = 65536
grid = GridSpec(lo=jnp.array([0.0]), hi=jnp.array([1.0]), n=(1,))
faces = boundary.Faces.uniform(boundary.BC.DIFFUSE, 1)
wall = boundary.WallParams.uniform(1, 1.0, acc=0.0)
original = wall_adjoint.conditional_time


def unconditional(raw, z, lower, upper, dt, epsilon):
    _, logp = original(raw, z, lower, upper, dt, epsilon)
    return jnp.maximum(raw, 0.0), logp


def none(raw, z, lower, upper, dt, epsilon):
    return jnp.maximum(raw, 0.0), jnp.zeros_like(raw)


def value(x, key, wall, observable):
    out = wall_adjoint._advance_impl(
        jnp.full((n, 1), x), jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (n, 1)), jnp.ones(n, bool),
        grid, faces, wall, K_BOLTZ, 0.1, key, epsilon=0.02,
    )
    phi = out[0][:, 0] if observable == "position" else out[1][:, 0]
    return adjoint.surrogate(phi, out[4], out[2])


# position after a specular-in-effect wall: continuous in tau at the hit, so the score
# has zero expectation; velocity after a diffuse wall: jumps at the hit, score required.
phi_z = np.exp(-0.5 * 0.5**2) / np.sqrt(2 * np.pi)
cases = (
    ("position, acc=0", boundary.WallParams.uniform(1, 1.0, acc=0.0), "position", 2 * ndtr(0.5) - 1),
    ("v_x, acc=1", boundary.WallParams.uniform(1, 1.0, acc=1.0), "velocity",
     (1.0 + np.sqrt(np.pi / 2)) * (-phi_z / 0.02)),
)
for label, wall_case, observable, expected in cases:
    print(f"--- {label}: exact {expected:+.4f}")
    for name, fn in (("conditional", original), ("unconditional", unconditional), ("none", none)):
        wall_adjoint.conditional_time = fn
        g = jax.jit(jax.grad(value), static_argnums=3)
        d = np.array([float(g(0.89, jax.random.PRNGKey(s), wall_case, observable)) for s in range(12)])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{name:>14}: {d.mean():+.4f} ± {se:.4f}   z = {(d.mean() - expected) / se:+.1f}")
wall_adjoint.conditional_time = original
