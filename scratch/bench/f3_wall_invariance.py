"""F3: the chronological wall tracker reproduces the old fold path where no randomness enters.

With ``acc = 0`` a DIFFUSE face is a specular reflector that consumes no random draw, so
the branch's ``wall_adjoint.advance`` (dispatched by ``apply_faces`` whenever a DIFFUSE face
exists) and the main repo's two-stage fold path must agree to round-off on positions,
velocities, liveness and the recross count.  ``epsilon = 0``, ``acc_delta = 0``.
Diffuse re-emission itself draws different streams in the two paths and is covered by
the statistical gate F1 instead.
"""

import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_dsmc import boundary as new_boundary  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

MAIN = "/home/aakash/jax_dsmc/JAX-DSMC/jax_dsmc/boundary.py"
spec = importlib.util.spec_from_file_location("old_boundary", MAIN)
old_boundary = importlib.util.module_from_spec(spec)
sys.modules["old_boundary"] = old_boundary
spec.loader.exec_module(old_boundary)


def case(dim, n, key):
    kx, kv = jax.random.split(key)
    lo, hi = jnp.zeros(dim), jnp.ones(dim)
    grid = GridSpec(lo=lo, hi=hi, n=(4,) * dim)
    x = jax.random.uniform(kx, (n, dim), dtype=jnp.float64)
    v = 3.0 * jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    alive = jnp.arange(n) % 7 != 0
    return grid, x, v, alive


def faces_for(mod, dim):
    """Each module has its own BC enum; rules are compared by identity, so build both."""
    bcs = [mod.BC.DIFFUSE, mod.BC.DIFFUSE] + [mod.BC.PERIODIC, mod.BC.SPECULAR] * (dim - 1)
    return mod.Faces(tuple(bcs[: 2 * dim])), mod.WallParams.uniform(dim, 300.0, acc=0.0)


def main(dt=0.04, strict=True):
    worst = 0.0
    for dim in (1, 2, 3):
        for seed in range(3):
            key = jax.random.PRNGKey(100 * dim + seed)
            grid, x, v, alive = case(dim, 4096, key)
            fo, wo = faces_for(old_boundary, dim)
            fn, wn = faces_for(new_boundary, dim)
            a = old_boundary.apply_faces(x, v, alive, grid, fo, wo, K_BOLTZ, dt, key)
            b = new_boundary.apply_faces(x, v, alive, grid, fn, wn, K_BOLTZ, dt, key)
            (xa, va, aa, ra, la), (xb, vb, ab, rb, lb) = a, b
            assert np.array_equal(np.asarray(aa), np.asarray(ab)), f"alive differs D={dim}"
            dx = float(jnp.max(jnp.abs(xa - xb) * aa[:, None]))
            dv = float(jnp.max(jnp.abs(va - vb) * aa[:, None]))
            worst = max(worst, dx, dv)
            hits = int(jnp.sum(jnp.any(va != v, axis=1) & aa))
            print(
                f"D={dim} seed={seed}: max|dx|={dx:.2e} max|dv|={dv:.2e} "
                f"recrossed old={int(ra)} new={int(rb)} hits={hits} "
                f"score old={float(jnp.sum(la)):.3g} new={float(jnp.sum(lb)):.3g}"
            )
    print(f"dt={dt}: worst deviation {worst:.2e}")
    if strict:
        assert worst < 1e-12


if __name__ == "__main__":
    # dt=0.04: ~20% of particles hit a wall, none needs a second bounce -> must agree.
    main(0.04, strict=True)
    # dt=0.25: multi-bounce regime the old path could not resolve -> report only.
    main(0.25, strict=False)
