"""_scatter_sigma: the six-point average of quadratic (and cubic) functions of the
post-collision velocities against a Monte Carlo over the VSS scatter, alpha 1 and 1.4."""
import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import ensemble_device
from jax_dsmc.collide.vss import scatter

rng = np.random.default_rng(0)
vi = jnp.asarray(rng.normal(size=(4, 3)))
vj = jnp.asarray(rng.normal(size=(4, 3)))
A = rng.normal(size=(3, 3)); A = A + A.T
b = rng.normal(size=3)


def quad(v):
    return jnp.einsum("na,ab,nb->n", v, A, v) + v @ b + jnp.prod(v, axis=-1) * 0.0


def cubic(v):
    return v[..., 0] ** 3 + v[..., 1] * v[..., 2] ** 2


for alpha in (1.0, 1.4):
    sp = type("SP", (), {})()
    sp.alpha = jnp.full((1, 1), alpha)
    sp.mass = jnp.ones(1)
    pi, pj = ensemble_device._scatter_sigma(vi, vj, sp)
    m = 400_000
    uc = jnp.asarray(rng.uniform(size=(m, 1)))
    ue = jnp.asarray(rng.uniform(size=(m, 1)))
    vin, vjn = scatter(vi[None], vj[None], 0, 0, sp, uc, ue)
    for name, f in (("quad", quad), ("cubic", cubic)):
        sig_i = sum(f(p) for p in pi) / 6
        sig_j = sum(f(p) for p in pj) / 6
        mc_i = jnp.mean(f(vin.reshape(-1, 3)).reshape(m, 4), 0)
        mc_j = jnp.mean(f(vjn.reshape(-1, 3)).reshape(m, 4), 0)
        se = jnp.std(f(vin.reshape(-1, 3)).reshape(m, 4), 0) / np.sqrt(m)
        print(f"alpha {alpha} {name}: max |sigma - MC| / MC se  i {float(jnp.max(jnp.abs(sig_i - mc_i) / se)):.2f}"
              f"  j {float(jnp.max(jnp.abs(sig_j - mc_j) / se)):.2f}")
