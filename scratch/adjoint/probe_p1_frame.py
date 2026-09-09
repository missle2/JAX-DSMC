"""Does [P1]'s analytic A describe OUR scatter at Maxwell molecules?

[P1] assumes `delta sigma = 0`: the post-collision direction is drawn
independently of the pre-collision velocities.  SPARTA's scatter draws it in the
frame ALIGNED TO v_r, so at frozen randomness the map v_r -> v_r' is a rotation
that depends on v_r's direction.  Both sample the same distribution; they have
different pathwise Jacobians.
"""
import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax import random
from jax_dsmc.collide import species_params
from jax_dsmc.collide.vss import scatter

MASS = 6.63e-26
sp = species_params(jnp.array([MASS]), jnp.array([[4.11e-10]]),
                    jnp.array([[1.0]]), jnp.array([[273.15]]), jnp.array([[1.0]]))


def p1_matrix(s_hat, a_hat):
    p = jnp.outer(s_hat, a_hat); eye = jnp.eye(3)
    return 0.5 * jnp.block([[eye + p, eye - p], [eye - p, eye + p]])


def ours(vi, vj, uc, ue):
    return scatter(vi, vj, 0, 0, sp, uc, ue)


def labframe(vi, vj, s_hat):
    """[P1]'s own collision: sigma drawn in the LAB frame, independent of v_r."""
    vcm = 0.5 * (vi + vj)
    vrm = jnp.linalg.norm(vi - vj)
    return vcm + 0.5 * vrm * s_hat, vcm - 0.5 * vrm * s_hat


def jac(f, vi, vj, *a):
    j = jax.jacobian(lambda x, y: f(x, y, *a), argnums=(0, 1))(vi, vj)
    top = jnp.concatenate([j[0][0], j[1][0]], axis=1)
    bot = jnp.concatenate([j[0][1], j[1][1]], axis=1)
    return jnp.concatenate([top, bot], axis=0)


key = random.PRNGKey(0)
for t in range(3):
    k1, k2, k3, k4 = random.split(random.fold_in(key, t), 4)
    vi = 300.0 * random.normal(k1, (3,)); vj = 300.0 * random.normal(k2, (3,))
    uc = random.uniform(k3); ue = random.uniform(k4)
    a_hat = (vi - vj) / jnp.linalg.norm(vi - vj)

    vin, vjn = ours(vi, vj, uc, ue)
    s_ours = (vin - vjn) / jnp.linalg.norm(vin - vjn)
    A_ours = p1_matrix(s_ours, a_hat)
    Jo = jac(ours, vi, vj, uc, ue)

    Jl = jac(labframe, vi, vj, s_ours)
    A_lab = p1_matrix(s_ours, a_hat)

    print(f"\n--- sample {t}")
    print(f"  |J_sparta - A|/|A|   = {np.linalg.norm(Jo - A_ours) / np.linalg.norm(A_ours):.3e}")
    print(f"  |J_labframe - A|/|A| = {np.linalg.norm(Jl - A_lab) / np.linalg.norm(A_lab):.3e}")
    print(f"  rank J_sparta = {np.linalg.matrix_rank(np.asarray(Jo))}   "
          f"rank J_lab = {np.linalg.matrix_rank(np.asarray(Jl))}   rank A = {np.linalg.matrix_rank(np.asarray(A_lab))}")
    print(f"  |v_r| preserved: sparta {float(jnp.linalg.norm(vin-vjn)/jnp.linalg.norm(vi-vj)):.12f}")
