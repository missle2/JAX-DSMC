"""Which particle, which step, and which branch produces the un-fingerprinted jump?"""
import sys; sys.path.insert(0, ".")
import dataclasses
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax_dsmc.boundary import BC
from jax_dsmc.collide import vremax_init
from jax_dsmc.grid import num_cells
from jax_dsmc.initialize import initial_state
from jax_dsmc.simulate import run
from jax_dsmc.state import CollideState
from examples import shock_adjoint as sa

cs = sa.case(1, npc=20, n_steps=4, n_x=8, transverse=2, a_max=256)
th = sa.reference_theta()
key = jax.random.PRNGKey(0)


def states(delta, steps):
    theta = sa.perturb(th, "temp", delta)
    flow, sp, grid, params, static, lam = sa.build(theta, cs)
    static = dataclasses.replace(static, n_steps=steps)
    st = initial_state(key, flow, grid, sp, static)
    C = num_cells(grid)
    csx = CollideState(vremax=vremax_init(jnp.full((C,), theta["temp"]), sp.omega[0, 0],
                                          sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0], C, 1))
    (s, _), _ = run(st, csx, grid, sp, params, static, key, bc=BC.SPECULAR,
                    emit_moments="none")
    return s


A, B = -7.5e-5, -5.0e-5   # the jump sits between these
for steps in (1, 2, 3, 4):
    sa_, sb = states(A, steps), states(B, steps)
    dv = np.abs(np.asarray(sa_.v) - np.asarray(sb.v)).max(axis=1)
    dx = np.abs(np.asarray(sa_.x) - np.asarray(sb.x)).max(axis=1)
    # expected smooth drift is O(|A-B|) ~ 2.5e-5 relative; anything far above is a jump
    big = np.where(dv > 100 * dv.mean())[0]
    print(f"steps={steps}: max|dv| {dv.max():.4e}  mean {dv.mean():.4e}  "
          f"max|dx| {dx.max():.4e}   outliers {big[:8]}")
    if len(big):
        for i in big[:4]:
            print(f"    particle {i}: v_A={np.asarray(sa_.v)[i]}  v_B={np.asarray(sb.v)[i]}")
            print(f"                x_A={np.asarray(sa_.x)[i]}  x_B={np.asarray(sb.x)[i]}")
