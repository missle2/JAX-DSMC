"""Narrow the D=1 escape: which ingredient makes FD disagree with no flip reported?"""
import sys; sys.path.insert(0, ".")
import functools
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax_dsmc import adjoint
from jax_dsmc.boundary import BC
from jax_dsmc.collide import vremax_init
from jax_dsmc.grid import num_cells
from jax_dsmc.initialize import initial_state
from jax_dsmc.state import CollideState
from examples import shock_adjoint as sa


def value(theta, cs, key, bc, steps):
    import dataclasses
    flow, sp, grid, params, static, lam = sa.build(theta, cs)
    static = dataclasses.replace(static, n_steps=steps)
    st = initial_state(key, flow, grid, sp, static)
    C = num_cells(grid)
    csx = CollideState(vremax=vremax_init(jnp.full((C,), theta["temp"]), sp.omega[0, 0],
                                          sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0], C, 1))
    phi = functools.partial(adjoint.phi_windowed, inner=adjoint.phi_kinetic, axis=0,
                            center=cs.window_mfp * lam, width=cs.width_mfp * lam)
    j, (_, _, diag) = adjoint.terminal_objective(st, csx, grid, sp, params, static, key, phi,
                                                 rho=1.0, bc=bc, baseline="mean", scored=False)
    return j, diag


V = jax.jit(value, static_argnums=(1, 3, 4))
G = jax.jit(jax.grad(lambda *a: value(*a)[0]), static_argnums=(1, 3, 4))

cs = sa.case(1, npc=20, n_steps=4, n_x=8, transverse=2, a_max=64)
th = sa.reference_theta()
key = jax.random.PRNGKey(0)

for bc in (BC.SPECULAR, BC.PERIODIC):
    for steps in (1, 2, 4):
        ad = float(G(th, cs, key, bc, steps)["temp"])
        print(f"\n--- bc={bc.value} steps={steps}   AD d/dtemp = {ad:.10e}")
        for h in (3.0e-4, 1.0e-4, 5.0e-5, 3.7e-5, 1.0e-5):
            jp, dp = V(sa.perturb(th, "temp", h), cs, key, bc, steps)
            jm, dm = V(sa.perturb(th, "temp", -h), cs, key, bc, steps)
            fd = float(jp - jm) / (2 * h * float(th["temp"]))
            same = np.array_equal(np.asarray(dp.decision_checksum),
                                  np.asarray(dm.decision_checksum))
            d = np.asarray(dp.decision_checksum) - np.asarray(dm.decision_checksum)
            print(f"    h={h:9.2e}  FD={fd:16.8e}  rel={abs(fd-ad)/abs(ad):9.2e}  "
                  f"same_chk={same}  dchk={np.max(np.abs(d)):.3e}")
