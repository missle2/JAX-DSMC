"""Does the section-11.4 baseline actually cut the score-term variance here?

Section 11.4 measured 84x on the count score in isolation.  This asks the same
question of the full shock estimator, where the reward is a windowed kinetic
energy rather than a synthetic scalar.
"""
import sys; sys.path.insert(0, ".")
import functools, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa
from jax_dsmc import adjoint

NK = 512
th = sa.reference_theta()
cs = sa.case(1)
keys = jax.random.split(jax.random.PRNGKey(101), NK)


def value_b(theta, base_key, baseline):
    """`sa.evaluate` with the baseline overridden."""
    import jax_dsmc.adjoint as A
    flow, sp, grid, params, static, lam = sa.build(theta, cs)
    from jax_dsmc.boundary import BC
    from jax_dsmc.collide import vremax_init
    from jax_dsmc.grid import num_cells
    from jax_dsmc.initialize import initial_state
    from jax_dsmc.state import CollideState
    st = initial_state(base_key, flow, grid, sp, static)
    C = num_cells(grid)
    csx = CollideState(vremax=vremax_init(jnp.full((C,), theta["temp"]), sp.omega[0, 0],
                                          sp.tref[0, 0], sp.mr[0, 0], sp.prefactor[0, 0], C, 1))
    phi = functools.partial(A.phi_windowed, inner=A.phi_kinetic, axis=0,
                            center=cs.window_mfp * lam, width=cs.width_mfp * lam)
    j, _ = A.terminal_objective(st, csx, grid, sp, params, static, base_key, phi,
                                rho=1.0, bc=BC.SPECULAR, baseline=baseline, scored=True)
    return j


for baseline in (None, "mean"):
    f = jax.jit(jax.vmap(functools.partial(jax.grad(value_b), baseline=baseline),
                         in_axes=(None, 0)))
    t0 = time.time()
    CH = 128
    parts = [jax.block_until_ready(f(th, keys[i:i + CH])) for i in range(0, NK, CH)]
    g = jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)
    print(f"\nbaseline = {baseline!r}   ({time.time()-t0:.0f}s)")
    for name in ("diam", "omega", "alpha", "temp", "nrho"):
        a = np.asarray(g[name]); m, s = a.mean(), a.std(ddof=1) / np.sqrt(NK)
        print(f"   {name:>6}: {m:14.6e} +- {s:9.2e}   sd/|mean| per seed = "
              f"{a.std(ddof=1)/abs(m):8.1f}   SE {100*s/abs(m):6.1f} %")
