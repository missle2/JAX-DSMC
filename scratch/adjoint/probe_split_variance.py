"""Is `alpha`'s error bar pathwise noise or score noise?  Decompose it properly.

The main variance probe prints `sd(full)` and `sd(full - pathwise)`, which is enough
to say the score *contribution* is small but not enough to attribute the rest: the two
halves may be correlated.  This measures all three -- `sd(pathwise)`, `sd(score)` and
their correlation -- so the attribution is a measurement rather than an inference.

It matters because the two have different remedies.  Score noise is what [P2]'s
`M_s = 100` ensemble is for and what a baseline (section 11.4) attacks.  Pathwise
noise is ordinary Monte Carlo error in the forward solve and is fixed only by more
particles -- no estimator change touches it.
"""

import functools
import sys

sys.path.insert(0, ".")
import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa  # noqa: E402

TH = sa.reference_theta()

import jax.numpy as jnp  # noqa: E402

from jax_dsmc import adjoint  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide import vremax_init  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.initialize import initial_state  # noqa: E402
from jax_dsmc.state import CollideState  # noqa: E402

# Deliberately NOT imported from probe_variance: that file is a script, and importing
# it would re-run the whole study.
NAMES = ("temp", "ux", "alpha", "diam")
BYTES_PER_PARTICLE_KEY = 4290
GPU_BUDGET = 2.6e9


def case_of(n_transverse, npc=50, steps=16):
    return sa.calibrate(
        sa.case(2, npc=npc, n_steps=steps, n_x=40, transverse=n_transverse, a_max=256)
    )


def auto_chunk(cs):
    return max(1, int(GPU_BUDGET / (BYTES_PER_PARTICLE_KEY * cs.n_part)))


def value(theta, base_key, cs, scored):
    flow, sp, grid, params, static, _ = sa.build(theta, cs)
    st = initial_state(base_key, flow, grid, sp, static)
    c = num_cells(grid)
    cst = CollideState(
        vremax=vremax_init(
            jnp.full((c,), theta["temp"]),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            c,
            1,
        )
    )
    phi = functools.partial(adjoint.phi_power, component=1, power=2)  # [P2]'s T_y
    j, _ = adjoint.terminal_objective(
        st,
        cst,
        grid,
        sp,
        params,
        static,
        base_key,
        phi,
        rho=1.0,
        bc=BC.SPECULAR,
        baseline="mean",
        scored=scored,
    )
    return j


def grads(cs, scored, keys, chunk):
    fn = functools.partial(jax.grad(value), cs=cs, scored=scored)
    f = jax.jit(jax.vmap(fn, in_axes=(None, 0)))
    outs = [jax.block_until_ready(f(TH, keys[i : i + chunk])) for i in range(0, len(keys), chunk)]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs), *outs) if len(outs) > 1 else outs[0]


keys = jax.random.split(jax.random.PRNGKey(101), 256)

for nt in (8, 32):
    cs = case_of(nt)
    gp = grads(cs, False, keys, auto_chunk(cs))
    gf = grads(cs, True, keys, auto_chunk(cs))
    print(f"\n--- N = {cs.n_part}, C = {cs.n_cells}, 256 keys, phi = T_y", flush=True)
    print(
        f"{'param':>7} {'mean(full)':>13} {'sd(pathwise)':>13} {'sd(score)':>13} "
        f"{'corr':>7} {'sd(full)':>13} {'dominated by':>14}"
    )
    for name in NAMES:
        a, b = np.asarray(gp[name]), np.asarray(gf[name])
        sc = b - a
        sa_, ss = a.std(ddof=1), sc.std(ddof=1)
        corr = float(np.corrcoef(a, sc)[0, 1]) if sa_ > 0 and ss > 0 else float("nan")
        who = "pathwise" if sa_ > ss else "score"
        print(
            f"{name:>7} {b.mean():13.4e} {sa_:13.4e} {ss:13.4e} {corr:7.2f} "
            f"{b.std(ddof=1):13.4e} {who:>14}"
        )
