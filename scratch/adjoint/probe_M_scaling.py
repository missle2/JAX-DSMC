"""Does `sd ~ sqrt(M)` hold here, and is that even the right comparison?

`probe_variance.py` section D measured the standard deviation against the step count and
got exponents 0.31 / 1.14 / 1.28 / 0.72 for temp / ux / alpha / diam, against [P2]'s
reported `O(sqrt(M))` (`main.tex:1502`).  Three of the four miss badly, but the sd alone
cannot settle it: [P2]'s M-sweep is a 0-D relaxation *towards equilibrium*, where the
objective saturates and the gradient magnitude stops growing, so their sd growth is pure
score-accumulation.  Ours is a developing shock -- over M = 4..32 the flow is still
forming, so `dJ/dm` itself grows with M and a steeper sd is expected on signal grounds
alone.

So measure the mean as well and report the *relative* spread `sd/|mean|`.  If that is the
quantity that goes as `sqrt(M)` -- or if it is flat -- we learn which part of the growth is
signal and which is noise.  Deliberately NOT imported from probe_variance: that file is a
script and importing it re-runs the whole study.
"""

import functools
import sys

sys.path.insert(0, ".")
import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from examples import shock_adjoint as sa  # noqa: E402
from jax_dsmc import adjoint  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide import vremax_init  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.initialize import initial_state  # noqa: E402
from jax_dsmc.state import CollideState  # noqa: E402

TH = sa.reference_theta()
NAMES = ("temp", "ux", "alpha", "diam")
BYTES_PER_PARTICLE_KEY = 4290
GPU_BUDGET = 2.6e9
STEPS = (4, 8, 16, 32)


def case_of(n_transverse, npc=50, steps=16):
    return sa.calibrate(
        sa.case(2, npc=npc, n_steps=steps, n_x=40, transverse=n_transverse, a_max=256)
    )


def value(theta, base_key, cs):
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
        st, cst, grid, sp, params, static, base_key, phi,
        rho=1.0, bc=BC.SPECULAR, baseline="mean", scored=True,
    )
    return j


def grads(cs, keys):
    # `A_max` scales with npc, not M, so the per-key memory is M-independent to first order;
    # halve on OOM anyway rather than lose the run.
    chunk = max(1, int(GPU_BUDGET / (BYTES_PER_PARTICLE_KEY * cs.n_part)))
    fn = functools.partial(jax.grad(value), cs=cs)
    f = jax.jit(jax.vmap(fn, in_axes=(None, 0)))
    while chunk >= 1:
        try:
            outs = [
                jax.block_until_ready(f(TH, keys[i : i + chunk])) for i in range(0, len(keys), chunk)
            ]
            return jax.tree.map(lambda *xs: jnp.concatenate(xs), *outs) if len(outs) > 1 else outs[0]
        except Exception as exc:  # noqa: BLE001
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
    raise RuntimeError("does not fit")


keys = jax.random.split(jax.random.PRNGKey(101), 128)
rec = {n: {"mean": [], "sd": []} for n in NAMES}

print("=== sd AND mean vs step count M   ([P2] main.tex:1502 reports sd ~ sqrt(M)) ===")
print("  D=2, N=64000, C=1280, A_max=256, 128 keys, phi = T_y\n")
print(f"  {'M':>3} " + " ".join(f"{n:>12} {'sd':>10} {'sd/|mean|':>10}" for n in NAMES))
for steps in STEPS:
    g = grads(case_of(32, steps=steps), keys)
    cells = []
    for n in NAMES:
        a = np.asarray(g[n])
        m, sd = a.mean(), a.std(ddof=1)
        rec[n]["mean"].append(m)
        rec[n]["sd"].append(sd)
        cells.append(f"{m:12.4e} {sd:10.3e} {sd / abs(m):10.3f}")
    print(f"  {steps:>3} " + " ".join(cells), flush=True)

lm = np.log(np.array(STEPS, dtype=float))
print("\n  fitted exponent q in  X ~ M^q     ([P2] predicts q = +0.5 for sd)")
print(f"  {'param':>7} {'q(sd)':>8} {'q(|mean|)':>10} {'q(sd/|mean|)':>13}")
for n in NAMES:
    sd = np.array(rec[n]["sd"])
    mu = np.abs(np.array(rec[n]["mean"]))
    q_sd = np.polyfit(lm, np.log(sd), 1)[0]
    q_mu = np.polyfit(lm, np.log(mu), 1)[0]
    print(f"  {n:>7} {q_sd:8.3f} {q_mu:10.3f} {q_sd - q_mu:13.3f}")
