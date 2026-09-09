"""Does switching argon -> helium make `dJ/d(alpha)` resolvable in the SHOCK?

`examples/p2_replication` established the mechanism in 0-D homogeneous relaxation:
anisotropy relaxation is driven by `<cos^2 chi>`, whose alpha-derivative vanishes at
exactly `sqrt(2)`, and argon's `alpha = 1.40` sits 1.0 % away.  There, helium (1.26)
resolved at 4.7 sigma against argon's 0.3.

That was 0-D.  The verification target is the 2-D reflected shock, which adds transport,
a specular wall and a developing flow -- none of which the 0-D argument covers.  This
measures the same comparison where it actually matters.

Reported per gas: the ensemble mean of `dT_y/d(alpha)` with its standard error, the
matching ensemble finite difference, `|z|` between them (do the two methods agree?) and
`|AD|/SE` (is the gradient resolvable from zero at all?).  The second is the number that
was ~0 for argon and is the reason this probe exists.
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
from jax_dsmc import species_data as sd  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide import vremax_init  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.initialize import initial_state  # noqa: E402
from jax_dsmc.state import CollideState  # noqa: E402

BYTES_PER_PARTICLE_KEY = 4290
GPU_BUDGET = 2.6e9
N_KEYS = 256
H_REL = 0.06  # relative FD step; large on purpose so frozen decisions flip


def value(alpha, theta, cs, base_key, scored=True):
    th = dict(theta)
    th["alpha"] = alpha
    flow, sp, grid, params, static, _ = sa.build(th, cs)
    st = initial_state(base_key, flow, grid, sp, static)
    c = num_cells(grid)
    cst = CollideState(
        vremax=vremax_init(
            jnp.full((c,), th["temp"]), sp.omega[0, 0], sp.tref[0, 0],
            sp.mr[0, 0], sp.prefactor[0, 0], c, 1,
        )
    )
    phi = functools.partial(adjoint.phi_power, component=1, power=2)  # [P2]'s T_y
    j, _ = adjoint.terminal_objective(
        st, cst, grid, sp, params, static, base_key, phi,
        rho=1.0, bc=BC.SPECULAR, baseline="mean", scored=scored,
    )
    return j


_grad = jax.grad(value, argnums=0)


def run_gas(gas, transverse, npc, steps, keys):
    cs = sa.calibrate(sa.case(2, npc=npc, n_steps=steps, n_x=40,
                              transverse=transverse, a_max=256, gas=gas))
    theta = sa.reference_theta(gas)
    a0 = jnp.float64(gas.alpha)
    h = H_REL * float(gas.alpha)

    def one(key):
        ad = _grad(a0, theta, cs, key, True)
        fd = (value(a0 + h, theta, cs, key, False)
              - value(a0 - h, theta, cs, key, False)) / (2.0 * h)
        return ad, fd

    f = jax.jit(jax.vmap(one))
    chunk = max(1, int(GPU_BUDGET / (BYTES_PER_PARTICLE_KEY * cs.n_part)))
    while chunk >= 1:
        try:
            outs = [jax.block_until_ready(f(keys[i:i + chunk])) for i in range(0, len(keys), chunk)]
            ad = np.concatenate([np.asarray(o[0]) for o in outs])
            fd = np.concatenate([np.asarray(o[1]) for o in outs])
            return cs, ad, fd
        except Exception as exc:  # noqa: BLE001
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
    raise RuntimeError("does not fit")


keys = jax.random.split(jax.random.PRNGKey(4242), N_KEYS)
print(f"devices: {jax.devices()}\n")
print("=== dT_y/d(alpha) in the 2-D reflected shock, argon vs helium ===")
print(f"    D=2, {N_KEYS} keys, phi = T_y, FD step = {H_REL:.0%} relative\n")
print(f"  {'gas':>8} {'alpha':>6} {'N':>7} {'A_max':>6} {'AD mean':>12} {'+-SE':>10} "
      f"{'FD mean':>12} {'+-SE':>10} {'|z|':>6} {'|AD|/SE':>8} {'cond':>6}")
for transverse, npc, steps in ((8, 50, 16),):
    for gas in (sd.ARGON, sd.HELIUM):
        cs, ad, fd = run_gas(gas, transverse, npc, steps, keys)
        k = len(ad)
        am, fm = ad.mean(), fd.mean()
        ase, fse = ad.std(ddof=1) / np.sqrt(k), fd.std(ddof=1) / np.sqrt(k)
        z = abs(am - fm) / np.hypot(ase, fse)
        print(f"  {gas.name:>8} {gas.alpha:>6.2f} {cs.n_part:>7} {cs.a_max:>6} "
              f"{am:>12.4e} {ase:>10.3e} {fm:>12.4e} {fse:>10.3e} {z:>6.2f} "
              f"{abs(am) / ase:>8.1f} {gas.alpha_conditioning:>5.1f}x", flush=True)
print("\n  `|AD|/SE`: is the gradient resolvable from zero?  This is the number that was")
print("  ~0 for argon and motivated the whole gas comparison.  `|z|`: do AD and FD agree?")
