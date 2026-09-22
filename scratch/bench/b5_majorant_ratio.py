"""B5 (static half): certified vs thermal majorant on an equilibrium cell.

The attempt count is proportional to Sigma, so Sigma_cert / Sigma_thermal is the
attempt-cost multiplier and mean(q)/Sigma the acceptance fraction.  Pure CPU probe.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sort  # noqa: E402
from jax_dsmc.collide import species_params  # noqa: E402
from jax_dsmc.collide.ntc import certified_majorant, thermal_majorant  # noqa: E402
from jax_dsmc.collide.vss import collision_rate  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.species_data import ARGON  # noqa: E402
from jax_dsmc.state import GridSpec, ParticleState  # noqa: E402

gas = ARGON
print(f"{'omega':>6} {'npc':>5} {'Sig_therm':>10} {'Sig_cert':>10} {'cert/therm':>11} {'<q>/therm':>10} {'<q>/cert':>9}")
for omega in (0.5, 0.81, 1.0):
    sp = species_params(jnp.array([gas.mass]), jnp.array([[gas.diam]]), jnp.array([[omega]]),
                        jnp.array([[gas.tref]]), jnp.array([[gas.alpha]]))
    for npc in (16, 64, 256):
        cells = 64
        n = cells * npc
        grid = GridSpec(lo=jnp.zeros(1), hi=jnp.ones(1), n=(cells,))
        key = jax.random.PRNGKey(npc)
        kx, kv, kp = jax.random.split(key, 3)
        vth = jnp.sqrt(K_BOLTZ * gas.tref / gas.mass)
        state = ParticleState(jax.random.uniform(kx, (n, 1), dtype=jnp.float64),
                              vth * jax.random.normal(kv, (n, 3), dtype=jnp.float64),
                              jnp.zeros(n, jnp.int32), jnp.ones(n, bool), jnp.zeros(n))
        table = sort.build_table(state, grid)
        cert = certified_majorant(state, table, sp)
        therm = thermal_majorant(jnp.full(cells, gas.tref), sp.omega[0, 0], sp.tref[0, 0],
                                 sp.mr[0, 0], sp.prefactor[0, 0])
        i, j = jax.random.randint(kp, (2, 20000), 0, n)
        q = jax.vmap(lambda a, b: collision_rate(state.v[a], state.v[b], 0, 0, sp))(i, j)
        qm = float(jnp.mean(q))
        print(f"{omega:>6.2f} {npc:>5d} {float(therm.mean()):>10.3e} {float(cert.mean()):>10.3e} "
              f"{float((cert / therm).mean()):>11.2f} {qm / float(therm.mean()):>10.3f} {qm / float(cert.mean()):>9.3f}")
