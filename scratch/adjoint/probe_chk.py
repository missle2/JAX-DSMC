"""Is the checksum actually varying, and what differs between the two evaluations?"""
import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa

cs = sa.case(1, npc=20, n_steps=4, n_x=8, transverse=2, a_max=64)
th = sa.reference_theta()
key = jax.random.PRNGKey(0)
print("h        checksum(+h)                              checksum(-h)")
for h in (3.0e-4, 1.0e-4, 5.0e-5, 1.0e-5):
    jp, dp = sa._jit_evaluate(sa.perturb(th, "temp", h), cs, key, False)
    jm, dm = sa._jit_evaluate(sa.perturb(th, "temp", -h), cs, key, False)
    cp = np.asarray(dp.decision_checksum); cm = np.asarray(dm.decision_checksum)
    print(f"{h:.1e}  {np.array2string(cp, precision=6)}  {np.array2string(cm, precision=6)}")
    print(f"          J+ = {float(jp):.10e}   J- = {float(jm):.10e}   "
          f"clip+={int(jnp.sum(dp.attempts_clipped))} clip-={int(jnp.sum(dm.attempts_clipped))} "
          f"viol+={int(jnp.sum(dp.majorant_violations))} viol-={int(jnp.sum(dm.majorant_violations))} "
          f"maxatt+={int(jnp.max(dp.max_attempts))} maxatt-={int(jnp.max(dm.max_attempts))}")
