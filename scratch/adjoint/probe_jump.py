"""Scan J(temp) finely: is there a jump, and does anything recorded change at it?"""
import sys; sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa

th = sa.reference_theta()
key = jax.random.PRNGKey(0)
for a_max in (64, 256):
    cs = sa.case(1, npc=20, n_steps=4, n_x=8, transverse=2, a_max=a_max)
    j0, d0 = sa._jit_evaluate(th, cs, key, False)
    _, g = sa.value_and_grad(th, cs, key, False)
    ad = float(g["temp"])
    print(f"\n=== A_max={a_max}  clipped={int(jnp.sum(d0.attempts_clipped))} "
          f"maxatt={int(jnp.max(d0.max_attempts))}   AD={ad:.6e}")
    print(f"{'delta':>11} {'J':>20} {'(J-J0)/(delta*T)':>20} {'chk equal to ref':>18}")
    ref = np.asarray(d0.decision_checksum)
    for delta in np.linspace(-3e-4, 3e-4, 25):
        j, d = sa._jit_evaluate(sa.perturb(th, "temp", float(delta)), cs, key, False)
        same = np.array_equal(np.asarray(d.decision_checksum), ref)
        slope = (float(j) - float(j0)) / (delta * float(th["temp"])) if delta != 0 else float("nan")
        print(f"{delta:11.3e} {float(j):20.10f} {slope:20.6f} {str(same):>18}")
