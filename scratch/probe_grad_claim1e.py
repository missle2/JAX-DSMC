import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
exec(open('/home/aakash/jax_dsmc/JAX-DSMC/scratch/probe_grad_claim1.py').read().split('DT, FNUM, VOL = 1.0e-3')[0])
DT, FNUM, VOL = 2.5e-5, 1.0e3, 1.0
d, REM = fixed_draws(0), 0.37
truth = (float(J_cond_exp(DT,FNUM,VOL,REM,d)[2]) - float(J_cond_exp(DT,FNUM,VOL,REM,d)[1])) * (0.5*N*(N-1.)*SIGMA*FNUM/VOL)

def J_fix_b(dt, fnum, vol, remain, draws, ucount, b):
    f = f_of(dt, fnum, vol, remain); fl = jnp.floor(f); fr = f - fl
    up = ucount < fr
    n  = fl + jnp.where(up, 1.0, 0.0)
    logP = jnp.where(up, jnp.log(jnp.clip(fr,1e-300,1.)), jnp.log(jnp.clip(1.-fr,1e-300,1.)))
    v, L = collide_with_n(jax.lax.stop_gradient(n), draws)
    p = phi(v)
    pathwise = jnp.mean(magic(L) * p)
    score_reward = jnp.mean(magic(L) * p) - b            # baselined reward for the count score
    return pathwise + magic(logP)*jax.lax.stop_gradient(score_reward) - jax.lax.stop_gradient(score_reward)

for b in [0.0, 12.0, 12.4389129]:
    E = jax.vmap(lambda s: jax.grad(J_fix_b)(DT,FNUM,VOL,REM,d,jax.random.uniform(jax.random.PRNGKey(s)),b))(jnp.arange(100000))
    m, sd = float(jnp.mean(E)), float(jnp.std(E))
    print(f" b={b:11.6f}: mean={m: .5e} +/- {sd/np.sqrt(100000):.2e}  sd={sd:.3e}  (truth {truth: .5e})")
