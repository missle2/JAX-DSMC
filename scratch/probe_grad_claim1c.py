import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
exec(open('/home/aakash/jax_dsmc/JAX-DSMC/scratch/probe_grad_claim1.py').read().split('DT, FNUM, VOL = 1.0e-3')[0])

DT, FNUM, VOL = 2.5e-5, 1.0e3, 1.0
dfd = np.array([0.5*N*(N-1.0)*SIGMA*FNUM/VOL,
                0.5*N*(N-1.0)*SIGMA*DT/VOL,
                -0.5*N*(N-1.0)*SIGMA*DT*FNUM/VOL**2])

@jax.jit
def pair(seed):
    d   = fixed_draws(seed)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1_000_000 + seed))
    rem = jax.random.uniform(k1); uc = jax.random.uniform(k2)
    _, Jd, Ju = J_cond_exp(DT, FNUM, VOL, rem, d)
    truth = (Ju - Jd) * jnp.asarray(dfd)                          # exact conditional derivative
    g = jax.grad(J_fix, argnums=(0,1,2))(DT, FNUM, VOL, rem, d, uc)
    est = jnp.asarray(g)
    return truth, est

names = ["d/d(dt)", "d/d(fnum)", "d/d(volume)"]
for M in [1000, 10000, 100000]:
    T, E = jax.vmap(pair)(jnp.arange(M))
    D = E - T
    print(f"\n--- M = {M} ---")
    for c in range(3):
        tm, ts = float(jnp.mean(T[:,c])), float(jnp.std(T[:,c])/np.sqrt(M))
        em, es = float(jnp.mean(E[:,c])), float(jnp.std(E[:,c])/np.sqrt(M))
        dm, ds = float(jnp.mean(D[:,c])), float(jnp.std(D[:,c])/np.sqrt(M))
        print(f" {names[c]:>12}: truth={tm: .5e}+/-{ts:.1e}  fix={em: .5e}+/-{es:.1e}  "
              f"bias(fix-truth)={dm: .3e}+/-{ds:.1e}  z={dm/max(ds,1e-300): .2f}")
