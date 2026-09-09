import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
exec(open('/home/aakash/jax_dsmc/JAX-DSMC/scratch/probe_grad_claim1.py').read().split('DT, FNUM, VOL = 1.0e-3')[0])

DT, FNUM, VOL = 2.5e-5, 1.0e3, 1.0
dfd = np.array([0.5*N*(N-1.0)*SIGMA*FNUM/VOL, 0.5*N*(N-1.0)*SIGMA*DT/VOL,
                -0.5*N*(N-1.0)*SIGMA*DT*FNUM/VOL**2])
names = ["d/d(dt)", "d/d(fnum)", "d/d(volume)"]

# ===== A. single fixed configuration (seed 0, remain=0.37): zero-variance ground truth =====
d, REM = fixed_draws(0), 0.37
Je, Jd, Ju = J_cond_exp(DT, FNUM, VOL, REM, d)
truth = (float(Ju) - float(Jd)) * dfd
print(f"f = {float(f_of(DT,FNUM,VOL,REM)):.4f}  frac = {float(f_of(DT,FNUM,VOL,REM))%1:.4f}")
print(f"J(n=3) = {float(Jd):.8f}   J(n=4) = {float(Ju):.8f}   E[J] = {float(Je):.8f}")
print("exact truth   :", [f"{x: .6e}" for x in truth])

@jax.jit
def est(s):
    uc = jax.random.uniform(jax.random.PRNGKey(int(0)+s))
    return jnp.asarray(jax.grad(J_fix, argnums=(0,1,2))(DT, FNUM, VOL, REM, d, uc))
for M in [100, 1000, 10000, 100000]:
    E = jax.vmap(est)(jnp.arange(M))
    m = np.array(jnp.mean(E,0)); se = np.array(jnp.std(E,0))/np.sqrt(M)
    print(f"  M={M:7d} fix mean={m[0]: .6e} +/- {se[0]:.2e}   z_bias={(m[0]-truth[0])/se[0]: .2f}"
          f"   [fnum z={(m[1]-truth[1])/se[1]: .2f}, vol z={(m[2]-truth[2])/se[2]: .2f}]")

# also an independent FD of the exact E[J] in dt at this configuration
for rel in [1e-3, 1e-2]:
    e = DT*rel
    Jp,_,_ = J_cond_exp(DT+e,FNUM,VOL,REM,d); Jm,_,_ = J_cond_exp(DT-e,FNUM,VOL,REM,d)
    print(f"  FD of exact E[J] wrt dt (rel={rel:.0e}): {float((Jp-Jm)/(2*e)): .6e}")

# ===== B. full ensemble (remain ~ U, 200 particle configs): heavy tail =====
print("\n--- full ensemble, remain ~ U[0,1) ---")
@jax.jit
def pair(s):
    dd = fixed_draws(s % 200)
    k1,k2 = jax.random.split(jax.random.PRNGKey(7_000_000+s))
    rem = jax.random.uniform(k1); uc = jax.random.uniform(k2)
    _, jd, ju = J_cond_exp(DT, FNUM, VOL, rem, dd)
    g = jnp.asarray(jax.grad(J_fix, argnums=(0,1,2))(DT, FNUM, VOL, rem, dd, uc))
    return (ju-jd)*jnp.asarray(dfd), g
for M in [10000, 100000, 400000]:
    T,E = jax.vmap(pair)(jnp.arange(M))
    D = E-T
    tm,ts = float(jnp.mean(T[:,0])), float(jnp.std(T[:,0])/np.sqrt(M))
    em,es = float(jnp.mean(E[:,0])), float(jnp.std(E[:,0])/np.sqrt(M))
    dm,ds = float(jnp.mean(D[:,0])), float(jnp.std(D[:,0])/np.sqrt(M))
    print(f"  M={M:7d} truth={tm: .4e}+/-{ts:.1e}  fix={em: .4e}+/-{es:.1e}  bias={dm: .3e}+/-{ds:.1e} z={dm/ds: .2f}")
