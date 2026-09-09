import jax, jax.numpy as jnp, numpy as np

jax.config.update("jax_enable_x64", True)
exec(
    open("/home/aakash/jax_dsmc/JAX-DSMC/scratch/probe_grad_claim1.py")
    .read()
    .split("DT, FNUM, VOL = 1.0e-3")[0]
)

DT, FNUM, VOL = 2.5e-5, 1.0e3, 1.0


def draws_for(s):
    return fixed_draws(s)


# ---- ensemble ground truth: d/dtheta E[J], expectation over remain~U[0,1) and draws ----
@jax.jit
def truth_one(seed):
    d = fixed_draws(seed)
    k = jax.random.PRNGKey(1_000_000 + seed)
    rem = jax.random.uniform(k)  # spec seeds remain ~ U[0,1)
    _, Jd, Ju = J_cond_exp(DT, FNUM, VOL, rem, d)
    dfd_dt = 0.5 * N * (N - 1.0) * SIGMA * FNUM / VOL  # df/d(dt)
    dfd_fnum = 0.5 * N * (N - 1.0) * SIGMA * DT / VOL
    dfd_vol = -0.5 * N * (N - 1.0) * SIGMA * DT * FNUM / VOL**2
    Jexp, _, _ = J_cond_exp(DT, FNUM, VOL, rem, d)
    return jnp.array([(Ju - Jd) * dfd_dt, (Ju - Jd) * dfd_fnum, (Ju - Jd) * dfd_vol]), Jexp


@jax.jit
def est_one(seed):
    d = fixed_draws(seed)
    k = jax.random.PRNGKey(1_000_000 + seed)
    k1, k2 = jax.random.split(k)
    rem = jax.random.uniform(k1)
    uc = jax.random.uniform(k2)
    g = jax.grad(J_fix, argnums=(0, 1, 2))(DT, FNUM, VOL, rem, d, uc)
    return jnp.array([g[0], g[1], g[2]])


@jax.jit
def spec_grad_one(seed):
    d = fixed_draws(seed)
    rem = jax.random.uniform(jax.random.PRNGKey(1_000_000 + seed))
    g = jax.grad(J_spec, argnums=(0, 1, 2))(DT, FNUM, VOL, rem, d)
    return jnp.array([g[0], g[1], g[2]])


names = ["d/d(dt)", "d/d(fnum)", "d/d(volume)"]
for M in [200, 2000, 20000]:
    seeds = jnp.arange(M)
    T, _ = jax.vmap(truth_one)(seeds)
    E = jax.vmap(est_one)(seeds)
    S = jax.vmap(spec_grad_one)(seeds)
    print(f"\n--- M = {M} seeds ---")
    for c in range(3):
        t, tm, ts = T[:, c], float(jnp.mean(T[:, c])), float(jnp.std(T[:, c]) / np.sqrt(M))
        em, es = float(jnp.mean(E[:, c])), float(jnp.std(E[:, c]) / np.sqrt(M))
        print(
            f" {names[c]:>12}: truth(RB) = {tm: .5e} +/- {ts:.1e} | "
            f"fix(REINFORCE) = {em: .5e} +/- {es:.1e} | spec(AD) = {float(jnp.mean(S[:, c])): .3e}"
        )


# ---- independent check: FD on the ensemble mean E[J] with common random numbers ----
@jax.jit
def Jbar(dt, fnum, vol, seeds):
    def one(s):
        d = fixed_draws(s)
        rem = jax.random.uniform(jax.random.PRNGKey(1_000_000 + s))
        Je, _, _ = J_cond_exp(dt, fnum, vol, rem, d)  # exact over count randomness
        return Je

    return jnp.mean(jax.vmap(one)(seeds))


seeds = jnp.arange(20000)
for rel in [1e-3, 1e-2]:
    e = DT * rel
    fd = (Jbar(DT + e, FNUM, VOL, seeds) - Jbar(DT - e, FNUM, VOL, seeds)) / (2 * e)
    print(f"\nFD of E[J] wrt dt (rel={rel:.0e}, 20000 CRN seeds): {float(fd): .5e}")
ef = FNUM * 1e-2
print(
    "FD of E[J] wrt fnum:",
    float((Jbar(DT, FNUM + ef, VOL, seeds) - Jbar(DT, FNUM - ef, VOL, seeds)) / (2 * ef)),
)
ev = VOL * 1e-2
print(
    "FD of E[J] wrt vol :",
    float((Jbar(DT, FNUM, VOL + ev, seeds) - Jbar(DT, FNUM, VOL - ev, seeds)) / (2 * ev)),
)
