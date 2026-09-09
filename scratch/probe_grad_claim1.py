import jax, jax.numpy as jnp, numpy as np

jax.config.update("jax_enable_x64", True)
from functools import partial

N = 10
AMAX = 16
SIGMA = 3.0  # vremax majorant, held fixed (stop_gradient in spec)
PREF = 2.0
OMEGA = 0.75


def fixed_draws(seed):
    """Random numbers held FIXED across parameter perturbations (reparameterisation)."""
    k = jax.random.PRNGKey(seed)
    ks = jax.random.split(k, 6)
    v = jax.random.normal(ks[0], (N, 3))
    ii = jax.random.randint(ks[1], (AMAX,), 0, N)
    jj = (ii + 1 + jax.random.randint(ks[2], (AMAX,), 0, N - 1)) % N
    uacc = jax.random.uniform(ks[3], (AMAX,))
    ct = 2.0 * jax.random.uniform(ks[4], (AMAX,)) - 1.0  # cos theta
    ph = 2.0 * jnp.pi * jax.random.uniform(ks[5], (AMAX,))
    return v, ii, jj, uacc, ct, ph


def collide_with_n(n, draws):
    """One NTC step in one cell with attempt count n (traced-float or int, used only via a<n)."""
    v, ii, jj, uacc, ct, ph = draws

    def body(carry, a):
        v, L = carry
        i, j = ii[a], jj[a]
        vr = v[i] - v[j]
        vr2 = jnp.sum(vr * vr)
        q = PREF * vr2 ** (1.0 - OMEGA)
        h = jnp.clip(q / SIGMA, 0.0, 1.0)
        active = a < n
        acc = active & (uacc[a] < h)
        vcm = 0.5 * (v[i] + v[j])
        vrm = jnp.sqrt(vr2)
        st = jnp.sqrt(jnp.maximum(1.0 - ct[a] ** 2, 0.0))
        sig = vrm * jnp.array([ct[a], st * jnp.cos(ph[a]), st * jnp.sin(ph[a])])
        vi_n = jnp.where(acc, vcm + 0.5 * sig, v[i])
        vj_n = jnp.where(acc, vcm - 0.5 * sig, v[j])
        v = v.at[i].set(vi_n).at[j].set(vj_n)
        dl = jnp.where(acc, jnp.log(jnp.maximum(h, 1e-300)), 0.0)
        L = L.at[i].add(dl).at[j].add(dl)
        return (v, L), None

    (v, L), _ = jax.lax.scan(body, (v, jnp.zeros(N)), jnp.arange(AMAX))
    return v, L


def phi(v):
    r2 = jnp.sum(v * v, axis=1)
    return r2 * r2  # 4th moment: NOT conserved by collisions


magic = lambda L: jnp.exp(L - jax.lax.stop_gradient(L))


def f_of(dt, fnum, vol, remain):
    return 0.5 * N * (N - 1.0) * SIGMA * dt * fnum / vol + remain


# ---------------- SPEC scheme: n = floor(f), deterministic carry ----------------
def J_spec(dt, fnum, vol, remain, draws):
    f = f_of(dt, fnum, vol, remain)
    n = jnp.floor(f)
    v, L = collide_with_n(n, draws)
    return jnp.mean(magic(L) * phi(v))


# ---------------- FIX: stochastic rounding + log P(n|f) in the score -------------
def J_fix(dt, fnum, vol, remain, draws, ucount):
    f = f_of(dt, fnum, vol, remain)
    fl = jnp.floor(f)
    fr = f - fl
    up = ucount < fr  # frozen integer decision
    n = fl + jnp.where(up, 1.0, 0.0)
    logP = jnp.where(
        up, jnp.log(jnp.clip(fr, 1e-300, 1.0)), jnp.log(jnp.clip(1.0 - fr, 1e-300, 1.0))
    )
    v, L = collide_with_n(jax.lax.stop_gradient(n), draws)
    L = L + logP  # cell-wide score on every particle in cell
    return jnp.mean(magic(L) * phi(v))


# ---- exact conditional expectation over the count decision (Rao-Blackwell truth) ----
def J_cond_exp(dt, fnum, vol, remain, draws):
    f = f_of(dt, fnum, vol, remain)
    fl = jnp.floor(f)
    fr = f - fl
    vd, Ld = collide_with_n(fl, draws)
    vu, Lu = collide_with_n(fl + 1.0, draws)
    Jd = jnp.mean(magic(Ld) * phi(vd))
    Ju = jnp.mean(magic(Lu) * phi(vu))
    return (1 - fr) * Jd + fr * Ju, Jd, Ju


DT, FNUM, VOL = 1.0e-3, 1.0e3, 1.0  # -> f0 = 45*3*1e-3*1e3/1 = 135.0 ... too big
# retune so f ~ 3.x with AMAX=16
DT, FNUM, VOL = 2.5e-5, 1.0e3, 1.0
draws = fixed_draws(0)
REM = 0.37
print("f =", float(f_of(DT, FNUM, VOL, REM)))

# ---------- (a)+(b)+(c): AD gradient through the floor path is exactly zero ----------
g = jax.grad(J_spec, argnums=(0, 1, 2))(DT, FNUM, VOL, REM, draws)
print("\n[SPEC] jax.grad J wrt (dt, fnum, volume) =", [float(x) for x in g])
print("[SPEC] J value =", float(J_spec(DT, FNUM, VOL, REM, draws)))


# gradient wrt a parameter that DOES appear in log h, for contrast: perturb SIGMA? use omega via closure
def J_spec_pref(pref, dt, draws):
    global PREF
    old = PREF
    return None


# instead: show score path is alive wrt velocities
gv = jax.grad(lambda vv: J_spec(DT, FNUM, VOL, REM, (vv,) + draws[1:]))(draws[0])
print("[SPEC] |grad wrt v| (score+pathwise path is alive) =", float(jnp.linalg.norm(gv)))

# ---------- finite differences with FIXED randoms, several step sizes ----------
print("\n[SPEC] central FD in dt with fixed random numbers:")
for rel in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
    e = DT * rel
    fp = f_of(DT + e, FNUM, VOL, REM)
    fm = f_of(DT - e, FNUM, VOL, REM)
    Jp = float(J_spec(DT + e, FNUM, VOL, REM, draws))
    Jm = float(J_spec(DT - e, FNUM, VOL, REM, draws))
    fd = (Jp - Jm) / (2 * e)
    print(f"  rel={rel:8.0e}  n(+)={int(np.floor(fp))} n(-)={int(np.floor(fm))}  FD={fd: .6e}")
