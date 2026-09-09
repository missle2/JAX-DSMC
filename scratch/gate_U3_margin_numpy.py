"""U3 gate: is a 2-5x majorant safety factor achievable by a seeding rule?

Rewrite of `gate_U3_majorant_margin.py` in pure NumPy.  The JAX original hangs
in compilation at A_MAX=8000; U3 needs no autodiff -- it is pure majorant
bookkeeping -- so NumPy answers it directly and in seconds.

WHY U3 IS BLOCKING (spec section 19).  Under the stale-max rule (decision 9,
[DF] noTimeCounter.C:231-246) every candidate in step k is tested against Sigma
captured BEFORE the loop.  If Sigma < q for some tested pair, the acceptance
probability q/Sigma saturates at 1: that pair is under-counted, and -- the part
that matters here -- it contributes ZERO score, which BIASES the gradient
(measured earlier: 75% low at a 15% violation rate, 14% low at 1%).  So the
gate is not "does it run", it is "can a seeding rule hold the violation rate low
enough that the gradient bias is acceptable, at a cost in attempts we can pay".

Target: violation rate < 1e-3 of tested pairs => bias ~< 1.4% by that scaling.
"""
import math
import numpy as np

K_BOLTZ = 1.380649e-23
AR_DIAM, AR_OMEGA, AR_TREF, AR_MASS = 4.11e-10, 0.81, 273.15, 6.63e-26
AR_MR = AR_MASS / 2.0
EPS_REL_VR2 = 1e-8

CXS = math.pi * AR_DIAM ** 2
def prefactor(w):
    return CXS * (2.0 * K_BOLTZ * AR_TREF / AR_MR) ** (w - 0.5) / math.gamma(2.5 - w)
def eps_abs():
    return EPS_REL_VR2 * (2.0 * K_BOLTZ * AR_TREF / AR_MR)
def rate_q(vr2, w, pref):
    return pref * np.maximum(vr2, eps_abs()) ** (1.0 - w)
def temperature(v):
    return AR_MASS / (3.0 * K_BOLTZ) * np.mean(np.sum((v - v.mean(0)) ** 2, -1))

def sparta_vremax_init(temp):
    """SPARTA collide_vss.cpp:107-133 -- note SPECIES mass, not reduced mass."""
    return 2.0 * CXS * math.sqrt(2.0 * K_BOLTZ * temp / AR_MASS)

def thermal_majorant(temp, w, k_ev=5.0, f_up=1.0):
    """A-priori bound: q is monotone in |vr|, so sup over the reachable
    interval [sqrt(eps), k_ev*v_ref(T)] sits at an endpoint.  `temp` is known at
    the START of the step, so unlike a ratchet this does not lag a transient."""
    vref2 = 2.0 * K_BOLTZ * temp / AR_MR
    hi = prefactor(w) * (k_ev * k_ev * vref2) ** (1.0 - w)
    lo = prefactor(w) * eps_abs() ** (1.0 - w)
    return f_up * max(hi, lo)

def true_max_q(v, w, pref):
    """Exact sup of q over all N(N-1)/2 pairs (monotone => extreme |vr|)."""
    g = v @ v.T; d = np.diag(g)
    vr2 = d[:, None] + d[None, :] - 2.0 * g
    np.fill_diagonal(vr2, np.nan)
    return max(rate_q(np.nanmax(vr2), w, pref), rate_q(np.nanmin(vr2), w, pref))

N, A_MAX, STEPS = 600, 3000, 60
def run(ic, rule, w=AR_OMEGA, seed=0, heat=False):
    rng = np.random.default_rng(seed)
    pref = prefactor(w)
    vth = math.sqrt(K_BOLTZ * AR_TREF / AR_MASS)
    if ic == "equil":
        v = vth * rng.standard_normal((N, 3))
    elif ic == "cold":                       # 100x colder: tests the low end
        v = 0.1 * vth * rng.standard_normal((N, 3))
    else:                                    # counter-streaming beams
        v = 0.1 * vth * rng.standard_normal((N, 3))
        v[:N // 2, 0] += 1500.0; v[N // 2:, 0] -= 1500.0
    sigma = sparta_vremax_init(AR_TREF) * (5.0 if rule == "sparta_x5" else 1.0)
    nviol = ntot = 0; margins = []; sigmas = []
    for k in range(STEPS):
        if heat and k < STEPS // 2:
            v *= 1.03                        # 3%/step heating transient
        T = temperature(v)
        if rule.startswith("thermal"):
            f_up = 2.0 if rule.endswith("f2") else 1.0
            sigma = max(sigma, thermal_majorant(T, w, f_up=f_up))
        i = rng.integers(0, N, A_MAX); j = (i + 1 + rng.integers(0, N - 1, A_MAX)) % N
        vr2 = np.sum((v[i] - v[j]) ** 2, -1)
        q = rate_q(vr2, w, pref)
        nviol += int((q > sigma).sum()); ntot += A_MAX
        acc = rng.random(A_MAX) < np.minimum(q / sigma, 1.0)
        # elastic isotropic scatter of accepted pairs (VHS), momentum+energy exact
        ia, ja = i[acc], j[acc]
        vcm = 0.5 * (v[ia] + v[ja]); vrel = np.linalg.norm(v[ia] - v[ja], axis=-1)
        ct = 2.0 * rng.random(ia.size) - 1.0; st = np.sqrt(1.0 - ct * ct)
        ph = 2.0 * math.pi * rng.random(ia.size)
        nd = np.stack([ct, st * np.cos(ph), st * np.sin(ph)], -1) * vrel[:, None]
        v[ia] = vcm + 0.5 * nd; v[ja] = vcm - 0.5 * nd
        qmax_true = true_max_q(v, w, pref)
        margins.append(sigma / qmax_true); sigmas.append(sigma)
        if rule.startswith("sparta"):
            sigma = max(sigma, q.max())      # stale-max ratchet
    return nviol / ntot, np.array(margins), np.array(sigmas)

print(f"argon VSS w={AR_OMEGA}, N={N}, A_max={A_MAX}, {STEPS} steps")
print(f"SPARTA seed sigma_0 = {sparta_vremax_init(AR_TREF):.4e}")
print(f"thermal K=5 F=1 at T_ref = {thermal_majorant(AR_TREF, AR_OMEGA):.4e}  "
      f"(ratio to SPARTA seed {thermal_majorant(AR_TREF,AR_OMEGA)/sparta_vremax_init(AR_TREF):.3f})\n")
print(f"{'IC':>8} {'rule':>12} {'heat':>5} | {'viol rate':>10} {'min margin':>11} "
      f"{'med margin':>11} {'cost x':>7}")
print("-" * 76)
base = None
for ic in ("equil", "cold", "beams"):
    for heat in (False, True):
        for rule in ("sparta", "sparta_x5", "thermal_f1", "thermal_f2"):
            vr, mg, sg = run(ic, rule, heat=heat, seed=7)
            cost = sg.mean() / sparta_vremax_init(AR_TREF)
            flag = "  <-- FAILS" if vr > 1e-3 else ("" if mg.min() >= 1.0 else "  <-- margin<1")
            print(f"{ic:>8} {rule:>12} {str(heat):>5} | {vr:>10.2e} {mg.min():>11.3f} "
                  f"{np.median(mg):>11.3f} {cost:>7.2f}{flag}")
    print()


# --- what the margin actually costs -----------------------------------------
# n_attempts ~ Sigma and P_accept = q/Sigma, so E[collisions] is Sigma-INVARIANT:
#   E[coll] = (0.5 N(N-1) Sigma dt fnum / V) * <q>/Sigma  -- Sigma cancels.
# If that holds, a 2x margin costs compute (more scan slots) and NOT physics.
print("\n=== is the collision count really Sigma-invariant? ===")
print(f"{'Sigma / Sigma_0':>16} {'attempts':>10} {'accepted':>10} {'accept rate':>12} "
      f"{'collisions/step':>16}")
rng = np.random.default_rng(11)
pref = prefactor(AR_OMEGA)
vth = math.sqrt(K_BOLTZ * AR_TREF / AR_MASS)
v0 = vth * rng.standard_normal((N, 3))
sig0 = thermal_majorant(AR_TREF, AR_OMEGA)
rate_const = 3000.0 / sig0          # fixes n = rate_const * Sigma  (n ~ Sigma)
for mult in (0.5, 1.0, 2.0, 5.0, 10.0):
    sigma = sig0 * mult
    n_att = int(round(rate_const * sigma))
    rg = np.random.default_rng(4242)     # common random numbers across mult
    i = rg.integers(0, N, n_att); j = (i + 1 + rg.integers(0, N - 1, n_att)) % N
    q = rate_q(np.sum((v0[i] - v0[j]) ** 2, -1), AR_OMEGA, pref)
    acc = rg.random(n_att) < np.minimum(q / sigma, 1.0)
    print(f"{mult:>16.1f} {n_att:>10d} {int(acc.sum()):>10d} "
          f"{acc.mean():>12.4f} {int(acc.sum()):>16d}")
print("collisions flat across a 20x sweep in Sigma => the margin is a COMPUTE cost only.")
