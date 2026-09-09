"""Gate U3 -- can a seeding rule hold a majorant margin without knowing max q?

Spec: PSEUDOCODE.md section 19 row U3; section 7.2 (stale-max rule); section 7.2.1
(violations bias the gradient); section 7.2.2 (Sigma is a variance knob);
section 8.1 (attempt count); section 8.2 (A_max sizing).

Run standalone::

    python scratch/gate_U3_majorant_margin.py

The claim under test is that "a 2-5x safety factor on Sigma eliminates majorant
violations".  Violations are known to be zero when ``Sigma / max q >= 1``, but
that ratio has only ever been evaluated post hoc.  Under the [DF] stale-max rule
we adopted (section 18.1) ``Sigma`` is frozen for the whole step, so ``q > Sigma``
is *possible* -- unlike SPARTA's live ratchet, where it is impossible by
construction.  This probe implements

  * SPARTA's ``vremax_init`` seed (collide_vss.cpp:107-133, ``2 * cxs * max(vscale)``),
  * the [DF] stale-max update (``Sigma`` fixed within a step, republished after),

runs a real 0D NTC relaxation across three transients, and measures
``majorant_violations`` and ``min_majorant_margin`` per step.

Scattering is the ``alpha == 1`` isotropic limit of VSS: it conserves momentum
and energy exactly and reproduces the same relative-speed statistics, which is
all the majorant question depends on.
"""

import functools
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC")

from jax_dsmc.constants import EPS_REL_VR2, K_BOLTZ  # noqa: E402
from jax_dsmc.safe import safe_log, safe_pow  # noqa: E402

# --- argon VSS, SPARTA data/ar.vss + data/ar.species ------------------------
AR_DIAM = 4.11e-10
AR_OMEGA = 0.81
AR_TREF = 273.15
AR_MASS = 6.63e-26
AR_MR = AR_MASS / 2.0

N_PART = 2000
A_MAX = 8000
N_STEPS = 80
K_EV = 5.0  # extreme-value multiplier for the thermal seed: P(u > 5) * N^2/2 ~ 1.6e-4


def cxs():
    return math.pi * AR_DIAM * AR_DIAM


def prefactor(omega):
    return cxs() * (2.0 * K_BOLTZ * AR_TREF / AR_MR) ** (omega - 0.5) / math.gamma(2.5 - omega)


def eps_abs():
    return EPS_REL_VR2 * (2.0 * K_BOLTZ * AR_TREF / AR_MR)


def sparta_vremax_init(temp_thermal):
    """SPARTA collide_vss.cpp:107-133.

    ``vscale[i] = sqrt(2 k T_thermal / m_i)`` (mixture.cpp:229-237 -- note the
    *species* mass, not the reduced mass), ``beta = max(vscale)``,
    ``vrm = 2 * cxs * beta``.
    """
    vscale = math.sqrt(2.0 * K_BOLTZ * temp_thermal / AR_MASS)
    return 2.0 * cxs() * vscale


def thermal_majorant(temp, omega, k_ev=K_EV):
    """Proposed replacement seed: an a-priori bound on q from the cell temperature.

    ``q(vr) = prefactor * (vr^2)^(1-omega)`` is monotone in ``vr``, so on the
    reachable interval ``[sqrt(eps), k_ev * v_ref(T)]`` its supremum sits at an
    endpoint: the top one for ``omega < 1``, the clamp for ``omega > 1``.
    """
    vref2 = 2.0 * K_BOLTZ * temp / AR_MR
    hi = prefactor(omega) * (k_ev * k_ev * vref2) ** (1.0 - omega)
    lo = prefactor(omega) * eps_abs() ** (1.0 - omega)
    return max(hi, lo)


# --- kernel ----------------------------------------------------------------


def rate_q(vr2, omega, pref):
    return pref * safe_pow(vr2, 1.0 - omega, eps_abs())


def temperature(v):
    vbar = jnp.mean(v, axis=0)
    return AR_MASS / (3.0 * K_BOLTZ) * jnp.mean(jnp.sum((v - vbar) ** 2, axis=-1))


def max_pair_q(v, omega, pref):
    """Exact max of q over ALL N(N-1)/2 pairs at the start of a step."""
    g = v @ v.T
    d = jnp.diag(g)
    vr2 = d[:, None] + d[None, :] - 2.0 * g
    vr2 = jnp.where(jnp.eye(v.shape[0], dtype=bool), jnp.nan, vr2)
    lo = jnp.nanmin(vr2)
    hi = jnp.nanmax(vr2)
    return jnp.maximum(rate_q(hi, omega, pref), rate_q(lo, omega, pref))


def _slot(carry, xs, omega, pref, sigma):
    v, nviol, minmarg, nacc, logh, qmax = carry
    i, j, u_acc, nhat, active = xs
    vr = v[i] - v[j]
    vr2 = jnp.sum(vr * vr)
    q = rate_q(vr2, omega, pref)
    p = jnp.clip(q / sigma, 0.0, 1.0)
    acc = active & (u_acc < p)
    viol = active & (q > sigma)
    marg = jnp.where(active & (q > 0.0), sigma / jnp.where(q > 0.0, q, 1.0), jnp.inf)
    vcm = 0.5 * (v[i] + v[j])
    vrm = jnp.sqrt(vr2)
    vr_new = vrm * nhat
    vi_new = jnp.where(acc, vcm + 0.5 * vr_new, v[i])
    vj_new = jnp.where(acc, vcm - 0.5 * vr_new, v[j])
    v = v.at[i].set(vi_new).at[j].set(vj_new)
    logh = logh + jnp.where(active, jnp.where(acc, safe_log(p), safe_log(1.0 - p)), 0.0)
    qmax = jnp.maximum(qmax, jnp.where(active, q, 0.0))
    return (v, nviol + viol.astype(jnp.int64), jnp.minimum(minmarg, marg),
            nacc + acc.astype(jnp.int64), logh, qmax), None


def step(v, sigma, key, omega, pref, rate_const, n_part, a_max):
    """One NTC step under the [DF] stale-max rule: ``sigma`` is frozen throughout."""
    kp, ku, kn, kr = jax.random.split(key, 4)
    f = 0.5 * n_part * (n_part - 1.0) * sigma * rate_const
    u_round = jax.random.uniform(kr, ())
    frac = f - jnp.floor(f)
    n_att = (jnp.floor(f) + (u_round < frac)).astype(jnp.int64)
    n_used = jnp.minimum(n_att, a_max)
    clipped = jnp.maximum(n_att - a_max, 0)

    ij = jax.random.randint(kp, (a_max, 2), 0, n_part)
    i = ij[:, 0]
    j = ij[:, 1]
    u_acc = jax.random.uniform(ku, (a_max,))
    nrm = jax.random.normal(kn, (a_max, 3))
    nhat = nrm / jnp.linalg.norm(nrm, axis=-1, keepdims=True)
    active = (jnp.arange(a_max) < n_used) & (i != j)

    body = functools.partial(_slot, omega=omega, pref=pref, sigma=sigma)
    carry0 = (v, jnp.int64(0), jnp.inf, jnp.int64(0), 0.0, 0.0)
    (v, nviol, minmarg, nacc, logh, qmax), _ = jax.lax.scan(
        body, carry0, (i, j, u_acc, nhat, active)
    )

    q_all = max_pair_q(v, omega, pref)
    return v, {
        "n_att": n_att,
        "n_used": n_used,
        "clipped": clipped,
        "viol": nviol,
        "min_margin": minmarg,
        "n_acc": nacc,
        "logh": logh,
        "q_max_seen": qmax,
        "q_max_end": q_all,
    }


_step_jit = jax.jit(step, static_argnames=("n_part", "a_max"))
_maxpair_jit = jax.jit(max_pair_q)
_temp_jit = jax.jit(temperature)


# --- transients ------------------------------------------------------------


def ic_equilibrium(key, temp):
    return math.sqrt(K_BOLTZ * temp / AR_MASS) * jax.random.normal(key, (N_PART, 3))


def ic_beams(key, temp_cold, drift):
    """Counter-streaming beams: a genuinely non-equilibrium IC that thermalises."""
    v = math.sqrt(K_BOLTZ * temp_cold / AR_MASS) * jax.random.normal(key, (N_PART, 3))
    sign = jnp.where(jnp.arange(N_PART) < N_PART // 2, 1.0, -1.0)
    return v.at[:, 0].add(sign * drift)


def forcing_none(v, k):
    del k
    return v


def forcing_heat(v, k, rate=1.02, n_on=60):
    return v * jnp.where(k < n_on, rate, 1.0)


# --- majorant update rules -------------------------------------------------
# Each returns the Sigma to use for the NEXT step, given the current Sigma, the
# max q actually observed this step, and the current cell temperature.


def rule_stale(sigma, q_obs, temp, omega, f_up=1.0, k_ev=K_EV, use_thermal=False):
    del temp, omega, k_ev, use_thermal
    return jnp.maximum(sigma, f_up * q_obs)


def rule_thermal(sigma, q_obs, temp, omega, f_up=2.0, k_ev=K_EV):
    """Proposed rule: an a-priori majorant from the current cell temperature.

    ``temp`` is known at the *start* of the step (it is sampled anyway), so this
    does not lag a transient the way a ratchet on the observed max does.
    """
    vref2 = 2.0 * K_BOLTZ * temp / AR_MR
    pref = prefactor(omega)
    hi = pref * (k_ev * k_ev * vref2) ** (1.0 - omega)
    lo = pref * eps_abs() ** (1.0 - omega)
    a_priori = f_up * jnp.maximum(hi, lo)
    return jnp.maximum(jnp.maximum(sigma, a_priori), f_up * q_obs)


RULES = {
    "sparta seed, F_up=1": (1.0, rule_stale, {"f_up": 1.0}),
    "sparta seed x2, F_up=1": (2.0, rule_stale, {"f_up": 1.0}),
    "sparta seed x5, F_up=1": (5.0, rule_stale, {"f_up": 1.0}),
    "sparta seed x5, F_up=2": (5.0, rule_stale, {"f_up": 2.0}),
    "thermal K=5 F=1": (1.0, rule_thermal, {"f_up": 1.0}),
    "thermal K=5 F=2": (1.0, rule_thermal, {"f_up": 2.0}),
}


def run(case, rule_name, omega, seed=0, verbose=False, n_steps=N_STEPS):
    pref = prefactor(omega)
    seed_mult, rule, kw = RULES[rule_name]
    key = jax.random.PRNGKey(seed)
    kic, krun = jax.random.split(key)

    if case == "equilibrium":
        v = ic_equilibrium(kic, AR_TREF)
        forcing = forcing_none
    elif case == "beams":
        v = ic_beams(kic, 20.0, 700.0)
        forcing = forcing_none
    elif case == "heated":
        v = ic_equilibrium(kic, AR_TREF)
        forcing = functools.partial(forcing_heat)
    else:
        raise ValueError(case)

    # rate_const = fnum * dt / cell_volume, chosen so the seeded Sigma gives
    # ~1500 attempts at step 0 -- a realistic per-cell attempt load.
    sigma0 = seed_mult * sparta_vremax_init(AR_TREF)
    if rule is rule_thermal:
        sigma0 = kw["f_up"] * thermal_majorant(AR_TREF, omega)
    rate_const = 1500.0 / (0.5 * N_PART * (N_PART - 1.0) * sparta_vremax_init(AR_TREF))

    sigma = jnp.asarray(sigma0)
    hist = []
    for k in range(n_steps):
        v = forcing(v, k)
        temp = _temp_jit(v)
        q_pre = _maxpair_jit(v, omega, pref)
        v, d = _step_jit(v, sigma, jax.random.fold_in(krun, k), omega, pref,
                         rate_const, N_PART, A_MAX)
        rec = {
            "k": k,
            "T": float(temp),
            "sigma": float(sigma),
            "q_max_pre": float(q_pre),
            "q_max_post": float(d["q_max_end"]),
            "q_max_seen": float(d["q_max_seen"]),
            "true_margin": float(sigma / q_pre),
            "min_margin": float(d["min_margin"]),
            "viol": int(d["viol"]),
            "n_att": int(d["n_att"]),
            "n_acc": int(d["n_acc"]),
            "clipped": int(d["clipped"]),
        }
        hist.append(rec)
        if verbose:
            print(
                f"  k={k:3d} T={rec['T']:8.2f} Sigma={rec['sigma']:.4e} maxq={rec['q_max_pre']:.4e} "
                f"S/maxq={rec['true_margin']:6.3f} minmarg={rec['min_margin']:8.3f} "
                f"viol={rec['viol']:5d} natt={rec['n_att']:6d} nacc={rec['n_acc']:5d}"
            )
        # The ratchet can only see the pairs it actually attempted -- exactly what
        # SPARTA's MAX(vre, vremax) and [DF]'s deferred publish both have access to.
        sigma = rule(sigma, d["q_max_seen"], temp, omega, **kw)
    return hist


def summarise(hist):
    v = sum(h["viol"] for h in hist)
    first = [h["viol"] for h in hist[:5]]
    mm = min(h["min_margin"] for h in hist)
    tm = min(h["true_margin"] for h in hist)
    att = max(h["n_att"] for h in hist)
    clip = sum(h["clipped"] for h in hist)
    steps_bad = sum(1 for h in hist if h["viol"] > 0)
    return v, first, mm, tm, att, clip, steps_bad


def main():
    omega = AR_OMEGA
    print("GATE U3 -- majorant seeding under the stale-max rule")
    print(f"argon VSS omega={omega}, N={N_PART}, A_max={A_MAX}, {N_STEPS} steps, f64")
    print(f"SPARTA vremax_init(T_ref) = {sparta_vremax_init(AR_TREF):.6e} m^3/s")
    print(f"thermal K=5 majorant(T_ref) = {thermal_majorant(AR_TREF, omega):.6e} m^3/s")
    print(f"ratio thermal/sparta = {thermal_majorant(AR_TREF, omega) / sparta_vremax_init(AR_TREF):.3f}")

    print("\n--- Step-by-step, the case that matters: SPARTA seed, equilibrium, first 10 steps ---")
    run("equilibrium", "sparta seed, F_up=1", omega, verbose=True, n_steps=10)

    for case in ("equilibrium", "beams", "heated"):
        print(f"\n=== case: {case} ===")
        print(
            f"{'rule':>26} {'tot viol':>9} {'steps w/ viol':>13} {'viol steps 0-4':>18} "
            f"{'min Sigma/q':>12} {'min Sigma/maxq_all':>19} {'max n_att':>10} {'clipped':>8}"
        )
        for rname in RULES:
            h = run(case, rname, omega)
            v, first, mm, tm, att, clip, bad = summarise(h)
            print(
                f"{rname:>26} {v:9d} {bad:13d} {str(first):>18} {mm:12.4f} {tm:19.4f} "
                f"{att:10d} {clip:8d}"
            )

    print("\n=== margin trajectory, heated case (T rises 3.3x over 60 steps) ===")
    for rname in ("sparta seed x5, F_up=1", "thermal K=5 F=2"):
        h = run("heated", rname, omega)
        print(f"\n  {rname}")
        print(f"  {'k':>4} {'T':>9} {'Sigma/maxq':>11} {'min Sigma/q':>12} {'viol':>6} {'n_att':>7}")
        for k in list(range(6)) + list(range(10, N_STEPS, 10)):
            r = h[k]
            print(
                f"  {r['k']:4d} {r['T']:9.2f} {r['true_margin']:11.4f} {r['min_margin']:12.4f} "
                f"{r['viol']:6d} {r['n_att']:7d}"
            )

    print("\n=== omega sweep: does the answer depend on the exponent? ===")
    print(f"{'omega':>6} {'rule':>26} {'tot viol':>9} {'min Sigma/q':>12} {'min Sigma/maxq_all':>19}")
    for w in (0.5, 0.7, AR_OMEGA, 1.0, 1.2):
        for rname in ("sparta seed, F_up=1", "thermal K=5 F=2"):
            h = run("beams", rname, w, n_steps=40)
            v, _f, mm, tm, _a, _c, _b = summarise(h)
            print(f"{w:6.2f} {rname:>26} {v:9d} {mm:12.4f} {tm:19.4f}")

    print("\n=== compute cost of the margin (section 7.2.2 vs section 8.2) ===")
    print("n_att ~ Sigma, so the margin is paid for one-for-one in attempts.")
    print(f"{'rule':>26} {'mean n_att':>11} {'max n_att':>10} {'mean accept':>12}")
    for rname in RULES:
        h = run("heated", rname, omega)
        na = np.mean([x["n_att"] for x in h])
        acc = np.mean([x["n_acc"] / max(x["n_att"], 1) for x in h])
        print(f"{rname:>26} {na:11.1f} {max(x['n_att'] for x in h):10d} {acc:12.4f}")


if __name__ == "__main__":
    main()
