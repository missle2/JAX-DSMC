"""Gate U1 -- does EPS_REL_VR2 = 1e-8 bound d q / d vr2 acceptably?

Spec: PSEUDOCODE.md section 19, row U1; section 2.3; section 2.4; section 7.2.

Run standalone::

    python scratch/gate_U1_clamp_bound.py

Three questions, in order:

  1. The gate as written in section 19: sweep ``omega in [0.5, 1.5]`` and ``|vr|``
     down to the clamp and assert ``max|dq/dvr2| <= 2|1-omega| eps^(0.5-omega)``.
     Report bound/achieved -- a vacuous bound is a failure of the *bound*.
  2. Is the value 1e-8 right at the loose end?  What fraction of pairs in an
     equilibrium Maxwellian at 300 K is truncated by the clamp?
  3. Is it right at the tight end?  How far above the equilibrium population does
     the worst-case (at-the-clamp) gradient sit?

Everything is prefactor-free and expressed in the natural dimensionless variable
``u = |vr| / v_ref`` with ``v_ref = sqrt(2 k T_ref / m_r)``, because that is the
variable the clamp is defined in: ``eps = EPS_REL_VR2 * v_ref^2`` means exactly
``u > sqrt(EPS_REL_VR2)``.  A separate SI pass repeats the headline numbers with
the argon VSS prefactor so the unit-dependence of the section-19 inequality is
visible.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

jax.config.update("jax_enable_x64", True)

import sys  # noqa: E402

sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC")

from jax_dsmc.constants import EPS_REL_VR2, K_BOLTZ  # noqa: E402
from jax_dsmc.safe import safe_pow  # noqa: E402

# --- argon VSS, SPARTA data/ar.vss + data/ar.species ------------------------
AR_DIAM = 4.11e-10
AR_OMEGA = 0.81
AR_TREF = 273.15
AR_ALPHA = 1.4
AR_MASS = 6.63e-26
AR_MR = AR_MASS / 2.0

OMEGAS = np.round(np.arange(0.5, 1.5001, 0.1), 6)


def prefactor(diam, tref, mr, omega):
    """SPARTA collide_vss.cpp:122-125."""
    cxs = math.pi * diam * diam
    return cxs * (2.0 * K_BOLTZ * tref / mr) ** (omega - 0.5) / math.gamma(2.5 - omega)


def q_of_vr2(vr2, omega, eps, pref):
    """Collision rate as a function of vr2 (section 7.2)."""
    return pref * safe_pow(vr2, 1.0 - omega, eps)


def q_of_vrvec(vrvec, omega, eps, pref):
    """Same rate, but as a function of the relative-velocity *vector*.

    This is the composition AD actually differentiates through in vss.py:
    ``vr2 = sum(vr*vr)`` then ``safe_pow``.  ``||grad_vr q|| = |dq/dvr2| * 2|vr|``.
    """
    return q_of_vr2(jnp.sum(vrvec * vrvec), omega, eps, pref)


_d_dvr2 = jax.jit(jax.vmap(jax.grad(q_of_vr2), in_axes=(0, None, None, None)))
_d_dvrvec = jax.jit(jax.vmap(jax.grad(q_of_vrvec), in_axes=(0, None, None, None)))


def sweep_one_omega(omega, eps, pref, u_hi, n_u):
    """Both derivative readings over |vr| from the clamp up to ``u_hi * sqrt(eps)``.

    Returns (max |dq/dvr2|, max ||grad_vr q||).
    """
    u_c = math.sqrt(eps)
    u = np.geomspace(u_c * (1.0 + 1e-12), u_hi, n_u)
    # put samples exactly on the clamp shoulder, where the maximum lives for omega > 0.5
    u = np.concatenate([u_c * (1.0 + np.geomspace(1e-14, 1e-3, 16)), u])
    vr2 = jnp.asarray(u * u)
    g2 = np.abs(np.asarray(_d_dvr2(vr2, omega, eps, pref)))
    vec = jnp.stack([jnp.asarray(u), jnp.zeros_like(jnp.asarray(u)), jnp.zeros_like(jnp.asarray(u))], axis=-1)
    gv = np.linalg.norm(np.asarray(_d_dvrvec(vec, omega, eps, pref)), axis=-1)
    return g2.max(), gv.max()


def stated_bound(omega, eps, pref):
    """The section-2.4 / section-19 formula ``2 |1-omega| eps^(0.5-omega)``."""
    return pref * 2.0 * abs(1.0 - omega) * eps ** (0.5 - omega)


def dq_dvr2_bound(omega, eps, pref):
    """The correct sup of |dq/dvr2| on the clamped domain: |1-omega| eps^(-omega)."""
    return pref * abs(1.0 - omega) * eps ** (-omega)


def part1_bound_sweep(eps, pref, label):
    print(f"\n=== U1 part 1: the section-19 inequality ({label}) ===")
    print(f"eps = {eps:.6e}   prefactor = {pref:.6e}")
    print(
        f"{'omega':>6} {'max|dq/dvr2|':>14} {'bound 2|1-w|e^(.5-w)':>21} {'ratio B/ach':>12} "
        f"{'max||grad_v q||':>16} {'ratio B/ach':>12}"
    )
    rows = []
    for omega in OMEGAS:
        a2, av = sweep_one_omega(float(omega), eps, pref, u_hi=5.0, n_u=400)
        b = stated_bound(float(omega), eps, pref)
        r2 = b / a2 if a2 > 0 else math.inf
        rv = b / av if av > 0 else math.inf
        rows.append((float(omega), a2, b, r2, av, rv))
        print(f"{omega:6.2f} {a2:14.6e} {b:21.6e} {r2:12.4e} {av:16.6e} {rv:12.6f}")
    return rows


# --- part 2: how much physics does the clamp truncate? ----------------------


def maxwell_pair_cdf(u_c):
    """P(|vr| < u_c * v_ref) for a pair drawn from an equilibrium Maxwellian.

    The relative velocity of two particles of mass m at temperature T is itself
    Maxwellian with the reduced mass m_r = m/2, so with u = |vr| / sqrt(2kT/m_r)
    the density is 4/sqrt(pi) u^2 exp(-u^2) -- species- and temperature-free.
    """
    return special.erf(u_c) - 2.0 / math.sqrt(math.pi) * u_c * math.exp(-u_c * u_c)


def maxwell_pair_cdf_small(u_c):
    """Leading term, 4/(3 sqrt(pi)) u_c^3."""
    return 4.0 / (3.0 * math.sqrt(math.pi)) * u_c**3


def truncated_fraction(eps_rel, t_over_tref):
    """Fraction of equilibrium pairs below the clamp at gas temperature ``T``.

    The clamp is pinned to ``T_ref``, the gas to ``T``: ``u_c = sqrt(eps_rel * T_ref / T)``.
    """
    return maxwell_pair_cdf(math.sqrt(eps_rel / t_over_tref))


def mc_truncated_fraction(eps_rel, temp, key, n_pairs):
    """Monte-Carlo cross-check of :func:`truncated_fraction` in SI units."""
    k1, k2 = jax.random.split(key)
    scale = math.sqrt(K_BOLTZ * temp / AR_MASS)
    vi = scale * jax.random.normal(k1, (n_pairs, 3))
    vj = scale * jax.random.normal(k2, (n_pairs, 3))
    vr2 = jnp.sum((vi - vj) ** 2, axis=-1)
    eps_abs = eps_rel * (2.0 * K_BOLTZ * AR_TREF / AR_MR)
    return float(jnp.mean(vr2 <= eps_abs)), float(jnp.sqrt(jnp.min(vr2)))


def part2_truncation(key):
    print("\n=== U1 part 2: what the clamp truncates in an equilibrium Maxwellian ===")
    print("Fraction of pairs with |vr| below the clamp.  Analytic, exact:")
    print(f"{'EPS_REL_VR2':>12} {'u_c=|vr|/v_ref':>15} {'T=T_ref':>12} {'T=T_ref/10':>12} {'T=T_ref/100':>12}")
    for e in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12]:
        print(
            f"{e:12.0e} {math.sqrt(e):15.4e} {truncated_fraction(e, 1.0):12.3e} "
            f"{truncated_fraction(e, 0.1):12.3e} {truncated_fraction(e, 0.01):12.3e}"
        )
    n = 20_000_000
    frac, vmin = mc_truncated_fraction(EPS_REL_VR2, 300.0, key, n)
    v_ref300 = math.sqrt(2.0 * K_BOLTZ * 300.0 / AR_MR)
    print(f"\nMonte Carlo, argon, T = 300 K, {n:,} pairs:")
    print(f"  clamp at |vr| = {math.sqrt(EPS_REL_VR2 * 2 * K_BOLTZ * AR_TREF / AR_MR):.4e} m/s")
    print(f"  min sampled |vr|            = {vmin:.4e} m/s  (u = {vmin / v_ref300:.3e})")
    print(f"  measured truncated fraction = {frac:.3e}   (analytic {truncated_fraction(EPS_REL_VR2, 300.0 / AR_TREF):.3e})")
    print(f"  smallest eps_rel this sample could have detected: {(vmin / v_ref300) ** 2 * 300.0 / AR_TREF:.3e}")
    u_exp = (1.0 / (4.0 / (3.0 * math.sqrt(math.pi)) * n)) ** (1.0 / 3.0)
    print(f"  expected min u over {n:,} pairs (analytic) = {u_exp:.3e} -> |vr| = {u_exp * v_ref300:.3f} m/s")
    print("  (section 2.3 quotes min 1.49 m/s over 1e8 samples; the analytic expectation there is 1.19 m/s)")


# --- part 3: how far above the population does the clamped gradient sit? ----


def pop_rms_grad_analytic(omega):
    """rms of ||grad_vr q|| over an unclamped equilibrium Maxwellian, prefactor-free.

    E[u^k] = (2/sqrt(pi)) Gamma((3+k)/2) for the density 4/sqrt(pi) u^2 exp(-u^2);
    here k = 2 - 4 omega, so the second moment converges iff omega < 1.25.
    """
    k = 2.0 - 4.0 * omega
    a = (3.0 + k) / 2.0
    if a <= 0.0:
        return math.nan
    m2 = 2.0 / math.sqrt(math.pi) * math.gamma(a)
    return 2.0 * abs(1.0 - omega) * math.sqrt(m2)


def pop_rms_grad_clamped(omega, eps_rel, key, n_pairs):
    """Same rms, measured on a clamped sample (finite for every omega)."""
    k1, k2 = jax.random.split(key)
    vi = jax.random.normal(k1, (n_pairs, 3)) / math.sqrt(2.0)
    vj = jax.random.normal(k2, (n_pairs, 3)) / math.sqrt(2.0)
    g = _d_dvrvec(vi - vj, float(omega), float(eps_rel), 1.0)
    return float(jnp.sqrt(jnp.mean(jnp.sum(g * g, axis=-1))))


def dominance(omega, eps_rel):
    """Worst-case (at-the-clamp) gradient divided by the equilibrium rms.

    The |1-omega| factor cancels, so this is a pure function of (omega, eps):
    ``eps^(0.5-omega) / sqrt((2/sqrt(pi)) Gamma((5-4 omega)/2))``.
    """
    r = pop_rms_grad_analytic(omega)
    if math.isnan(r) or r == 0.0:
        return math.nan
    return stated_bound(omega, eps_rel, 1.0) / r


def part3_dominance(key):
    print("\n=== U1 part 3: gradient dominance of a single at-the-clamp pair ===")
    print("D = max||grad_v q|| / rms(||grad_v q||) over equilibrium pairs; prefactor cancels.")
    print("A pair sitting on the clamp (a duplicated parcel, a zero-T IC) contributes D times")
    print("the typical pair, so it swamps a sum of M pairs whenever D >~ sqrt(M).")
    print(f"\n{'EPS_REL':>10}" + "".join(f"{f'w={w:.2f}':>12}" for w in [0.6, 0.81, 0.95, 0.99, 1.1, 1.2]))
    for e in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12]:
        cells = "".join(f"{dominance(w, e):12.3e}" for w in [0.6, 0.81, 0.95, 0.99, 1.1, 1.2])
        print(f"{e:10.0e}{cells}")
    print("\nomega >= 1.25: E[||grad q||^2] diverges over an unclamped Maxwellian")
    print("(the exponent 2-4*omega makes int u^2 u^(2-4w) e^-u^2 du divergent at u->0),")
    print("so for those omega the clamp is the *sole* regulator of the gradient variance:")
    print(f"{'omega':>6} {'analytic rms':>14} {'clamped rms (MC)':>18}")
    for w in [1.2, 1.25, 1.3, 1.5]:
        a = pop_rms_grad_analytic(w)
        m = pop_rms_grad_clamped(w, EPS_REL_VR2, key, 400_000)
        print(f"{w:6.2f} {a:14.4e} {m:18.4e}")


def part4_recommendation():
    print("\n=== U1 part 4: the trade, side by side ===")
    print("truncation ~ eps^1.5 (cheap to buy), dominance ~ eps^(0.5-omega) (dear to buy).")
    print(
        f"{'EPS_REL':>10} {'trunc T=Tref':>14} {'trunc T=Tref/10':>16} {'trunc T=Tref/100':>17} "
        f"{'D(w=0.81)':>11} {'D(w=0.99)':>11} {'D(w=1.2)':>11}"
    )
    for e in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        print(
            f"{e:10.0e} {truncated_fraction(e, 1.0):14.2e} {truncated_fraction(e, 0.1):16.2e} "
            f"{truncated_fraction(e, 0.01):17.2e} {dominance(0.81, e):11.3e} "
            f"{dominance(0.99, e):11.3e} {dominance(1.2, e):11.3e}"
        )


def main():
    key = jax.random.PRNGKey(0)
    pref_ar = prefactor(AR_DIAM, AR_TREF, AR_MR, AR_OMEGA)
    v_ref2 = 2.0 * K_BOLTZ * AR_TREF / AR_MR
    print("GATE U1 -- EPS_REL_VR2 and the collision-rate derivative bound")
    print(f"argon VSS: diam={AR_DIAM:.3e} omega={AR_OMEGA} tref={AR_TREF} alpha={AR_ALPHA}")
    print(f"m_r = {AR_MR:.4e} kg, v_ref = sqrt(2kT_ref/m_r) = {math.sqrt(v_ref2):.4f} m/s")
    print(f"prefactor(argon) = {pref_ar:.6e},  eps_abs = EPS_REL_VR2*v_ref^2 = {EPS_REL_VR2 * v_ref2:.6e} m^2/s^2")

    # dimensionless pass: v_ref = 1, prefactor = 1, so eps = EPS_REL_VR2 exactly
    rows_nd = part1_bound_sweep(EPS_REL_VR2, 1.0, "dimensionless, u = |vr|/v_ref, prefactor = 1")
    # SI pass: the same inequality with real units, to expose its unit-dependence
    rows_si = part1_bound_sweep(EPS_REL_VR2 * v_ref2, pref_ar, "SI, argon prefactor")

    print("\n--- verdict on the section-19 inequality ---")
    worst_nd = min(r[3] for r in rows_nd)
    worst_si = min(r[3] for r in rows_si)
    print(f"min over omega of bound/achieved for |dq/dvr2|: dimensionless {worst_nd:.4e}, SI {worst_si:.4e}")
    print(f"both equal 2*sqrt(eps): {2 * math.sqrt(EPS_REL_VR2):.4e} and {2 * math.sqrt(EPS_REL_VR2 * v_ref2):.4e}")
    print("=> the stated bound is BELOW the achieved max, i.e. the assertion as written is FALSE.")
    tight_nd = [r[5] for r in rows_nd]
    print(f"for ||grad_vr q|| the same formula gives bound/achieved in [{min(tight_nd):.6f}, {max(tight_nd):.6f}]")
    print("=> the formula is the exact, attained sup of the AD-reachable gradient. Right bound, wrong LHS.")

    part2_truncation(key)
    part3_dominance(jax.random.fold_in(key, 1))
    part4_recommendation()

    print("\n=== RECOMMENDATION ===")
    for e in (1e-8, 1e-6):
        print(
            f"  eps_rel={e:.0e}: truncates {truncated_fraction(e, 1.0):.2e} of equilibrium pairs at T=T_ref "
            f"({truncated_fraction(e, 0.01):.2e} at T=T_ref/100); dominance D(0.81)={dominance(0.81, e):.3e}, "
            f"D(1.2)={dominance(1.2, e):.3e}"
        )
    print("  -> raise EPS_REL_VR2 from 1e-8 to 1e-6.  See the report.")


if __name__ == "__main__":
    main()
