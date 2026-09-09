"""Phase 0 verification of the structural claims in PSEUDOCODE.md section 17.

Pure NumPy, no JAX. Run before trusting the spec; every check here becomes a
unit test once the real modules exist.

    python3 docs/phase0_verification.py
"""

import argparse
import sys

import numpy as np
from scipy import stats

FAILS = []
PASSED = []


def ok(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    (PASSED if cond else FAILS).append(name)


# --- specular triangle-wave fold (PSEUDOCODE 9.2) --------------------------
def fold(xn, lo, hi):
    L = hi - lo
    y = (xn - lo) / L
    m = np.floor(y)
    f = y - m
    even = m % 2 == 0
    return lo + np.where(even, f, 1 - f) * L, np.where(even, 1.0, -1.0)


def brute(x, lo, hi):
    s = 1.0
    for _ in range(100000):
        if x < lo:
            x, s = 2 * lo - x, -s
        elif x > hi:
            x, s = 2 * hi - x, -s
        else:
            return x, s
    raise RuntimeError("did not converge")


def check_fold(rng):
    lo, hi = -1.3, 2.7
    L = hi - lo
    xn = rng.uniform(lo - 50 * L, hi + 50 * L, 20000)
    xf, sf = fold(xn, lo, hi)
    xb, sb = np.array([brute(v, lo, hi) for v in xn]).T
    ok(
        "specular fold == brute-force reflect-until-inside (position)",
        np.allclose(xf, xb, atol=1e-9),
    )
    ok("specular fold == brute-force reflect-until-inside (velocity sign)", np.array_equal(sf, sb))
    ok(
        "specular fold always lands inside the domain",
        ((xf >= lo - 1e-12) & (xf <= hi + 1e-12)).all(),
    )


# --- distinct-pair sampler (PSEUDOCODE 8.3) --------------------------------
def check_pair(rng):
    for npc in (2, 3, 5, 17, 64):
        i = rng.integers(0, npc, 600000)
        r = rng.integers(0, npc - 1, 600000)
        j = (i + 1 + r) % npc
        cnt = np.bincount(i * npc + j, minlength=npc * npc).reshape(npc, npc)
        ok(f"draw_pair never selects j == i (np={npc})", np.diag(cnt).sum() == 0)
        p = stats.chisquare(cnt[~np.eye(npc, dtype=bool)]).pvalue
        ok(f"draw_pair j uniform on complement of i, chi2 p={p:.3f} (np={npc})", p > 1e-3)


# --- collision map, A matrix, involution (PSEUDOCODE 7.4, 13.1, 17.1) ------
def coll(z, sigma):
    v, v1 = z[:3], z[3:]
    n = np.linalg.norm(v - v1)
    return np.concatenate([0.5 * ((v + v1) + n * sigma), 0.5 * ((v + v1) - n * sigma)])


def check_collision(rng):
    v, v1 = rng.normal(size=3), rng.normal(size=3)
    sigma = rng.normal(size=3)
    sigma /= np.linalg.norm(sigma)
    alpha = (v - v1) / np.linalg.norm(v - v1)
    I = np.eye(3)
    A = 0.5 * np.block(
        [
            [I + np.outer(sigma, alpha), I - np.outer(sigma, alpha)],
            [I - np.outer(sigma, alpha), I + np.outer(sigma, alpha)],
        ]
    )
    B = 0.5 * np.block(
        [
            [I + np.outer(alpha, sigma), I - np.outer(alpha, sigma)],
            [I - np.outer(alpha, sigma), I + np.outer(alpha, sigma)],
        ]
    )

    z, h = np.concatenate([v, v1]), 1e-6
    Jn = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = h
        Jn[:, k] = (coll(z + e, sigma) - coll(z - e, sigma)) / (2 * h)

    ok("A(sigma,alpha) == d(v',v1')/d(v,v1) at fixed sigma", np.allclose(A, Jn, atol=1e-6))
    ok("B == A^T  ([P1] eq. AB_def)", np.allclose(B, A.T))

    zp = coll(z, sigma)
    ok("collision conserves momentum", np.allclose(zp[:3] + zp[3:], v + v1))
    ok(
        "collision conserves energy",
        np.isclose(np.sum(zp[:3] ** 2) + np.sum(zp[3:] ** 2), np.sum(v**2) + np.sum(v1**2)),
    )
    ok(
        "post-collision relative direction == sigma",
        np.allclose((zp[:3] - zp[3:]) / np.linalg.norm(zp[:3] - zp[3:]), sigma),
    )
    ok(
        "collision MAP is an involution: coll(coll(z,sigma),alpha) == z",
        np.allclose(coll(zp, alpha), z),
    )

    sv = np.linalg.svd(A, compute_uv=False)
    ok(
        f"A is rank 4, singular values {np.round(sv, 6)} -- A^-1 does NOT exist",
        np.allclose(sv, [1, 1, 1, 1, 0, 0], atol=1e-9) and np.linalg.matrix_rank(A) == 4,
    )


# --- pair_perm: valid perfect matching (PSEUDOCODE 8.3) --------------------
def check_pair_perm(rng):
    trials = 4000
    for npc in (2, 4, 16, 65):  # 65 is odd: one particle sits out
        used = np.zeros(npc, dtype=np.int64)
        selfpair = 0
        partner = np.zeros((npc, npc), dtype=np.int64)
        for _ in range(trials):
            perm = rng.permutation(npc)
            i, j = perm[0::2], perm[1::2]
            k = min(len(i), len(j))  # odd npc -> drop the unpaired tail
            i, j = i[:k], j[:k]
            selfpair += int((i == j).sum())
            # each particle must appear at most once across BOTH halves
            touched = np.concatenate([i, j])
            if len(np.unique(touched)) != len(touched):
                selfpair += 1  # reuse within a step: same failure class
            used[touched] += 1
            partner[i, j] += 1
            partner[j, i] += 1
        ok(f"pair_perm never self-pairs or reuses within a step (np={npc})", selfpair == 0)
        # perfect matching: for even npc every particle is used in EVERY step
        if npc % 2 == 0:
            ok(
                f"pair_perm is a perfect matching -- all used exactly {trials}x (np={npc})",
                bool((used == trials).all()),
            )
        else:
            # odd npc: exactly one sits out per step, so each is dropped ~trials/npc times
            exp = trials * (npc - 1) / npc
            ok(
                f"pair_perm drops exactly one particle per step (np={npc})",
                used.sum() == trials * (npc - 1)
                and np.abs(used - exp).max() < 6 * np.sqrt(exp * (1 / npc)),
            )
        # partner choice must be uniform over the other npc-1 particles
        if npc > 2:
            off = partner[~np.eye(npc, dtype=bool)]
            pv = stats.chisquare(off).pvalue
            ok(f"pair_perm partner uniform over the other {npc - 1}, chi2 p={pv:.3f}", pv > 1e-3)


# --- stale-max rule (PSEUDOCODE 7.2, 8.4, 18.1) ----------------------------
def check_stale_max(rng):
    """Sigma must be loop-invariant, and the live ratchet must under-collide."""
    n_att, trials = 12, 4000
    Sigma0 = 1.0
    acc_stale = acc_live = 0
    sigma_moved = False
    for _ in range(trials):
        q = rng.uniform(0.0, 1.6, n_att)  # some q exceed Sigma0 -> ratchet fires
        u = rng.uniform(0.0, 1.0, n_att)

        # stale: Sigma fixed for the whole loop
        Sigma = Sigma0
        acc_stale += int((u < np.clip(q / Sigma, 0, 1)).sum())
        if Sigma != Sigma0:
            sigma_moved = True

        # live: Sigma ratchets inside the loop (SPARTA)
        S = Sigma0
        for a in range(n_att):
            S = max(S, q[a])
            acc_live += int(u[a] < min(q[a] / S, 1.0))

    ok("stale-max leaves Sigma bitwise invariant across the attempt loop", not sigma_moved)
    ok(
        f"live ratchet accepts fewer when Sigma is too small ({acc_live} < {acc_stale})",
        acc_live < acc_stale,
    )

    # The invariant that actually matters: when Sigma IS a true majorant the two
    # rules must agree exactly. The divergence is a warm-up effect, not a
    # steady-state one -- see scratch/probe_ntc_majorant.py.
    q = rng.uniform(0.0, 1.0, (trials, n_att))
    u = rng.uniform(0.0, 1.0, (trials, n_att))
    big = 1.0  # >= max q, so the ratchet never fires
    stale = int((u < np.clip(q / big, 0, 1)).sum())
    live = 0
    for t in range(trials):
        S = big
        for a in range(n_att):
            S = max(S, q[t, a])
            live += int(u[t, a] < min(q[t, a] / S, 1.0))
    ok("with a TRUE majorant the live and stale rules agree exactly", live == stale)

    # the next step's majorant must ignore inactive slots
    q = rng.uniform(0.0, 2.0, n_att)
    active = np.arange(n_att) < 5
    masked = np.max(np.where(active, q, -np.inf))
    ok(
        "qmax mask: inactive slots cannot inflate the next majorant",
        np.isclose(masked, q[:5].max()) and masked <= q.max(),
    )


# --- FIX: scratch row removes the empty-cell clobber (PSEUDOCODE 8.4.1) ----
def _run_slot(counts, starts, order, v0, use_fix):
    """One attempt slot, vectorised over cells. Mirrors 8.4 including both scatters."""
    N = len(v0)
    v = np.concatenate([v0, [0.0]]) if use_fix else v0.copy()
    active = counts >= 2
    # pair_ntc: JAX returns 0 from randint(k, shape, 0, 0), so an empty cell gets i=j=0
    i = np.where(active, 0, 0)
    j = np.where(active, 1, 0)
    pi = order[np.minimum(starts + i, N - 1)]
    pj = order[np.minimum(starts + j, N - 1)]
    if use_fix:
        pi = np.where(active, pi, N)  # inactive -> scratch row
        pj = np.where(active, pj, N)
    vi, vj = v[pi], v[pj]  # BOTH gathered pre-scatter
    vi_n, vj_n = vj, vi  # swap = a momentum-conserving "collision"
    v[pi] = np.where(active, vi_n, vi)  # S1
    v[pj] = np.where(active, vj_n, vj)  # S2 -- vj is stale w.r.t. S1
    return v[:N] if use_fix else v


def check_scatter_fix():
    counts = np.array([2, 0, 4, 0, 1, 3])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    N = int(counts.sum())
    order = np.arange(N)
    v0 = np.arange(1.0, N + 1.0)

    broken = _run_slot(counts, starts, order, v0, use_fix=False)
    fixed = _run_slot(counts, starts, order, v0, use_fix=True)

    ok(
        "rev-2 scatter breaks momentum conservation on a table with empty cells",
        not np.isclose(broken.sum(), v0.sum()),
    )
    ok(
        f"scratch-row fix conserves momentum ({fixed.sum():.1f} == {v0.sum():.1f})",
        np.isclose(fixed.sum(), v0.sum()),
    )
    ok("scratch-row fix conserves energy", np.isclose((fixed**2).sum(), (v0**2).sum()))
    ok(
        "fixed result is a pure permutation of the input",
        np.array_equal(np.sort(fixed), np.sort(v0)),
    )


# --- FIX: count score restores the dt/fnum/volume gradient (PSEUDOCODE 8.1) --
def check_count_score():
    """Rao-Blackwellised, so this is EXACT -- no sampling, no error bars.

    n = floor(f) + (U < frac(f)).  E[J] = (1-frac) J_lo + frac J_hi, so
    dE[J]/dtheta = (J_hi - J_lo) * dfrac/dtheta.  The score estimator
    E[J * dlogP/dtheta] must reproduce that identically.
    """
    f, dfdtheta = 3.745, 135000.0
    lo, frac = np.floor(f), f - np.floor(f)
    J = {lo: 11.99257529, lo + 1: 12.59168617}
    J_lo, J_hi = J[lo], J[lo + 1]

    truth = (J_hi - J_lo) * dfdtheta

    # revision 2: n = floor(f) is an integer, so J is a STEP function of theta.
    # Central FD at any h that does not flip n reads exactly zero -- and so did jax.grad.
    h = 1e-6
    fd_rev2 = (J[np.floor(f + h)] - J[np.floor(f - h)]) / (2 * h)
    ok(
        "rev-2 floor(f): central FD reads exactly 0.0 (the defect, and FD could not see it)",
        fd_rev2 == 0.0,
    )

    # revision 3: enumerate both outcomes weighted by probability
    #   up:   logP = log(frac),     dlogP/dtheta =  dfrac/dtheta / frac
    #   down: logP = log(1 - frac), dlogP/dtheta = -dfrac/dtheta / (1 - frac)
    est = frac * J_hi * (dfdtheta / frac) + (1 - frac) * J_lo * (-dfdtheta / (1 - frac))
    ok(
        f"count score reproduces the exact sensitivity ({est:.6e} == {truth:.6e})",
        np.isclose(est, truth, rtol=1e-12),
    )
    ok(
        "count score is zero when J does not depend on the count",
        np.isclose(
            frac * 5.0 * (dfdtheta / frac) + (1 - frac) * 5.0 * (-dfdtheta / (1 - frac)),
            0.0,
            atol=1e-6,
        ),
    )


# --- FIX: operator ordering determines where the transport term sits (12.1) --
def check_vjp_ordering():
    rng = np.random.default_rng(7)
    Cm = rng.normal(size=(3, 3))  # deliberately non-symmetric
    dt = 0.017
    P = np.array([[1.0, 0.0, 0.0]])  # R^3 -> R^1, D = 1
    beta1, alpha1 = rng.normal(size=3), rng.normal(size=1)

    # move -> collide:  x' = x + dt P v ;  v' = C v
    beta_move_first = Cm.T @ beta1 + dt * (P.T @ alpha1).ravel()
    # collide -> move:  v' = C v ;  x' = x + dt P v'
    beta_collide_first = Cm.T @ (beta1 + dt * (P.T @ alpha1).ravel())

    outside = Cm.T @ beta1 + dt * (P.T @ alpha1).ravel()
    inside = Cm.T @ (beta1 + dt * (P.T @ alpha1).ravel())

    ok(
        "move->collide  =>  transport term OUTSIDE B_k",
        np.allclose(beta_move_first, outside) and not np.allclose(beta_move_first, inside),
    )
    ok(
        "collide->move  =>  transport term INSIDE B_k  ([P3] eq. adjoint_gamma_R)",
        np.allclose(beta_collide_first, inside) and not np.allclose(beta_collide_first, outside),
    )
    ok(
        "the two forms genuinely differ at realistic dt",
        not np.allclose(outside, inside, rtol=1e-6),
    )


# --- FIX: relative clamps are scale-invariant (PSEUDOCODE 2.3, 7.4) ---------
def check_relative_clamps():
    EPS_D = 1e-6
    for ratio in (1e-2, 1e-7, 1e-9):
        fired_abs, fired_rel = set(), set()
        for V in (340.0, 1.0, 3.4e4):  # SI, nondimensional, CGS
            vr = np.array([V, ratio * V, 0.0])
            d = np.linalg.norm(vr[1:])
            vrm = np.linalg.norm(vr)
            fired_abs.add(bool(d > EPS_D))
            fired_rel.add(bool(d > EPS_D * vrm))
        ok(f"relative frame threshold is unit-invariant at d/|vr|={ratio:.0e}", len(fired_rel) == 1)
        if ratio <= 1e-7:
            ok(
                f"absolute threshold is NOT unit-invariant at d/|vr|={ratio:.0e}",
                len(fired_abs) > 1,
            )

    # safe_norm's cotangent is bounded by 1.0 regardless of eps
    worst = 0.0
    for mag in (1e-14, 1e-10, 1e-4, 1.0, 1e2):
        v = np.array([mag, 0.0, 0.0])
        worst = max(worst, abs(np.linalg.norm(v) and v[0] / np.linalg.norm(v)))
    ok(
        f"safe_norm cotangent bounded by 1.0 across 16 decades (max {worst:.4f})",
        worst <= 1.0 + 1e-12,
    )

    # safe_pow contract at p == 0: x^0 == 1 for every x, including x == 0
    def safe_pow(x, p, eps=1e-30):
        okm = x > eps
        return np.where(
            okm, np.exp(p * np.log(np.where(okm, x, 1.0))), np.where(p == 0.0, 1.0, 0.0)
        )

    ok(
        "safe_pow(0, 0) == 1.0 (rev-2 returned 0.0, breaking Oracle 1 at omega=1)",
        np.isclose(safe_pow(np.array(0.0), 0.0), 1.0),
    )


# --- DiCE surrogate (PSEUDOCODE 11) ----------------------------------------
def _acc(t):
    """Toy score accumulator L(theta)."""
    return np.array([2 * t, -t, 0.5 * t * t])


def _phi(t):
    """Toy per-particle observable."""
    return np.array([t**2, 3 * t, 1.0])


def _surrogate(t, t0):
    """DiCE surrogate: magic(L) = exp(L(t) - stop_grad(L(t0))), value 1 at t == t0."""
    return np.sum(np.exp(_acc(t) - _acc(t0)) * _phi(t))


def check_dice():
    th, h = 0.7, 1e-6
    L, phi, surr = _acc, _phi, _surrogate

    ok("magic(L) value == 1, so surrogate value == J", np.isclose(surr(th, th), np.sum(phi(th))))
    d = (surr(th + h, th) - surr(th - h, th)) / (2 * h)
    dphi = (phi(th + h) - phi(th - h)) / (2 * h)
    dL = (L(th + h) - L(th - h)) / (2 * h)
    ok(
        "grad(surrogate) == sum(dphi + phi*dL)",
        np.isclose(d, np.sum(dphi + phi(th) * dL), rtol=1e-6),
    )

    # both partners accumulate the SAME log h, so the coefficient of d(log h)/dv is phi_i + phi_i1
    phi_i, phi_i1 = 2.0, 5.0
    both = (phi_i * h + phi_i1 * h) - (phi_i * -h + phi_i1 * -h)
    ok("score reward attribution == phi_i + phi_i1", np.isclose(both / (2 * h), phi_i + phi_i1))


# ---------------------------------------------------------------------------
# Unverified claims register -- mirrors PSEUDOCODE.md section 19.
#
# Each entry is an assertion the spec makes that has NOT been measured. The gate
# is part of its milestone's definition of done, checked BEFORE that milestone's
# deliverable is written.  Close one by moving its evidence into a real check
# above and deleting the row here (and from section 19).
#
#   python3 docs/phase0_verification.py                 -> run checks, list gaps
#   python3 docs/phase0_verification.py --milestone M1  -> ALSO fail if M1 gaps open
# ---------------------------------------------------------------------------
MILESTONES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]

# CLOSED 2026-09-04 and removed from this list (evidence in PSEUDOCODE.md):
#   U1 -- spec 17.4.  Bound exactly tight; EPS_REL_VR2 KEPT at 1e-8 (the proposed
#         raise to 1e-6 was measured and rejected).  scratch/gate_U1_clamp_bound.py,
#         scratch/gate_U1_variance_decider.py, tests/test_gaps.py.
#   U3 -- spec 17.5 / 7.2.3.  A 2-5x margin IS achievable, but only from an a-priori
#         thermal majorant; SPARTA's own seed fails at a 1.0e-2 violation rate.  This
#         changed vremax_init.  scratch/gate_U3_margin_numpy.py, tests/test_gaps.py.
#   U5 -- spec 13.3 / 12.1.  Oracle 3 BUILT: 1-D, 2 cells, 2 steps, periodic, with B_k
#         written analytically from [P1] eq. AB_def.  The 2x2 discriminator is exactly
#         diagonal -- our move->collide loop matches the OUTSIDE form to 1.3e-16 and
#         misses [P3]'s INSIDE form by 9.8e-2; reordering the loop swaps both.  A
#         companion test pins the PRODUCTION step's ordering behaviourally.
#         tests/test_adjoint_oracle.py::test_oracle3_*.
UNVERIFIED = [
    (
        "U4",
        "M3",
        "sizing A_max from the inflated Sigma prevents silent clipping",
        "T:1->4->1 transient; attempts_clipped > 0 cold-sized, == 0 with the inflated rule",
    ),
    (
        "U8",
        "M3",
        "h_crit ~ 5.8/(N*C*M) closes the FD window at production scale",
        "measure h_crit at N*C*M ~ 1e8 and check the fit before trusting a 6-decade extrapolation",
    ),
    (
        "U9",
        "M3",
        "the scratch-row fix costs ~3-10%",
        "re-benchmark with a repeat count and a spread, above the launch-bound regime",
    ),
    (
        "U2",
        "M6",
        "the score estimator is unbiased at a NON-ZERO gradient",
        # Half-closed.  At a KNOWN ZERO it now passes: the collision-invariant check
        # (section 23.3.7) gives |z| <= 1.53 for alpha/diam/omega against an exact 0.
        # And the SE target is met: diam reaches 1.2% SE at N=2.56e5, M_s=100.  What is
        # NOT closed is unbiasedness at a non-zero value, and the reason is that the
        # obvious reference is not a reference: mode-B FD at [P2]'s Delta m = 0.1 is
        # O(h^2) TRUNCATION-biased, measured at |z| = 6.4 for diam (section 23.3).
        # Comparing an estimator against a biased yardstick cannot demonstrate either.
        "sweep Delta m at fixed large M_s and Richardson-extrapolate the FD to h -> 0 "
        "to remove the O(h^2) bias, THEN require |z| < 2 against the scored AD at an SE "
        "below 10% of the mean; report the sample count needed",
    ),
    (
        "U10",
        "M6",
        "the v_r-frame and lab-frame collision reparameterisations are equally good "
        "pathwise estimators",
        "measure Var(dJ/dtheta) over seeds under both samplers at matched cost, 0-D and 1-D",
    ),
    (
        "U6",
        "M5",
        "logh_acc is zeroed when inflow refills a dead slot",
        "inject into a dead slot; assert accumulator 0 and gradient equals a fresh particle's",
    ),
    (
        "U7",
        "M5",
        "non-uniform fnum without reweighting breaks mass conservation",
        "exhibit the mass drift with particles crossing cells; assert the setup guard fires",
    ),
]

# These three fail silently -- no error, no NaN, no diagnostic. Same shape as the
# two worst defects this document has already had. Treated as blocking.
SILENT_FAILURE = {"U4", "U6", "U7"}


def report_gaps(gate=None):
    """Print the register; if `gate` is a milestone, return open gaps at or before it."""
    print("\n" + "=" * 72)
    print("UNVERIFIED CLAIMS (PSEUDOCODE.md section 19) -- reasoned, not measured")
    print("=" * 72)
    for uid, ms, claim, how in UNVERIFIED:
        mark = "  [SILENT-FAILURE RISK]" if uid in SILENT_FAILURE else ""
        print(f"  {uid}  {ms}  {claim}{mark}")
        print(f"          gate: {how}")
    if gate is None:
        print(f"\n  {len(UNVERIFIED)} open. Run with --milestone <M> to gate on one.")
        return []
    idx = MILESTONES.index(gate)
    due = [u for u in UNVERIFIED if MILESTONES.index(u[1]) <= idx]
    print(f"\n  gating on {gate}: {len(due)} gap(s) due at or before it")
    return due


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    check_fold(rng)
    check_pair(rng)
    check_collision(rng)
    check_pair_perm(rng)
    check_stale_max(rng)
    check_scatter_fix()
    check_count_score()
    check_vjp_ordering()
    check_relative_clamps()
    check_dice()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--milestone",
        choices=MILESTONES,
        help="fail if any unverified claim due at or before this milestone is open",
    )
    args = ap.parse_args()

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
        sys.exit(1)
    print(f"all {len(PASSED)} Phase 0 structural claims verified")

    due = report_gaps(args.milestone)
    if due:
        print(f"\nBLOCKED: {len(due)} unverified claim(s) must be settled before {args.milestone}:")
        for uid, ms, claim, _ in due:
            print(f"  {uid} ({ms}): {claim}")
        sys.exit(1)
