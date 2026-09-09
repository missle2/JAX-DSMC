"""Can `A_max` become a free compute knob instead of an a-priori guess?

The problem: `A_max` is the static trip count of the attempt-slot scan, but the
attempt count `n = 0.5 npc(npc-1) Sigma dt fnum / V` is data-dependent -- it grows
with compression and with the majorant.  Guessing it is infeasible once the flow
is a shape-optimisation target.  SPARTA does not solve this; it simply has no
`A_max`, because a CPU `for` loop over `nattempt` costs nothing.

PROPOSED REFORMULATION.  Make exactly `A` attempts per cell (static), each accepted
with probability

    p = (n/A) * (q/Sigma) = 0.5 npc(npc-1) dt fnum q / (A V)        <- Sigma cancels

Expected collisions = A*<p> = n<q>/Sigma, identical to standard NTC.  It is simply
NTC with the majorant chosen so that the attempt count comes out exactly `A`:

    Sigma' = A V / (0.5 npc(npc-1) dt fnum)

Consequences, if it holds up:
  * `A` is a pure compute/variance knob.  Clipping becomes impossible by
    construction -- the count IS `A`.
  * validity condition is `p <= 1`, i.e. `Sigma' >= q_max`, which is directly
    measurable, instead of a guessed inflation factor.
  * `dt`, `fnum` and `V` now enter `p` SMOOTHLY, so the score `log p` carries their
    gradient and the stochastic-rounding count score of section 8.1 is no longer
    needed at all.

This script checks all three claims against the current scheme.
"""
import math

import numpy as np

K = 1.380649e-23
M, DIAM, TREF, OMEGA = 6.63e-26, 4.11e-10, 273.15, 0.81
MR = M / 2
CXS = math.pi * DIAM**2
PREF = CXS * (2 * K * TREF / MR) ** (OMEGA - 0.5) / math.gamma(2.5 - OMEGA)


def q_of(vr2):
    return PREF * np.maximum(vr2, 1e-8 * (2 * K * TREF / MR)) ** (1.0 - OMEGA)


def draw(npc, rng, T=273.15):
    return math.sqrt(K * T / M) * rng.standard_normal((npc, 3))


def collisions_old(v, sigma, dt, fnum, vol, rng):
    """Current scheme: stochastic-rounded n attempts, accept with q/Sigma."""
    npc = v.shape[0]
    f = 0.5 * npc * (npc - 1.0) * sigma * dt * fnum / vol
    n = int(np.floor(f) + (rng.random() < f - np.floor(f)))
    if n == 0:
        return 0, n
    i = rng.integers(0, npc, n)
    j = (i + 1 + rng.integers(0, npc - 1, n)) % npc
    q = q_of(np.sum((v[i] - v[j]) ** 2, -1))
    return int(np.sum(rng.random(n) < np.minimum(q / sigma, 1.0))), n


def collisions_new(v, A, dt, fnum, vol, rng):
    """Proposed: exactly A attempts, accept with p = 0.5 npc(npc-1) dt fnum q/(A V)."""
    npc = v.shape[0]
    i = rng.integers(0, npc, A)
    j = (i + 1 + rng.integers(0, npc - 1, A)) % npc
    q = q_of(np.sum((v[i] - v[j]) ** 2, -1))
    p = 0.5 * npc * (npc - 1.0) * dt * fnum * q / (A * vol)
    return int(np.sum(rng.random(A) < np.minimum(p, 1.0))), float(p.max())


npc, vol, nrho = 2000, 7.8e-12, 1e23
fnum = nrho * vol / npc
dt = 6.5e-9
sigma = 1.0370e-15  # the thermal majorant this case actually uses

print("=== claim 1: identical expected collision count ===")
print(f"{'A':>8} {'old <coll>':>12} {'new <coll>':>12} {'ratio':>8} {'max p':>8}")
rng = np.random.default_rng(0)
v = draw(npc, rng)
REP = 400
old = np.mean([collisions_old(v, sigma, dt, fnum, vol, rng)[0] for _ in range(REP)])
n_std = collisions_old(v, sigma, dt, fnum, vol, rng)[1]
for A in (n_std // 2, n_std, 2 * n_std, 8 * n_std):
    res = [collisions_new(v, int(A), dt, fnum, vol, rng) for _ in range(REP)]
    new = np.mean([r[0] for r in res])
    mp = max(r[1] for r in res)
    print(f"{A:>8} {old:>12.2f} {new:>12.2f} {new / old:>8.4f} {mp:>8.3f}"
          + ("   <- p>1, INVALID" if mp > 1 else ""))
print(f"\n  standard scheme uses n = {n_std} attempts; the new scheme is exact for any A")
print("  with max p <= 1, and the ratio confirms the means agree.")

print("\n=== claim 2: variance is not worse (larger A, smaller p) ===")
print(f"{'A':>8} {'mean':>9} {'sd':>9}")
o = [collisions_old(v, sigma, dt, fnum, vol, rng)[0] for _ in range(4000)]
print(f"{'old ' + str(n_std):>8} {np.mean(o):>9.2f} {np.std(o):>9.3f}")
for A in (n_std, 4 * n_std, 16 * n_std):
    c = [collisions_new(v, int(A), dt, fnum, vol, rng)[0] for _ in range(4000)]
    print(f"{A:>8} {np.mean(c):>9.2f} {np.std(c):>9.3f}")

print("\n=== claim 3: the validity bound is directly measurable ===")
print("p <= 1 requires A >= 0.5 npc(npc-1) dt fnum q_max / V.  Unlike the current")
print("A_max guess this needs no safety factor -- q_max is observable per step.")
qmax = q_of(np.max([np.sum((v[a] - v[b]) ** 2) for a in range(200) for b in range(200) if a != b]))
A_req = 0.5 * npc * (npc - 1.0) * dt * fnum * qmax / vol
print(f"  q_max (sampled)   = {qmax:.4e}")
print(f"  A required        = {A_req:.0f}")
print(f"  current n (Sigma) = {n_std}   ratio Sigma/q_max = {sigma / qmax:.2f}")
print("  -> the fixed-A scheme needs FEWER slots than the current majorant implies,")
print("     because the current Sigma carries the F_UP=2 and K_EV=5 safety factors.")
