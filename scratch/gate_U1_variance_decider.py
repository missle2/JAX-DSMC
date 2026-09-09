"""U1 decider: is EITHER eps effect non-negligible in absolute terms?

`gate_U1_clamp_bound.py` recommends raising EPS_REL_VR2 1e-8 -> 1e-6 on a
peak-ratio metric D = max/rms.  D is a *conditional* number: it says how bad one
at-the-clamp pair is, not how often one occurs.  The probability-weighted figure
runs the other way:

    P(clamped) ~ eps^1.5        (Maxwell speed pdf ~ u^2 near 0)
    ||grad q||^2 at clamp ~ eps^(1-2w)
    variance contribution ~ eps^(2.5-2w)   -> DECREASES with eps for w < 1.25

So for the argon regime (w ~ 0.75-0.81) the equilibrium argument prefers the
SMALLER eps and the peak argument prefers the LARGER one.  This script measures
both in absolute units so the decision rests on which is actually non-negligible
rather than on which metric is quoted.
"""
import numpy as np

rng = np.random.default_rng(20260904)

# --- q(vr) = (vr2)^(1-w) in units where v_ref = 1 (prefactor cancels in ratios)
def grad_norm(vr, w, eps):
    """||d q / d vr_vec|| with the relative clamp on vr2, v_ref = 1."""
    vr2 = np.sum(vr * vr, axis=-1)
    vr2c = np.maximum(vr2, eps)
    # q = vr2c^(1-w);  dq/dvr_i = (1-w) vr2c^(-w) * 2 vr_i   (zero below clamp)
    coef = np.where(vr2 > eps, (1.0 - w) * vr2c ** (-w) * 2.0, 0.0)
    return np.abs(coef) * np.sqrt(vr2)

def relative_speeds(n, rng):
    """|vr| for pairs from a unit-variance Maxwellian: vr ~ N(0, 2 I)."""
    return np.sqrt(2.0) * rng.standard_normal((n, 3))

M = 40_000_000
print(f"equilibrium Maxwellian pairs, M = {M:,};  v_ref = 1, prefactor = 1\n")
print(f"{'omega':>6} {'eps':>8} | {'mean|g|':>11} {'rms|g|':>11} {'max|g|':>11} "
      f"{'sd of mean':>11} {'n clamped':>9}")
print("-" * 84)
res = {}
for w in (0.75, 0.81, 1.00, 1.20, 1.30):
    for eps in (1e-8, 1e-6, 1e-4):
        tot = tot2 = 0.0; mx = 0.0; nclamp = 0
        for _ in range(8):
            vr = relative_speeds(M // 8, rng)
            g = grad_norm(vr, w, eps)
            tot += g.sum(); tot2 += (g * g).sum(); mx = max(mx, g.max())
            nclamp += int((np.sum(vr * vr, axis=-1) <= eps).sum())
        mean = tot / M; ms = tot2 / M
        var = max(ms - mean * mean, 0.0)
        sd_of_mean = np.sqrt(var / M)
        res[(w, eps)] = (mean, np.sqrt(ms), mx, sd_of_mean)
        print(f"{w:>6.2f} {eps:>8.0e} | {mean:>11.4e} {np.sqrt(ms):>11.4e} "
              f"{mx:>11.4e} {sd_of_mean:>11.4e} {nclamp:>9d}")
    print()

print("=== the decision quantity: MC standard error of a gradient estimator ===")
print("ratio of sd(mean) at eps=1e-6 to eps=1e-8.  >1 means 1e-6 is NOISIER.\n")
print(f"{'omega':>6} {'sd ratio 1e-6/1e-8':>20} {'predicted eps^(1.25-w)':>24}")
for w in (0.75, 0.81, 1.00, 1.20, 1.30):
    r = res[(w, 1e-6)][3] / res[(w, 1e-8)][3]
    pred = (1e-6 / 1e-8) ** (1.25 - w)
    print(f"{w:>6.2f} {r:>20.6f} {pred:>24.4f}")

print("\n=== absolute size of the effect ===")
print("relative standard error of the mean gradient, per M samples:\n")
print(f"{'omega':>6} {'eps':>8} {'rel.s.e. @M=4e7':>17} {'rel.s.e. @M=1e4':>17}")
for w in (0.75, 0.81, 1.20):
    for eps in (1e-8, 1e-6):
        mean, rms, mx, sdm = res[(w, eps)]
        rse = sdm / mean
        print(f"{w:>6.2f} {eps:>8.0e} {rse:>17.3e} {rse*np.sqrt(M/1e4):>17.3e}")
