"""JAX accuracy of the Halley flux quantile vs the 64-step bisection and the mpmath truth."""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax_dsmc import inflow

@jax.jit
def bisect(u, s):
    u, s = jnp.broadcast_arrays(u, s)
    lo = jnp.zeros_like(u); hi = jnp.maximum(s, 0.0) + 12.0
    def body(_, b):
        a, c = b; mid = (a + c) / 2; left = inflow.flux_cdf(mid, s) < u
        return jnp.where(left, mid, a), jnp.where(left, c, mid)
    lo, hi = jax.lax.fori_loop(0, 64, body, (lo, hi))
    return (lo + hi) / 2

newton = jax.jit(inflow.flux_quantile)
cdf = jax.jit(inflow.flux_cdf)
eps = np.finfo(float).eps
d = np.load("scratch/inflow_newton/truth.npz")
U, S, T = d["U"], d["S"], d["T"]
yb = np.asarray(bisect(U, S)); yn = np.asarray(newton(U, S))
eb = np.abs(yb / T - 1); en = np.abs(yn / T - 1)
rb = np.abs(np.asarray(cdf(yb, S)) - U); rn = np.abs(np.asarray(cdf(yn, S)) - U)
print("truth grid", U.size, "points")
regions = {"u<1e-9": U < 1e-9, "1e-9..1e-3": (U >= 1e-9) & (U < 1e-3), "mid": (U >= 1e-3) & (U <= 1 - 1e-3), "upper": U > 1 - 1e-3}
for k, m in regions.items():
    print(f"  {k:12s} rel err vs mp: newton max {en[m].max():.2e} median {np.median(en[m]):.1e} | bisect max {eb[m].max():.2e} median {np.median(eb[m]):.1e} | newton-bisect rel max {np.abs(yn[m]/yb[m]-1).max():.2e}")
print("  |cdf-u| max: newton %.2e bisect %.2e; accepted(<1e-10): newton %d/%d bisect %d/%d" % (rn.max(), rb.max(), (rn < 1e-10).sum(), U.size, (rb < 1e-10).sum(), U.size))
worse = en > 2 * eb + 4 * eps
print("  points where newton err > 2x bisect err + 4eps:", worse.sum(), "max newton err there %.1e" % (en[worse].max() if worse.any() else 0))
# dense grid
s1 = np.linspace(-10, 10, 801)
u1 = np.concatenate([np.logspace(np.log10(eps), -1, 200), np.linspace(0.1, 0.9, 401), 1 - np.logspace(-1, np.log10(eps), 200)])
Ug, Sg = np.meshgrid(u1, s1); Ug, Sg = Ug.ravel(), Sg.ravel()
yb = np.asarray(bisect(Ug, Sg)); yn = np.asarray(newton(Ug, Sg))
rn = np.abs(np.asarray(cdf(yn, Sg)) - Ug); rb = np.abs(np.asarray(cdf(yb, Sg)) - Ug)
rel = np.abs(yn / yb - 1)
print("dense grid", Ug.size, "points: finite", np.isfinite(yn).all(), "; |cdf-u| newton max %.2e bisect max %.2e; rejects newton %d bisect %d" % (rn.max(), rb.max(), (~(rn < 1e-10)).sum(), (~(rb < 1e-10)).sum()))
for lo, hi in [(eps, 1e-9), (1e-9, 1e-6), (1e-6, 1e-3), (1e-3, 1 - 1e-3), (1 - 1e-3, 1 - 1e-6), (1 - 1e-6, 1 - 1e-9), (1 - 1e-9, 1)]:
    m = (Ug >= lo) & (Ug < hi) if hi < 1 else (Ug >= lo)
    print(f"  u in [{lo:.0e}, {hi:.0e}): |y_new/y_bis-1| max {rel[m].max():.1e} p99 {np.quantile(rel[m], 0.99):.1e} median {np.median(rel[m]):.1e}; |cdf-u| newton max {rn[m].max():.1e} bisect max {rb[m].max():.1e}")
m = (np.abs(Sg) <= 5) & (Ug > 1e-6) & (Ug < 1 - 1e-6)
print("  |s|<=5, u in (1e-6, 1-1e-6): |y_new/y_bis-1| max %.1e" % rel[m].max())
