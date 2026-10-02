import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax_dsmc import inflow
import sys; sys.path.insert(0, "scratch/inflow_newton")
exec(open("scratch/inflow_newton/jax_accuracy.py").read().split("newton = jax.jit")[0].split("from jax_dsmc import inflow")[1])
d = np.load("scratch/inflow_newton/truth.npz"); U, S, T = d["U"], d["S"], d["T"]
yb = np.asarray(bisect(U, S)); yn = np.asarray(jax.jit(inflow.flux_quantile)(U, S))
eb = np.abs(yb / T - 1); en = np.abs(yn / T - 1)
w = np.where(en > 2 * eb + 1e-15)[0]
j = w[np.argsort(-en[w])[:15]]
for i in j: print(f"u={U[i]:.3e} s={S[i]:+.2f} T={T[i]:.6e} new={yn[i]:.6e} bis={yb[i]:.6e} en={en[i]:.1e} eb={eb[i]:.1e}")
import collections; print(collections.Counter(np.round(S[w],0)))
