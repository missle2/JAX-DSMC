"""How many base keys does the ensemble FD need, and how big must h be?

Mode B is the only check available for a score-carried parameter, so its cost has
to be measured rather than guessed.
"""
import sys; sys.path.insert(0, ".")
import time
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa

th = sa.reference_theta()
cs = sa.case(1)
NK = 512
keys = jax.random.split(jax.random.PRNGKey(101), NK)

t0 = time.time()
g = sa.ensemble_grads(th, cs, keys, chunk=64)
print(f"grads over {NK} keys in {time.time()-t0:.1f}s")
for name in ("diam", "omega", "temp", "alpha", "nrho"):
    m, s = sa.mean_sem(g[name])
    print(f"  AD {name:>6}: {m:14.6e} +- {s:.2e}   ({100*s/abs(m):.1f} % SE)")

j = np.asarray(sa.ensemble_values(th, cs, keys, chunk=64))
print(f"\nJ: mean {j.mean():.5e}  sd {j.std():.3e}  ({100*j.std()/j.mean():.2f} % per seed)")

print(f"\n{'param':>6} {'h':>7} {'FD':>15} {'+-':>10} {'AD':>15} {'|z|':>6} {'wall':>6}")
for name in ("diam", "temp"):
    ad, ad_s = sa.mean_sem(g[name])
    for h in (0.02, 0.06, 0.15):
        t0 = time.time()
        fd, fd_s = sa.fd_ensemble(th, cs, keys, name, h, chunk=64)
        z = (fd - ad) / np.hypot(fd_s, ad_s)
        print(f"{name:>6} {h:7.3f} {fd:15.6e} {fd_s:10.2e} {ad:15.6e} {abs(z):6.2f} "
              f"{time.time()-t0:5.1f}s")

# how the SE shrinks with sample count -> what n_keys is needed
print("\nSE scaling for diam (AD):")
for n in (32, 64, 128, 256, 512):
    m, s = sa.mean_sem(np.asarray(g["diam"])[:n])
    print(f"  n={n:4d}  mean {m:13.6e}  SE {s:9.2e}  ({100*s/abs(m):5.1f} %)")
