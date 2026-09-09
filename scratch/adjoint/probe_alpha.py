"""Why does alpha's FD sit 40-1850x above its cancellation floor?

Truncation (O(h^2)) and round-off (O(1/h)) have opposite slopes in h, so a sweep
separates them.  A flat error would mean neither -- i.e. a real discrepancy.
"""
import sys; sys.path.insert(0, ".")
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa

th = sa.reference_theta()
key = jax.random.PRNGKey(3)
for dim in (1, 3):
    cs = sa.calibrate(sa.case(dim))
    _, g = sa.value_and_grad(th, cs, key, False)
    for name in ("alpha", "temp"):
        ad = float(g[name])
        print(f"\n--- D={dim}  {name}   AD = {ad:.10e}")
        print(f"    {'h':>10} {'FD':>18} {'flip':>6} {'|FD-AD|':>11} {'floor':>10} {'err/floor':>10}")
        h = 3e-4
        for _ in range(12):
            fd, flip, floor = sa.fd_central(th, cs, key, name, h)
            err = abs(fd - ad)
            print(f"    {h:10.2e} {fd:18.10e} {str(flip):>6} {err:11.3e} {floor:10.2e} "
                  f"{err / floor:10.1f}")
            h /= 3.0
