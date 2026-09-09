"""Throughput of the ensemble estimator, and how many keys U2 actually needs."""
import sys; sys.path.insert(0, ".")
import time
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa
print("devices:", jax.devices(), flush=True)

th = sa.reference_theta()
for dim, chunk in ((1, 128), (2, 64), (3, 32)):
    cs = sa.case(dim)
    keys = jax.random.split(jax.random.PRNGKey(101), chunk)
    jax.block_until_ready(sa.ensemble_grads(th, cs, keys, chunk=chunk))
    t0 = time.time(); jax.block_until_ready(sa.ensemble_grads(th, cs, keys, chunk=chunk))
    tg = time.time() - t0
    jax.block_until_ready(sa.ensemble_values(th, cs, keys, chunk=chunk))
    t0 = time.time(); jax.block_until_ready(sa.ensemble_values(th, cs, keys, chunk=chunk))
    tv = time.time() - t0
    print(f"D={dim} chunk={chunk}: grad {1e3*tg/chunk:7.2f} ms/key   "
          f"value {1e3*tv/chunk:7.2f} ms/key", flush=True)
