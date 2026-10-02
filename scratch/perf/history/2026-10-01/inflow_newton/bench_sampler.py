"""Microbenchmark of the inflow velocity sampler (run with PYTHONPATH pointing at the tree to time)."""
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax_dsmc import inflow
from jax_dsmc.constants import K_BOLTZ

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
mass = 6.63e-26; temp = 273.0
stream = jnp.array([400.0, 0.0, 0.0]); normal = jnp.array([1.0, 0.0, 0.0])

@jax.jit
def sampler(key):
    v, ok = inflow.sample_surface_velocities(key, n, temp, stream, normal, mass)
    return v, ok

@jax.jit
def quantile(u, s):
    return inflow.flux_quantile(u, s)

key = jax.random.key(0)
t0 = time.perf_counter(); jax.block_until_ready(sampler(key)); tc = time.perf_counter() - t0
u = jax.random.uniform(key, (n,), dtype=jnp.float64); s = jnp.float64(0.7)
jax.block_until_ready(quantile(u, s))
keys = jax.random.split(key, reps)
for fn, name in ((lambda k: sampler(k), "sampler"), (lambda k: quantile(u, s), "quantile")):
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for k in keys:
            out = fn(k)
        jax.block_until_ready(out)
        best.append((time.perf_counter() - t0) / reps)
    print(f"{jax.devices()[0].platform} n={n} {name}: {min(best)*1e3:.4f} ms/call (min of 3 x {reps})")
print(f"sampler first-call (compile) {tc:.2f} s")
v, ok = sampler(key); print("accepted", int(ok.sum()), "/", n)
