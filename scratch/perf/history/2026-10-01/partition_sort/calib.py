"""Calibrate a collisional periodic box for the partition-sort statistics test."""
import dataclasses, sys, time
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC/tests")
import jax, jax.numpy as jnp, numpy as np
import test_ntc as T
from jax_dsmc import simulate, sort
from jax_dsmc.state import ParticleState

def case(seed, sort_mode, dt, fnum, steps):
    state, cs, g, sp, params, static, _t, k = T.build(n=(4, 4), N=4000, dt=dt, seed=seed, A_max=200)
    v = state.v.at[:, 0].multiply(1.6)
    state = dataclasses.replace(state, v=v) if dataclasses.is_dataclass(state) else state._replace(v=v)
    params = dataclasses.replace(params, fnum=jnp.full_like(params.fnum, fnum))
    static = dataclasses.replace(static, n_steps=steps, sort=sort_mode)
    (st, _c), (m, diag) = simulate.run(state, cs, g, sp, params, static, k, emit_moments="none")
    v = np.asarray(st.v)
    x0 = np.asarray(state.x)
    moved = 0
    return (v[:, 0].var() - v[:, 1].var()) / v.var(axis=0).mean(), int(np.asarray(diag.max_attempts).max()), diag

for dt, fnum in [(2e-9, 3e10), (2e-9, 1e11), (2e-9, 3e11)]:
    t = time.time()
    a, n, diag = case(0, "full", dt, fnum, 40)
    b, n2, _ = case(0, "partition", dt, fnum, 40)
    print(dt, fnum, "aniso full", a, "part", b, "accepted", n, n2, f"{time.time()-t:.1f}s", flush=True)
print([f for f in dir(diag) if not f.startswith("_")])
