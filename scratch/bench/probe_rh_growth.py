"""Growth of |lambda| over the 300-step RH backward vs partners K (device virtual, staged)."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from examples import ensemble_inlet_shock as eis
from jax_dsmc import ensemble_device as device

geometry = eis.inlet_shock_geometry(3200, 80, 0.2)
fwd = eis.compiled_forward(geometry, 80, 192, 300, "certified", True)
t = eis.controls("mach", jnp.float64(1.0))
state0, out, diag, rec = fwd(t, jax.random.key(32))
length = geometry["length"]
totals = eis._window_counts(out.x, out.alive, length)
phi = eis.objective_terminal("density-ratio", out.x, out.v, length, 3200, totals)
dx = (np.asarray(geometry["grid"].hi) - np.asarray(geometry["grid"].lo)) / np.asarray(geometry["grid"].n)
v_scale = float(np.sqrt(eis.K_BOLTZ * eis.GAS.tref / eis.GAS.mass))
scale = jnp.concatenate((jnp.asarray(dx), jnp.full(3, v_scale)))
sp = eis._species_of(t)
for K in [int(k) for k in sys.argv[1:]]:
    t0 = time.time()
    dev = device.backward_virtual(rec, state0.alive, out, phi, geometry["grid"], scale, sp, jax.random.key(1), 640, 32, "quadratic_full", 4096, 2, K, "staged", None)
    lam = np.asarray(dev.lam)  # (M, N)
    alive = np.asarray(rec.alive)
    mx = np.array([np.abs(lam[k][alive[k]]).max() for k in range(lam.shape[0])])
    rms = np.array([np.sqrt(np.mean(lam[k][alive[k]] ** 2)) for k in range(lam.shape[0])])
    print(f"K={K}: {time.time()-t0:.0f}s  max|lam| at steps 299,250,200,100,0: {mx[[299,250,200,100,0]]}  rms: {rms[[299,250,200,100,0]]}", flush=True)
