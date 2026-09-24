"""Trace the inside_body particle pass by pass (hit time, point, phi there, normal,
v_in . n, v_out . n) and the level set around it."""
import dataclasses, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
from jax_dsmc import wall_adjoint
from jax_dsmc.mesh import levelset as L
from jax_dsmc.rng import STAGE_MOVE, key_for

tag, k, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"examples/mesh/results/optimize/{tag}/coef_{k}.npy")))
grid, mesh, sp, params, static, faces, res, wall, cap = EM.build(cfg, "radius", None, 10, ls)
z = np.load(f"scratch/mesh/inside_particle_{tag}_{k}_{seed}.npz")
x, v = jnp.asarray(z["x"][None]), jnp.asarray(z["v"][None])
for kb in (4, 16):
    out = wall_adjoint.advance(x, v, jnp.ones(1, bool), grid, faces, wall, sp.mass[0], params.dt,
                               key_for(jax.random.key(seed), int(z["step"]), STAGE_MOVE), mesh=mesh,
                               max_bounces=kb, return_events=True, trace=True)
    ev = out[5]
    print(f"K_bounce {kb}: exhausted {int(out[3].exhausted)} inside {int(out[3].inside_body)} x_end {np.asarray(out[0][0]).round(7)} alive {bool(out[2][0])}")
    for p in range(kb):
        f = int(ev.face[p, 0])
        if f < 0 and float(ev.time[p, 0]) == 0.0:
            continue
        xh = ev.x_hit[p, 0]; n = np.asarray(ev.normal[p, 0]); vin = np.asarray(ev.v_in[p, 0]); vout = np.asarray(ev.v_out[p, 0])
        print(f"  pass {p}: face {f} t {float(ev.time[p,0]):.6e}  x_hit {np.asarray(xh).round(7)} phi {float(L.value(ls, xh[None])[0]):+.3e}  "
              f"v_in.n {vin @ n:+.2f}  v_out.n {vout @ n:+.2f}")
# the local geometry: phi and curvature near the start point
xs = np.asarray(z["x"])
g = np.linspace(-0.08, 0.08, 9)
print("phi on a 9x9 patch (rows y, cols x) around the start, x0 =", xs.round(4))
for dy in g[::-1]:
    row = [float(L.value(ls, jnp.asarray([[xs[0] + dx, xs[1] + dy]]))[0]) for dx in g]
    print("  " + " ".join(f"{r:+.3f}" for r in row))
