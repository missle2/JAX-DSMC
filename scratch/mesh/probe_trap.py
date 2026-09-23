"""Where do particles exhaust the 4-event budget on the optc shapes?  Single moves (dt) of
freestream-Maxwellian particles started within 0.3 of the surface, max_bounces 4; failing
batches bisected down to the particle, whose track is then replayed with 16 bounces."""
import dataclasses, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
from examples.mesh import forward as F
from jax_dsmc import wall_adjoint
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.mesh import levelset as L

cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
R = "examples/mesh/results/optimize"
cases = [a.split(":") for a in sys.argv[1:]] or [["circle_coarse_S1_gd", "0"], ["circle_coarse_S1_gd", "8"]]
for tag, k in cases:
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{R}/{tag}/coef_{k}.npy")))
    grid, mesh, sp, params, static, faces, res, wall, cap = EM.build(cfg, "radius", None, 10, ls)
    mass, dt = sp.mass[0], params.dt
    sig = float(np.sqrt(K_BOLTZ * F.TEMP / float(mass)))
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)

    def fails(x, v, mb=4):
        out = wall_adjoint.advance(x, v, jnp.ones(x.shape[0], bool), grid, faces, wall, mass, dt,
                                   jax.random.key(0), mesh=mesh, max_bounces=mb)
        return int(out[3].exhausted), int(out[3].inside_body)

    rng = np.random.default_rng(0)
    B, total, bad, inside = 65536, 0, [], 0
    for it in range(60):
        x = rng.uniform(lo, hi, (4 * B, 2))
        phi = np.asarray(L.value(ls, jnp.asarray(x)))
        x = x[(phi > 0) & (phi < 0.3)][:B]
        v = rng.normal(0, sig, (x.shape[0], 3)); v[:, 0] += F.VSTREAM; v[:, 2] = 0
        # sample the fast tail harder: a third of the batch at 3x the speed
        v[: x.shape[0] // 3, :2] *= 3.0
        x, v = jnp.asarray(x), jnp.asarray(v)
        total += x.shape[0]
        ex, ins = fails(x, v); inside += ins
        if ex:
            idx = np.arange(x.shape[0])
            while idx.size > 1:
                half = idx[: idx.size // 2]
                idx = half if fails(x[half], v[half])[0] else idx[idx.size // 2 :]
            bad.append((np.asarray(x[idx[0]]), np.asarray(v[idx[0]])))
    print(f"{tag} coef_{k}: {total} moves, exhausted {len(bad)} (rate {len(bad)/total:.2e}), inside_body {inside}")
    for xb, vb in bad[:6]:
        ex16 = fails(jnp.asarray(xb[None]), jnp.asarray(vb[None]), 16)
        out = wall_adjoint.advance(jnp.asarray(xb[None]), jnp.asarray(vb[None]), jnp.ones(1, bool), grid, faces, wall,
                                   mass, dt, jax.random.key(0), mesh=mesh, max_bounces=16, return_events=True)
        ev = out[-1]
        faces_hit = np.asarray(ev.face)[:, 0]
        print(f"   x {xb.round(3)} |v| {np.linalg.norm(vb[:2]):.0f}  hits(16 passes) {faces_hit[faces_hit >= 0].tolist()}  exhausted@16 {ex16[0]}")
