"""Where do single moves end inside the body on the failing optc shapes?  Freestream-
Maxwellian particles started within 0.3 of the surface (a third at 3x speed), one move of
dt with the stencil tracker (K_bounce 4, as the runs); failing batches bisected down to
the particle, whose ray is then examined against the level set."""
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
for arg in sys.argv[1:]:
    tag, k = arg.split(":")
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{R}/{tag}/coef_{k}.npy")))
    grid, mesh, sp, params, static, faces, res, wall, cap = EM.build(cfg, "radius", None, 10, ls)
    mass, dt = sp.mass[0], params.dt
    sig = float(np.sqrt(K_BOLTZ * F.TEMP / float(mass)))
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)

    def run(x, v):
        return wall_adjoint.advance(x, v, jnp.ones(x.shape[0], bool), grid, faces, wall, mass, dt,
                                    jax.random.key(0), mesh=mesh, max_bounces=static.K_bounce,
                                    return_events=True)

    rng = np.random.default_rng(1)
    total, bad = 0, []
    for it in range(40):
        x = rng.uniform(lo, hi, (4 * 65536, 2))
        phi = np.asarray(L.value(ls, jnp.asarray(x)))
        x = x[(phi > 0) & (phi < 0.3)][:65536]
        v = rng.normal(0, sig, (x.shape[0], 3)); v[:, 0] += F.VSTREAM; v[:, 2] = 0
        v[: x.shape[0] // 3, :2] *= 3.0
        x, v = jnp.asarray(x), jnp.asarray(v)
        total += x.shape[0]
        if int(run(x, v)[3].inside_body):
            idx = np.arange(x.shape[0])
            while idx.size > 1:
                half = idx[: idx.size // 2]
                idx = half if int(run(x[half], v[half])[3].inside_body) else idx[idx.size // 2:]
            bad.append((np.asarray(x[idx[0]]), np.asarray(v[idx[0]])))
    print(f"{tag} coef_{k}: {total} moves, inside_body {len(bad)} ({len(bad)/total:.1e})", flush=True)
    for xb, vb in bad[:5]:
        out = run(jnp.asarray(xb[None]), jnp.asarray(vb[None]))
        faces_hit = np.asarray(out[5].face)[:, 0]
        # the undeflected ray: phi sampled finely along the move
        s = np.linspace(0, np.linalg.norm(vb[:2]) * float(dt), 4001)
        d = vb[:2] / np.linalg.norm(vb[:2])
        f = np.asarray(L.value(ls, jnp.asarray(xb[None] + s[:, None] * d[None])))
        cross = np.nonzero(np.diff(np.sign(f)))[0]
        dips = [(round(float(s[c]), 4)) for c in cross]
        seg_len = np.diff([s[c] for c in cross]) if len(cross) > 1 else []
        print(f"   x {xb.round(4)} v {vb[:2].round(0)} |v|dt {s[-1]:.4f}  events {faces_hit[faces_hit >= 0].tolist()}  "
              f"sign changes at s={dips} inside-chords {np.round(seg_len, 4).tolist()}  min phi {f.min():.4f}", flush=True)
