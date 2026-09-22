"""Decompose the silhouette term on the GP plate: per window step, the band counts on the
hit / miss side, the mean virtual reward, the mean fitted value of the virtual state, the
mean own value, and the resulting contribution to dJ/dw -- with the real adjoint and with
lam = 0 (pure hit-count x reward).  One seed, a few steps.

    python3 scratch/mesh/probe_silhouette.py --steps 20
"""

import argparse
import functools
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from examples import ensemble_mesh as EM  # noqa: E402
from examples.ensemble_inlet_shock import device_backward  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--band", type=float, default=0.05)
    args = ap.parse_args()
    ensemble_backward.configure("virtual", 1, 2, "staged")
    cfg = EM.default_config()
    cfg.update({"case": "plate", "body": "levelset", "acc": 1.0, "silhouette_band": args.band})
    key = jax.random.key(32)
    state0, final, diag, rec, tallies, s = EM.forward(cfg, "width", EM.nominal(cfg, "width"), key, True)
    static, grid, mesh, wall, sp, params = s["static"], s["grid"], s["mesh"], s["wall"], s["sp"], s["params"]
    dim, m, w = static.D, cfg["steps"], cfg["window"]
    mass, dt = sp.mass[0], params.dt
    scale_r = float(params.fnum[0]) / (w * float(dt))
    first = cfg["warmup"] + m - w
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    reward = functools.partial(EM._drag_reward, mass=mass, dim=dim, first=first, scale=scale_r)
    rep = shape.replay(rec, steps, state0, key, grid, s["faces"], wall, mass, dt, mesh, 0.0, 0.0, static, reward=reward)
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    v_scale = float(np.sqrt(K_BOLTZ * 273.15 / float(mass)))
    fit_scale = jnp.concatenate((jnp.asarray(dx), jnp.full(3, v_scale)))
    npc_max = max(64, 4 * int(np.asarray(rec.alive).sum(axis=1).max()) // num_cells(grid), int(ensemble_backward.max_occupancy(rec, grid)) + 8)
    _, dev = device_backward(rec, state0, final, jnp.zeros(static.N), 1.0, grid, v_scale, sp, jax.random.fold_in(key, 12345), 32, npc_max=npc_max, hit_reward=rep.hit_reward, return_device=True)
    setup = shape._setup(key, grid, s["faces"], wall, mass, dt, mesh, 0.0, 0.0, static)["map"]
    fit = {"npc_max": npc_max, "scale": fit_scale, "k": 96, "basis": "quadratic_full", "chunk": 4096, "pool": 1}
    geom = shape.geometry(mesh)
    _, vjp_w = jax.vjp(lambda th: EM.levelset_coef(th, cfg, "width", grid)[0], jnp.float64(EM.nominal(cfg, "width")))
    tot = {"real": 0.0, "lam0": 0.0}
    for k in range(m - args.steps, m):
        x = rec.x[k - 1]
        v = rec.v_post[k - 1]
        alive = rec.alive[k - 1]
        kk = key_for(key, cfg["warmup"] + k, STAGE_MOVE)
        x1, v1, _a, _lp, _tal, events = shape.wall_map(geom, x, v, alive, kk, **setup)
        rec_k = jax.tree.map(lambda a, k=k: a[k], rec)
        survivors = alive & rec_k.alive & (rec_k.born_face < 0)
        r_k = jnp.where(survivors[None, :], reward(events, None, rec_k, cfg["warmup"] + k), 0.0)
        for name, lam_k in (("real", dev.lam[k]), ("lam0", jnp.zeros_like(dev.lam[k]))):
            to_go = shape._to_go(jnp.where(survivors, lam_k, 0.0), r_k)
            c, xs = shape.silhouette_cotangent(mesh.body, x, v, survivors, events, to_go, rec_k, lam_k, cfg["warmup"] + k, jax.random.fold_in(kk, 7919), setup, fit, args.band, reward)
            _, vjp = jax.vjp(functools.partial(shape._phi_at, ls=mesh.body, x_star=xs), mesh.body.coef)
            (g,) = vjp(c)
            d_dw = float(vjp_w(g)[0])
            tot[name] += d_dw
            cn = np.asarray(c)
            hit0 = np.asarray(events.face[0] >= 2 * dim)
            band = cn != 0
            print(f"step {k} [{name}]: d/dw {d_dw:+.3e}  band hit-side {int((band & hit0).sum())} miss-side {int((band & ~hit0).sum())}  c: hit-side mean {cn[band & hit0].mean() if (band & hit0).any() else 0:+.3e} miss-side mean {cn[band & ~hit0].mean() if (band & ~hit0).any() else 0:+.3e}", flush=True)
    print("totals over", args.steps, "steps:", {k: f"{v:.4e}" for k, v in tot.items()}, " (x W/steps for the window:", {k: f"{v * w / args.steps:.4e}" for k, v in tot.items()}, ")")
    print("reference: FD ~1.2e-20, J =", float(jnp.sum(rep.hit_reward)))


if __name__ == "__main__":
    main()
