"""Where the backward's guard cost goes (local GPU, reduced size as bench.py): the value-form
backward (EM._backwards) alone, repeated on one captured record, 2026-10-22:

* ``shipped``      -- CELL_BASIS (guard32);
* ``no_leverage``  -- the guard's per-query leverage replaced by the fit's top order (what an
  unguarded query reads; times the leverage evaluation);
* ``unguarded``    -- the basis without the guard (no whitening, LU solve).

    PYTHONPATH=. python3 scratch/shapefast/backward.py [overrides-json]
"""

import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import cell_fit, ensemble_backward, ensemble_device, shape  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402

SHIPPED_LEVEL = cell_fit._guard_level


def top_level(field, c, f, guard, q_valid, sizes, layout=None):
    del f, guard, q_valid, sizes, layout
    return field.top[c]


def setup(cfg, ls, key, full=False, static_over=None):
    value0 = EM.nominal(cfg, "radius")
    state0, final, diag, rec, _tallies, s = EM.forward(cfg, "radius", value0, key, True, ls)
    grid, mesh, wall, sp, params, static = (s[k] for k in ("grid", "mesh", "wall", "sp",
                                                            "params", "static"))  # fmt: skip
    if static_over:
        static = dataclasses.replace(static, **static_over)
    dim, m, w = static.D, cfg["steps"], cfg["window"]
    mass, dt = sp.mass[0], params.dt
    scale_r = float(params.fnum[0]) / (w * float(dt))
    first = cfg["warmup"] + m - w
    steps = cfg["warmup"] + jnp.arange(m, dtype=jnp.int32)
    reward, jump, band = EM._drag_setup(cfg, float(mass), dim, first, scale_r, cfg["timing_band"])
    rep, static = EM._checked_replay(rec, steps, state0, key, grid, s["faces"], wall, mass, dt, mesh,
                             static, reward, jump, band)  # fmt: skip
    v_scale = float(np.sqrt(K_BOLTZ * EM.F.TEMP / float(mass)))
    npc_max = int(ensemble_backward.max_occupancy(rec, grid)) + 8
    npc_max = -(-npc_max // EM.NPC_BUCKET) * EM.NPC_BUCKET
    print(
        "K_bounce",
        static.K_bounce,
        "lane",
        static.lane_size,
        "max hits/step",
        int(jnp.max(diag.surface_hits)) if diag.surface_hits.ndim else int(diag.surface_hits),
    )
    print("slots", rec.x.shape[1], "alive mean", float(rec.alive.sum(1).mean()), "cells",
          int(grid.volume.shape[0]), "npc_max", npc_max, flush=True)  # fmt: skip
    args = (rec, state0, final, grid, v_scale, sp, key, cfg["neighbors"], npc_max,
            rep.hit_reward, rep.timing_total[:, None])  # fmt: skip
    if not full:
        return args
    fit_scale = jnp.concatenate((jnp.asarray((np.asarray(grid.hi) - np.asarray(grid.lo))
                                             / np.asarray(grid.n)), jnp.full(3, v_scale)))  # fmt: skip
    tally_cot = EM._drag_tally_cotangent(_tallies, dim, mesh, scale_r)

    def shape_kwargs(dev, dev_t):
        del dev_t
        return dict(records=rec, lam=dev.lam, steps=steps, state0=state0, base_key=key,
                    grid=grid, faces=s["faces"], wall=wall, mass=mass, dt=dt, mesh=mesh,
                    epsilon=0.0, acc_delta=0.0, static=static, scale=fit_scale,
                    npc_max=npc_max, hit_reward=rep.hit_reward, tally_cotangent=tally_cot,
                    tally_weight=(steps >= first).astype(jnp.float64), timing_jump=jump,
                    timing_band=band, silhouette_band=0.0, reward=None, value_class="none",
                    hit_fits=True)  # fmt: skip

    return args, {"shape_kwargs": shape_kwargs, "geom": shape.geometry(mesh)}


def main():
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cfg = dict(OPT.default_config(), case="circle_long", fnum_scale=16.0, levelset_h=0.0625,
               levelset_samples=16, grazing=0.05, boundary="free", knudsen=0.5, mach=10.0,
               dt=2.5e-5, a_max=256, warmup=2800, steps=80, window=50, smooth=8.0,
               variant="along_pool_plane", neighbors=32, timing_band=10.0)  # fmt: skip
    cfg.update(over)
    _grid, ls = OPT.initial_levelset(cfg)
    key = jax.random.key(32)
    ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
    args = setup(cfg, ls, key)
    ref = None
    for name, basis, level in (
        ("shipped", ensemble_backward.CELL_BASIS, SHIPPED_LEVEL),
        ("no_leverage", ensemble_backward.CELL_BASIS, top_level),
        ("unguarded", "cell:cubicv:adapt1.5:align0.9:both", SHIPPED_LEVEL),
    ):
        ensemble_backward.configure("virtual", 4, 8, "staged", basis)
        cell_fit._guard_level = level
        ensemble_device.backward_virtual.clear_cache()
        for label in ("compile", "timed", "timed"):
            t0 = time.time()
            dev, _dev_t = EM._backwards(*args)
            lam = np.asarray(dev.lam)
            print(f"{name:12s} {label:8s} {time.time() - t0:7.2f} s", flush=True)
        if ref is None:
            ref = lam
        print(f"  rel diff of lam vs shipped {np.linalg.norm(lam - ref) / np.linalg.norm(ref):.3g}",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
