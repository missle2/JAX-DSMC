"""The particle of the 2026-10-21 inside_body event (scratch/insidebody/repro.py found it at
captured step STEP of seed SEED, iterate IT): the run to the start of that step, the step, the
slot that died with phi < 0, its start / end, velocity, the hit search on its move and phi
sampled along it.

    PYTHONPATH=. python3 scratch/insidebody/culprit.py RUN_DIR IT SEED STEP
"""

import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import boundary, initialize  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402


def main():
    run, it, seed, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    with open(f"{run}/history.json") as fh:
        cfg = dict(O.default_config(), **json.load(fh)["config"])
    _grid, ls = O.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    key = jax.random.key(cfg["seed_start"] + seed)
    state, cstate = O.load_warm(O._warm_path(run, seed))
    grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
        cfg, "radius", EM.nominal(cfg, "radius"), cfg["steps"], ls
    )
    flow = FlowParams(nrho=jnp.float64(EM.F.NRHO * EM.knudsen_density(cfg)),
                      temp=jnp.float64(EM.F.TEMP),
                      vstream=jnp.array([EM.stream_speed(cfg), 0.0, 0.0]))  # fmt: skip
    fresh = (initialize.initial_state(key, flow, grid, sp, static, n_alive=0),
             initialize.initial_collide_state(flow, grid, sp, static))  # fmt: skip
    state, cstate = EM._carried(state, cstate, mesh, fresh, grid)
    common = (grid, sp, params, key, faces, wall, res, mesh)
    settle = int(cfg["restart_steps"])
    w = cfg["warmup"]
    (state, cstate), _ = EM._run(state, cstate, *common,
                                 dataclasses.replace(static, n_steps=settle), capacity,
                                 w - settle, False)  # fmt: skip
    cache = f"scratch/insidebody/state_{seed}_{step}.npz"
    try:
        z = np.load(cache)
        x0, v0, a0 = (jnp.asarray(z[k]) for k in ("x", "v", "alive"))
    except FileNotFoundError:
        (s0, _c0), _ = EM._run(state, cstate, *common,
                               dataclasses.replace(static, n_steps=step), capacity, w,
                               False)  # fmt: skip
        x0, v0, a0 = s0.x, s0.v, s0.alive
        np.savez(cache, x=np.asarray(x0), v=np.asarray(v0), alive=np.asarray(a0))
    # the step's move, as simulate._step calls it (before the inflow and the collisions)
    x1, v1, a1, counts, _logh, *_ = boundary.apply_faces(
        x0, v0, a0, grid, faces, wall, sp.mass[0], params.dt,
        key_for(key, w + step, STAGE_MOVE), epsilon=params.epsilon, max_bounces=static.K_bounce,
        mesh=mesh, tracker=static.tracker, max_walk=static.K_walk,
        lane_size=static.lane_size,
    )  # fmt: skip
    print("move counts:", counts, flush=True)
    x0n, v0n, x1n, v1n = (np.asarray(a) for a in (x0, v0, x1, v1))
    phi1 = np.asarray(L.value(ls, x1))
    cand = np.nonzero(np.asarray(a0) & ~np.asarray(a1) & (phi1 < 0.0))[0]
    print("killed inside:", cand.tolist())
    dt = float(params.dt)
    for i in cand:
        print(f"slot {i}: start {x0n[i]}, end {x1n[i]}, v start {v0n[i]}, v end {v1n[i]}")
        print(f"  phi start {float(L.value(ls, x0[i:i + 1])[0]):.3e}  phi end {phi1[i]:.3e}  "
              f"|v|dt {np.linalg.norm(v0n[i][:2]) * dt:.4f}  h {ls.h}  reach {ls.reach}")
        t = L.hit_body(x0[i:i + 1], v0[i:i + 1], ls, jnp.asarray([dt]))
        print(f"  hit_body(start, v start, dt): t = {float(t[0]):.4e} (dt {dt:.4e})")
        sgrid = np.linspace(0.0, 1.0, 401)[:, None]
        seg = x0n[i][None] + sgrid * v0n[i][None, :2] * dt
        ph = np.asarray(L.value(ls, jnp.asarray(seg)))
        neg = np.nonzero(ph < 0)[0]
        print(f"  phi along the straight dt move: min {ph.min():.3e} at s={sgrid[ph.argmin(), 0]:.4f}"
              f"; negative for s in {[float(sgrid[neg[0], 0]), float(sgrid[neg[-1], 0])] if len(neg) else 'none'}")
        if not np.allclose(v0n[i], v1n[i]):
            print("  the velocity changed in the move: a reflection happened")
    np.savez(f"scratch/insidebody/culprit_{seed}_{step}.npz", slots=cand, x0=x0n[cand],
             v0=v0n[cand], x1=x1n[cand], v1=v1n[cand])  # fmt: skip


if __name__ == "__main__":
    main()
