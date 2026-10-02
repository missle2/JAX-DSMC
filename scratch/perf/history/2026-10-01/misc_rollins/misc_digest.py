"""Bitwise digest for the non-tracker speed-ups (run with PYTHONPATH=<tree>, cwd=<tree>):
  forward_<body>: 300 SPARTA-circle steps (scratch/perf/forward_digest.py's run)
  capture: 60 polygon steps with capture=True (lean records, every leaf)
  dense_*: a collisional case (test_ntc.build, 4x4 cells, ~many attempts): one collide with
           capture in paper and trajectory modes, its reverse-mode gradients, and a
           12-step periodic simulate.run.
usage: misc_digest.py --save TAG | --compare TAG [--parts ...]"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

TREE = os.getcwd()
sys.path.insert(0, os.path.join(TREE, "scratch", "perf"))
sys.path.insert(0, os.path.join(TREE, "examples", "mesh"))
sys.path.insert(0, os.path.join(TREE, "tests"))
DIR = "/home/aakash/jax_dsmc/wt_misc/scratch/misc/digest"


def leaves(prefix, tree):
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        out[prefix + jax.tree_util.keystr(path)] = np.asarray(leaf)
    return out


def forward(body):
    import forward_digest as FD
    return {f"{body}/{k}": v for k, v in FD.run(body).items()}


def capture():
    import bench_forward as B
    from examples import ensemble_mesh as EM
    from jax_dsmc import initialize
    from jax_dsmc.state import FlowParams
    grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup("polygon")
    static = dataclasses.replace(static, n_steps=60, record="lean")
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
    key = jax.random.key(11)
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    (state, cstate), aux = EM._run(state, cstate, grid, sp, params, key, faces, wall, res, mesh,
                                   static, capacity, 0, True)
    return leaves("capture", ((state, cstate), aux))


def dense():
    import test_ntc as T
    from jax_dsmc import simulate
    from jax_dsmc.collide.ntc import collide, post_collision_velocities, step_pairs
    from jax_dsmc.sort import build_table
    out = {}
    state, cs, g, sp, params, static, temp, k = T.build(n=(4, 4), N=4000, A_max=64, dt=1.2e-9)
    table = build_table(state, g)
    out.update(leaves("dense/table", table))
    for mode in ("paper", "trajectory"):
        res = collide(state, table, cs, g, sp, params, static, temp, 3, k, score_mode=mode, capture=True)
        out.update(leaves(f"dense/{mode}", res))
        pairs = res[3]
        out[f"dense/{mode}/replay_v"] = np.asarray(post_collision_velocities(state.v, pairs))

        def obj(v, dt, mode=mode):
            s = dataclasses.replace(state, v=v)
            p = dataclasses.replace(params, dt=dt)
            new = collide(s, table, cs, g, sp, p, static, temp, 3, k, score_mode=mode)[0]
            return jnp.sum(new.v[:, 0] ** 2) * 1e-6 + jnp.sum(new.logh_acc)

        gv, gdt = jax.grad(obj, argnums=(0, 1))(state.v, params.dt)
        out[f"dense/{mode}/grad_v"] = np.asarray(gv)
        out[f"dense/{mode}/grad_dt"] = np.asarray(gdt)
    print("dense attempts", int(jnp.sum(res[3].pi < static.N)), "accepted", int(jnp.sum(res[3].accept)), "max", int(res[2].max_attempts), "min", int(jnp.min(jnp.sum(res[3].pi < static.N, axis=0))))
    st = dataclasses.replace(static, n_steps=12)
    run = simulate.run(state, cs, g, sp, params, st, k, emit_moments=False)
    out.update(leaves("dense/run", run))
    return out


PARTS = {"polygon": lambda: forward("polygon"), "levelset": lambda: forward("levelset"),
         "capture": capture, "dense": dense}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--parts", nargs="*", default=["dense", "capture", "polygon", "levelset"])
    args = ap.parse_args()
    import jax_dsmc
    print("jax_dsmc from", jax_dsmc.__file__, flush=True)
    ndiff = 0
    for part in args.parts:
        got = PARTS[part]()
        if args.save:
            np.savez(os.path.join(DIR, f"{args.save}_{part}.npz"), **got)
            print("saved", part, len(got), flush=True)
        if args.compare:
            ref = np.load(os.path.join(DIR, f"{args.compare}_{part}.npz"))
            missing = sorted(set(ref.files) ^ set(got))
            if missing:
                print(f"  {part}: key mismatch {missing[:8]}")
            for name in sorted(set(ref.files) & set(got)):
                a, b = got[name], ref[name]
                if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                    ndiff += 1
                    if a.shape == b.shape:
                        d = np.abs(a.astype(float) - b.astype(float))
                        rel = np.nanmax(d) / max(np.nanmax(np.abs(b.astype(float))), 1e-300)
                        print(f"  {part} {name}: DIFFERS max|d| {np.nanmax(d):.3e} rel {rel:.2e} n {int(np.sum(d > 0))}")
                    else:
                        print(f"  {part} {name}: SHAPE {a.shape} vs {b.shape}")
            print(f"{part}: {len(ref.files)} arrays compared", flush=True)
    if args.compare:
        print("BITWISE IDENTICAL" if ndiff == 0 else f"DIFFERENCES: {ndiff}")


if __name__ == "__main__":
    main()
