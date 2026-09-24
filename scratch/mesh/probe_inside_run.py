"""Reproduce `inside_body` in the real optc forward: S4 gd coef_6, seeds 32..35.  Find the
captured step where it fires, re-run to the step before it, and bisect the pre-move state
down to the particle (the body and the box faces are specular / outflow: no draws, so a
subset tracks exactly like the full set)."""
import dataclasses, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
from examples.mesh import forward as F
from jax_dsmc import wall_adjoint, initialize
from jax_dsmc.rng import STAGE_MOVE, key_for
from jax_dsmc.state import FlowParams
from jax_dsmc.mesh import levelset as L

tag, k = sys.argv[1].split(":")
seeds = [int(s) for s in sys.argv[2:]] or [32, 33, 34, 35]
cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"examples/mesh/results/optimize/{tag}/coef_{k}.npy")))
for seed in seeds:
    key = jax.random.key(seed)
    s0, final, diag, _, _, s = EM.forward(cfg, "radius", None, key, False, body=ls)
    ins = np.asarray(diag.inside_body)
    print(f"seed {seed}: inside_body per captured step nonzero at {np.nonzero(ins)[0].tolist()} "
          f"(total {int(ins.sum())}), stuck {int(diag.surface_stuck.sum())}", flush=True)
    if not ins.any():
        continue
    j = int(np.nonzero(ins)[0][0])
    step = cfg["warmup"] + j
    # the pre-move state of that step: warm-up + j captured steps from the empty start
    grid, mesh, sp, params, static = s["grid"], s["mesh"], s["sp"], s["params"], s["static"]
    faces, wall = s["faces"], s["wall"]
    flow = FlowParams(nrho=jnp.float64(F.NRHO), temp=jnp.float64(F.TEMP), vstream=jnp.array([F.VSTREAM, 0.0, 0.0]))
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    _grid, _mesh, _sp, _params, _static, _faces, res, _wall, capacity = EM.build(cfg, "radius", None, cfg["steps"], ls)
    pre = dataclasses.replace(static, n_steps=step)
    (state, cstate), _ = EM._run(state, cstate, grid, sp, params, key, faces, wall, res, mesh, pre, capacity, 0, False)
    x, v, alive = state.x, state.v, state.alive
    kmove = key_for(key, step, STAGE_MOVE)

    def run(idx):
        return wall_adjoint.advance(x[idx], v[idx], alive[idx], grid, faces, wall, sp.mass[0], params.dt,
                                    kmove, mesh=mesh, max_bounces=static.K_bounce, return_events=True)

    idx = np.arange(x.shape[0])
    print("   full-state move inside_body:", int(run(idx)[3].inside_body), flush=True)
    while idx.size > 1:
        half = idx[: idx.size // 2]
        idx = half if int(run(half)[3].inside_body) else idx[idx.size // 2:]
    i = int(idx[0])
    xb, vb = np.asarray(x[i]), np.asarray(v[i])
    out = run(idx)
    ev = np.asarray(out[5].face)[:, 0]
    phi0 = float(L.value(ls, jnp.asarray(xb[None]))[0])
    _, g = L.value_and_grad(ls, jnp.asarray(xb[None]))
    sp_ = np.linalg.norm(vb[:2]); d = vb[:2] / sp_
    s_ = np.linspace(0, sp_ * float(params.dt), 20001)
    f = np.asarray(L.value(ls, jnp.asarray(xb[None] + s_[:, None] * d[None])))
    cross = np.nonzero(np.diff(np.sign(f)))[0]
    print(f"   particle {i}: x {xb.round(6)} v {vb.round(2)} |v|dt {s_[-1]:.5f}  phi(x0) {phi0:.3e}  "
          f"v.grad phi {float(jnp.dot(jnp.asarray(vb[:2]), g[0])):.3e}  events {ev[ev >= 0].tolist()}  "
          f"x_end {np.asarray(out[0][0]).round(6)}  sign changes along the ray at s={[round(float(s_[c]), 6) for c in cross]}", flush=True)
    np.savez(f"scratch/mesh/inside_particle_{tag}_{k}_{seed}.npz", x=xb, v=vb, step=step)
    break
