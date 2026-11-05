"""The fit-free truth of pathwise_x (2026-11-05): a forward difference of the objective under
a shift of the post-hit positions only.

pathwise_x is the objective's response to the body hits' post-move positions, through
``d x_post / d theta``.  For a hit on the circle that tangent is, to < 1 % on the pathwise sums
(checked on the fit lab's dumps), ``w(theta) (v_out - v_in) / |v_in . n|`` with ``w`` the
control's outward normal displacement per unit (1 for the radius, the bumps' cos^4 profiles).
Here every captured step's body hits (alive before and after the move, velocity changed)
get ``x_post += eps * w * dir`` with ``dir`` that vector (``full``: the pathwise term
itself) or the normal ``n`` (``normal``: its wall-normal part only).  The wall, the hit
tallies and the re-emitted velocities are untouched, so ``(J(eps) - J(0)) / eps -> the true
pathwise_x`` -- no fit -- and the fits' predictions are ``sum_hits grad_x lambda . (w dir)``
on the same seeds' dumps.  ``w >= 0`` and ``dir . n >= 1``: the shift is outward, never
into the body, so the difference is one-sided; the eps ladder shows its linearity and noise.

The warm-up runs once per seed; each (control, mode, eps) is a captured-window run from the
same state with the same keys (common random numbers).  Drag and heat from the same runs
(the two gates' forwards are identical: same key, only ``objective`` / ``heat_patch``
differ).  J(0) reproduces the bump gate's J.  Rows to
``scratch/fitlab/cluster/shift_s<a>-<b>.jsonl``.

    PYTHONPATH=. python3 scratch/fitlab/shiftfd.py --seeds 0 4 [--smoke]
"""

import argparse
import dataclasses
import functools
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import boundary, initialize, simulate  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

OUT = Path("scratch/fitlab/cluster")
GATES = {"drag": "c10_diff_m2", "heat": "c10_diff_heat_m2"}
# (control, mode): the radius and the nose / side bumps (w >= 0: outward shifts); the nose
# also with the normal part alone (the along-flight identity's share is the rest)
CASES = (("radius", "full"), ("bump_nose", "full"), ("bump_nose", "normal"),
         ("bump_side", "full"))  # fmt: skip
EPS = (1e-4, 1e-3, 3e-3, 1e-2)
WEIGHT_INDEX = {"radius": 1, "bump_nose": 2, "bump_side": 3, "bump_base": 4}

_ORIGINAL = boundary.apply_faces
_SHIFT = {"p": None, "centre": None}
"""The traced shift parameters ``f64[6] = (eps, c_radius, a_nose, a_side, a_base,
normal_only)``, set only while :func:`_run_shift` traces; ``None`` everywhere else (the
warm-up's and every other caller's ``apply_faces`` pass through)."""


def _bump(theta, centres):
    return sum(jnp.maximum(0.0, jnp.cos(theta - t0)) ** 4 for t0 in centres)


def _shifted_apply_faces(x, v, alive, *args, **kwargs):
    """``boundary.apply_faces`` with the body hits' final positions shifted."""
    out = _ORIGINAL(x, v, alive, *args, **kwargs)
    p = _SHIFT["p"]
    if p is None:
        return out
    x_new, v_new, alive_new = out[0], out[1], out[2]
    dim = x.shape[1]
    hit = alive & alive_new & jnp.any(v_new != v, axis=1)
    rel = x_new - _SHIFT["centre"][None, :]
    r = jnp.sqrt(jnp.sum(rel * rel, axis=1))
    n = rel / jnp.where(r > 0.0, r, 1.0)[:, None]
    theta = jnp.arctan2(rel[:, 1], rel[:, 0])
    w = (p[1] + p[2] * _bump(theta, EM.BUMP_CENTRES["bump_nose"])
         + p[3] * _bump(theta, EM.BUMP_CENTRES["bump_side"])
         + p[4] * _bump(theta, EM.BUMP_CENTRES["bump_base"]))  # fmt: skip
    vin_n = jnp.sum(v[:, :dim] * n, axis=1)
    incoming = vin_n < -1e-12
    full = (v_new[:, :dim] - v[:, :dim]) / jnp.where(incoming, -vin_n, 1.0)[:, None]
    direction = jnp.where(p[5] > 0.5, n, full)
    move = hit & incoming
    x_out = x_new + jnp.where(move[:, None], p[0] * w[:, None] * direction, 0.0)
    return (x_out, *out[1:])


@functools.partial(jax.jit, static_argnames=("faces", "static", "capacity", "capture"))
def _run_shift(
    state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, step0,
    capture, shift,
):  # fmt: skip
    """``EM._run`` with the traced ``shift`` visible to :func:`_shifted_apply_faces`."""
    _SHIFT["p"] = shift
    try:
        return simulate.run(
            state, cstate, grid, sp, params, static, key, faces=faces, wall=wall, res=res,
            capacity=capacity, emit_moments="none", step0=step0, mesh=mesh, capture=capture,
            emit_surface=True,
        )  # fmt: skip
    finally:
        _SHIFT["p"] = None


def _captured(state0, cstate, common, static, capacity, step0, shift):
    """The captured window from ``state0`` (``EM._run_grown``'s attempt-bound regrowth)."""
    for _ in range(EM.A_MAX_RESIZES + 1):
        carry, aux = _run_shift(state0, cstate, *common, static, capacity, step0, False, shift)
        clipped = int(jnp.sum(aux[1].attempts_clipped))
        if not clipped:
            return aux, static
        peak = int(jnp.max(aux[1].max_attempts))
        a_max = max(static.A_max + 1, math.ceil(EM.A_MAX_GROWTH * peak))
        print(f"  A_max {static.A_max} clipped -> {a_max}", flush=True)
        static = dataclasses.replace(static, A_max=a_max)
    raise RuntimeError("A_max still clipped")


def _shift_vector(eps, control, mode):
    p = np.zeros(6)
    p[0] = eps
    p[WEIGHT_INDEX[control]] = 1.0
    p[5] = 1.0 if mode == "normal" else 0.0
    return jnp.asarray(p)


def _objectives(aux, cfgs, dim, scale_r):
    EM._check(aux[1])
    return {obj: EM.objective_value(c, aux[-1], dim, scale_r) for obj, c in cfgs.items()}


def run_seed(seed, smoke, eps_list, cases):
    cfgs = {}
    for obj, gate in GATES.items():
        cfg, _par, _steps, seed_start, _rows = V.gate_config(gate)
        if smoke:
            cfg = {**cfg, "fnum_scale": 16.0 * cfg["fnum_scale"], "warmup": 150, "steps": 40,
                   "window": 20}  # fmt: skip
        cfgs[obj] = cfg
    cfg = cfgs["drag"]
    key = jax.random.key(seed_start + seed)
    t0 = time.time()
    grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
        cfg, "radius", EM.nominal(cfg, "radius"), cfg["steps"]
    )
    _SHIFT["centre"] = jnp.asarray(EM.F.CASES[cfg["case"]]["centre"], dtype=jnp.float64)
    flow = FlowParams(
        nrho=jnp.float64(EM.F.NRHO * EM.knudsen_density(cfg)),
        temp=jnp.float64(EM.F.TEMP),
        vstream=jnp.array([EM.stream_speed(cfg), 0.0, 0.0]),
    )
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    common = (grid, sp, params, key, faces, wall, res, mesh)
    (state0, cstate0), static, _frames = EM._warm_up(
        cfg, state, cstate, common, static, capacity, None, 0
    )
    state0 = jax.block_until_ready(state0)
    t_warm = time.time() - t0
    scale_r = float(params.fnum[0]) / (cfg["window"] * float(params.dt))

    aux, static = _captured(state0, cstate0, common, static, capacity, cfg["warmup"],
                            jnp.zeros(6))  # fmt: skip
    j0 = _objectives(aux, cfgs, static.D, scale_r)
    row = {"seed": seed, "key": seed_start + seed, "J0": j0, "smoke": smoke, "shifts": [],
           "seconds_warmup": t_warm}  # fmt: skip
    print(f"seed {seed}: J0 {j0}  warm-up {t_warm:.0f}s", flush=True)
    for control, mode in cases:
        for eps in eps_list:
            t1 = time.time()
            aux, static = _captured(state0, cstate0, common, static, capacity, cfg["warmup"],
                                    _shift_vector(eps, control, mode))  # fmt: skip
            j = _objectives(aux, cfgs, static.D, scale_r)
            slope = {obj: (j[obj] - j0[obj]) / eps for obj in j}
            row["shifts"].append({"control": control, "mode": mode, "eps": eps, "J": j,
                                  "slope": slope})  # fmt: skip
            print(f"  {control:9s} {mode:6s} eps {eps:.0e}: slope drag {slope['drag']:+.5g}  "
                  f"heat {slope['heat']:+.5g}  {time.time() - t1:.0f}s", flush=True)  # fmt: skip
    row["seconds"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPS))
    ap.add_argument("--smoke", action="store_true", help="fnum x 16, 150 + 40 steps, *_smoke")
    args = ap.parse_args()
    boundary.apply_faces = _shifted_apply_faces
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    out = OUT / f"shift{tag}_s{args.seeds[0]}-{args.seeds[1]}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text().splitlines()}
    cases = CASES[:2] if args.smoke else CASES
    for seed in range(*args.seeds):
        if seed in done:
            continue
        row = run_seed(seed, args.smoke, args.eps, cases)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
