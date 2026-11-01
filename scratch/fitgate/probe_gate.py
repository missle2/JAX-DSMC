"""The wall-probe gate (2026-11-01): the adjoint's value gradients at the particles leaving the
body against central differences of the same perturbation, the geometry unchanged.

The M 10 specular circle's +8..11 % is the same for every fit, dt and cell size; the
lambda-mediated terms (pathwise_x, pathwise_v, volume, order) carry about half the collisional
reduction the FD needs.  Those terms are the fitted ``grad_x lambda`` / ``grad_v lambda`` at
the post-move states of the body-hit particles, contracted with the geometry's displacement of
them.  Here the same fits are contracted with a displacement we impose instead, in the
captured steps only (the adjoint's horizon), on exactly those particles (a free box: only the
body changes a surviving particle's velocity in the move):

* ``xn``: ``x_post += eps n``, ``n`` the level set's unit normal at ``x_post`` (what
  ``pathwise_x`` contracts ``grad_x lambda`` with, ``2 delta n`` for a wall moved by ``delta``);
* ``vs``: ``v_post *= 1 + eps`` (``grad_v lambda . v``);
* ``vr``: ``v_post`` rotated by ``eps`` in the plane (``grad_v lambda . (-v_y, v_x, 0)``).

Prediction per fit (every estimator of ``--variants``, its own ``grad lambda`` and the
along-flight identity where it has one): ``sum_k sum_hits grad lambda . d``.  FD: the
objective with the probe at ``+-eps`` (common random numbers; the warm-up run once and the
captured steps rerun), two ``eps`` per probe (linearity).

The forward hook wraps ``boundary.apply_faces``; the probe is read when the captured run is
traced, so it rides in the run's static configuration (:class:`ProbeStatic`, a field of the
jit cache key) -- a cached program never sees another probe's setting.

    PYTHONPATH=. python3 scratch/fitgate/probe_gate.py --gate c10_spec --seeds 0 2 [--smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import boundary, ensemble_backward, shape  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide.ntc import step_v_post  # noqa: E402
from jax_dsmc.ensemble_device import node_gradients  # noqa: E402
from jax_dsmc.mesh import levelset as levelset_mod  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402
from jax_dsmc.state import SimStatic  # noqa: E402

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path("examples/mesh/results/fitgate")
PROBES = {"xn": (0.01, 0.02), "vs": (0.02, 0.01), "vr": (0.02, 0.01)}
"""Probe sizes.  ``xn`` is one-sided (outward only: ``-eps`` would put the particles within
``eps`` of the wall inside the body, ``inside_body``): ``J(eps)``, ``J(2 eps)`` with the
second-order one-sided difference ``(4 J(eps) - J(2 eps) - 3 J(0)) / (2 eps)`` and the plain
forward ones; the velocity probes are central at both sizes."""
DEFAULT_VARIANTS = ("along_pool_plane", "cell_pool_plane", "cell_along_pool_plane",
                    "cellxv_along_pool_plane", "one_pool_plane")  # fmt: skip
PROBE = {"probe": ()}
_APPLY_FACES = boundary.apply_faces
_GRADIENT_VARIANTS = shape.gradient_variants
STASH = {}


@dataclasses.dataclass(frozen=True)
class ProbeStatic(SimStatic):
    """``SimStatic`` with the probe ``(kind, eps)`` as one more static field (the jit key)."""

    probe: tuple = ()


def probed(static, probe):
    fields = {f.name: getattr(static, f.name) for f in dataclasses.fields(SimStatic)}
    return ProbeStatic(**fields, probe=probe)


def unit_normal(mesh, x):
    _phi, g = levelset_mod.value_and_grad(mesh.body, x)
    return g / jnp.maximum(jnp.linalg.norm(g, axis=1, keepdims=True), 1e-30)


def perturb(kind, eps, x, v, hit, mesh, grid):
    """The probe on the hit rows: ``(x, v)``."""
    if kind == "xn":
        x_new = x + eps * unit_normal(mesh, x)
        span = grid.hi - grid.lo
        x_new = jnp.clip(x_new, grid.lo + 1e-9 * span, grid.hi - 1e-9 * span)
        return jnp.where(hit[:, None], x_new, x), v
    if kind == "vs":
        return x, jnp.where(hit[:, None], (1.0 + eps) * v, v)
    if kind == "vr":
        c, s = jnp.cos(eps), jnp.sin(eps)
        rot = v.at[:, 0].set(c * v[:, 0] - s * v[:, 1]).at[:, 1].set(s * v[:, 0] + c * v[:, 1])
        return x, jnp.where(hit[:, None], rot, v)
    raise ValueError(kind)


def probed_apply_faces(x, v, alive, grid, *args, **kwargs):
    """``boundary.apply_faces`` with :data:`PROBE` applied (read at trace time) to the
    particles the body reflected: alive before and after, velocity changed."""
    out = _APPLY_FACES(x, v, alive, grid, *args, **kwargs)
    mesh = kwargs.get("mesh")
    if not PROBE["probe"] or mesh is None:
        return out
    kind, eps = PROBE["probe"]
    x2, v2, alive2, *rest = out
    hit = alive & alive2 & jnp.any(v2 != v, axis=1)
    x2, v2 = perturb(kind, eps, x2, v2, hit, mesh, grid)
    return (x2, v2, alive2, *rest)


def stash_gradient_variants(lam_timing, names=None, collision=None, **kwargs):
    """``shape.gradient_variants``, keeping its inputs for the prediction scan."""
    STASH.clear()
    STASH.update(kwargs, lam_timing=lam_timing, collision=collision)
    return _GRADIENT_VARIANTS(lam_timing, names=names, collision=collision, **kwargs)


def _probe_step(carry, xs, geom, setup, fits, mesh):
    """One captured step of the prediction: per fit ``(sum grad_x . n, sum grad_v . v,
    sum grad_v . rot v)`` over the step's surviving body-hit slots."""
    x, v, alive = carry
    rec_k, lam_k, step, lam_timing_k, coll_k = xs
    key = key_for(setup["base_key"], step, STAGE_MOVE)
    out = shape.wall_map(geom, x=x, v=v, alive=alive, key=key, **setup["map"])
    events = out[-1]
    dim = rec_k.x.shape[1]
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    queries, _lost = shape._body_hit_slots(events, survivors, dim)
    hit = jnp.zeros(x.shape[0], bool).at[queries].set(True, mode="drop") & survivors
    direction = shape._last_normal(events)
    n_hat = unit_normal(mesh, rec_k.x)
    vp = rec_k.v
    rot = jnp.stack((-vp[:, 1], vp[:, 0], jnp.zeros_like(vp[:, 0])), axis=1)
    m = hit[:, None]
    sums = []
    for fit in fits:
        settings = {k: val for k, val in fit.items() if k != "along"}
        gx, gv = node_gradients(
            rec_k.x, rec_k.v, rec_k.alive, lam_k, setup["map"]["grid"], direction=direction,
            klass=None, queries=queries, **settings,
        )  # fmt: skip
        if fit["along"]:
            t_k = lam_timing_k if coll_k is None else lam_timing_k + coll_k / setup["map"]["dt"]
            gx = shape.along_flight(gx, rec_k.v, t_k)
        sums.append(jnp.stack((jnp.sum(jnp.where(m, gx * n_hat, 0.0)),
                               jnp.sum(jnp.where(m, gv * vp, 0.0)),
                               jnp.sum(jnp.where(m, gv * rot, 0.0)))))  # fmt: skip
    return (rec_k.x, step_v_post(rec_k), rec_k.alive), (jnp.stack(sums), jnp.sum(hit))


def predictions(variants):
    """``{variant: {"xn", "vs", "vr"}}`` from the stashed shape-gradient inputs, and the
    captured hits."""
    kw = STASH
    setup = shape._setup(
        kw["base_key"], kw["grid"], kw["faces"], kw["wall"], kw["mass"], kw["dt"], kw["mesh"],
        0.0, 0.0, kw["static"], kw.get("reward_weight"),
    )  # fmt: skip
    fits = []
    for name in variants:
        e = shape.VARIANTS[name]
        fits.append({
            "npc_max": kw["npc_max"], "scale": kw["scale"], "k": e[1],
            "basis": e[4] if len(e) > 4 else kw.get("basis", "quadratic_full"),
            "chunk": kw.get("chunk", 4096), "pool": e[0], "along": e[2],
            "halfspace": e[3] if len(e) > 3 else None,
        })  # fmt: skip
    body = functools.partial(
        _probe_step, geom=shape.geometry(kw["mesh"]), setup=setup, fits=tuple(fits),
        mesh=kw["mesh"],
    )  # fmt: skip
    m = kw["lam"].shape[0]
    lam_t = kw["lam_timing"] if kw["lam_timing"] is not None else jnp.zeros_like(kw["lam"])
    xs = (kw["records"], kw["lam"], kw["steps"], lam_t, kw["collision"])
    s0 = kw["state0"]
    _, (sums, hits) = lax.scan(body, (s0.x, s0.v, s0.alive), xs)
    total = jnp.sum(sums, axis=0)
    del m
    out = {
        name: {p: float(total[i, j]) for j, p in enumerate(("xn", "vs", "vr"))}
        for i, name in enumerate(variants)
    }
    return out, int(jnp.sum(hits))


def warm(cfg, parameter, value, key):
    """:func:`EM.forward` up to the captured steps: ``(state0, cstate, common, static,
    capacity, params)``."""
    grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
        cfg, parameter, value, cfg["steps"]
    )
    for bc in faces.bcs:
        if bc in (BC.SPECULAR, BC.DIFFUSE):
            raise ValueError("the probe's hit test needs a box without reflecting faces")
    seen = EM._a_max_seen.get((static.N, tuple(grid.n)), 0)
    static = dataclasses.replace(static, A_max=max(static.A_max, seen))
    flow = EM.FlowParams(
        nrho=jnp.float64(EM.F.NRHO * EM.knudsen_density(cfg)),
        temp=jnp.float64(EM.F.TEMP),
        vstream=jnp.array([EM.stream_speed(cfg), 0.0, 0.0]),
    )
    state = EM.initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = EM.initialize.initial_collide_state(flow, grid, sp, static)
    common = (grid, sp, params, key, faces, wall, res, mesh)
    (state, cstate), static, _frames = EM._warm_up(cfg, state, cstate, common, static, capacity,
                                                   None, 0)  # fmt: skip
    return state, cstate, common, static, capacity, params


def captured_objective(cfg, warmed, probe):
    """The objective of the captured steps from the warm state, with ``probe`` (or none)."""
    state, cstate, common, static, capacity, params = warmed
    PROBE["probe"] = probe
    try:
        _carry, aux, _st = EM._run_grown(state, cstate, common, probed(static, probe), capacity,
                                         cfg["warmup"], False)  # fmt: skip
    finally:
        PROBE["probe"] = ()
    EM._check(aux[1])
    scale_r = float(params.fnum[0]) / (cfg["window"] * float(params.dt))
    return EM.objective_value(cfg, aux[-1], static.D, scale_r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first, end (exclusive)")
    ap.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    ap.add_argument("--smoke", action="store_true",
                    help="fnum x 16, 300 + 60 steps (window 40): the plumbing, locally")  # fmt: skip
    ap.add_argument("--fnum", type=float, default=16.0, help="the smoke's fnum factor")
    args = ap.parse_args(argv)
    cfg, parameter, _fd, seed_start, _rows = V.gate_config(args.gate)
    name = args.gate
    if args.smoke:
        cfg = {**cfg, "fnum_scale": args.fnum * cfg["fnum_scale"], "warmup": 300, "steps": 60,
               "window": 40}  # fmt: skip
        name = f"{args.gate}_smoke_f{args.fnum:g}"
    out = OUT_DIR / f"probe_{name}_s{args.seeds[0]}-{args.seeds[1]}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary.apply_faces = probed_apply_faces
    shape.gradient_variants = stash_gradient_variants
    ensemble_backward.configure("virtual", 4, 8, "staged", V.BASES["base"])
    value = float(EM.nominal(cfg, parameter))
    for seed in range(*args.seeds):
        key = jax.random.key(seed_start + seed)
        t0 = time.time()
        _terms, info, _s = EM.shape_terms(
            cfg, key, None, parameter, 32, cfg.get("timing_band", 10.0), tuple(args.variants)
        )
        pred, hits = predictions(args.variants)
        STASH.clear()
        print(json.dumps({"seed": seed, "hits": hits, "pred": pred}), flush=True)
        warmed = warm(cfg, parameter, value, key)
        j0 = captured_objective(cfg, warmed, ())
        fd = {}
        e = PROBES["xn"][0]
        j1 = captured_objective(cfg, warmed, ("xn", e))
        j2 = captured_objective(cfg, warmed, ("xn", 2.0 * e))
        fd["xn"] = [(4.0 * j1 - j2 - 3.0 * j0) / (2.0 * e), (j1 - j0) / e, (j2 - j0) / (2.0 * e)]
        for kind in ("vs", "vr"):
            fd[kind] = []
            for eps in PROBES[kind]:
                jp = captured_objective(cfg, warmed, (kind, eps))
                jm = captured_objective(cfg, warmed, (kind, -eps))
                fd[kind].append((jp - jm) / (2.0 * eps))
            print(json.dumps({"seed": seed, kind: fd[kind]}), flush=True)
        row = {"gate": args.gate, "seed": seed, "smoke": args.smoke, "objective": info["objective"],
               "objective_rerun": j0, "hits": hits, "eps": PROBES, "fd": fd, "pred": pred,
               "seconds": time.time() - t0}  # fmt: skip
        print(json.dumps(row), flush=True)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
