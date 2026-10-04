"""Exact free-molecular reference for the squash gradient dD/ds, per seed, beside the
adjoint's terms and the CRN FD.

Without collisions every particle's path is a deterministic function of the geometry
(fixed draws: inflow, diffuse re-emission), so the forward-mode derivative of the
replayed wall map, CHAINED step to step (x, v and their tangents carried; newborns enter
with zero tangent), is the exact pathwise derivative of the realised windowed drag.  For
a specular body the drag reward is continuous at hit / miss switches (a grazing hit
transfers no momentum), so this is unbiased.  Two horizons:

* ``full``: tangents from step 0 (the empty box) -- the whole dependence;
* ``trunc``: tangents from the capture start (``warmup``) -- what the adjoint's
  ``steps``-step horizon can see.

and two hit-time derivatives: the body's ``grazing`` cut-off as configured, and 0 (exact).

    PYTHONPATH=. python3 scratch/bias/fm_exact.py OUT.jsonl S SEED0 SEED1 [overrides-json]
"""

import dataclasses
import functools
import json
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, "examples/mesh")
import forward as F  # noqa: E402
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, initialize, shape  # noqa: E402
from jax_dsmc.collide.ntc import step_v_post  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

BLOCK = 100
_gradient_variants = shape.gradient_variants


CLASSES = [True]


def gradient_variants(lam_timing, names=None, collision=None, **kw):
    """Every variant twice: the mixed pool (as configured) and the hit-ahead class."""
    out = _gradient_variants(lam_timing, names=names, collision=collision, **kw)
    if not CLASSES[0]:
        return out
    kw = dict(kw, value_class="hit_ahead")
    cls = _gradient_variants(lam_timing, names=names, collision=collision, **kw)
    out.update({f"{n}_class": t for n, t in cls.items()})
    return out


shape.gradient_variants = gradient_variants


def _tangent_step(carry, xs, geom, base_key, smap, first, scale_r, dim):
    x, v, alive, total = carry
    rec_k, step = xs
    key = key_for(base_key, step, STAGE_MOVE)
    x_post, v_post, alive_post, _lp, tallies, _ev = shape.wall_map(geom, x, v, alive, key, **smap)
    drag = jnp.sum(tallies.momentum[2 * dim :, 0]) * scale_r
    total = total + jnp.where(step >= first, drag, 0.0)
    surv = alive & rec_k.alive & (rec_k.born_face < 0)
    x_next = jnp.where(surv[:, None], x_post, rec_k.x)
    v_next = jnp.where(surv[:, None], v_post, step_v_post(rec_k))
    return (x_next, v_next, rec_k.alive, total), None


def _block(geom, x0, v0, alive0, recs, steps, base_key, smap, first, scale_r, dim):
    body = functools.partial(
        _tangent_step, geom=geom, base_key=base_key, smap=smap, first=first,
        scale_r=scale_r, dim=dim,
    )
    (x, v, _a, total), _ = lax.scan(body, (x0, v0, alive0, jnp.float64(0.0)), (recs, steps))
    return x, v, total


@functools.partial(jax.jit, static_argnames=("faces", "static", "first", "scale_r", "dim"))
def block_jvp(geom, x0, v0, dgeom, dx0, dv0, alive0, recs, steps, base_key, grid, faces,
              wall, mass, dt, mesh, static, first, scale_r, dim):
    smap = shape._setup(base_key, grid, faces, wall, mass, dt, mesh, 0.0, 0.0, static)["map"]
    f = functools.partial(_block, alive0=alive0, recs=recs, steps=steps, base_key=base_key,
                          smap=smap, first=first, scale_r=scale_r, dim=dim)
    return jax.jvp(f, (geom, x0, v0), (dgeom, dx0, dv0))


def exact(cfg, s, key):
    grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
        cfg, "squash_y", s, BLOCK
    )
    flow = FlowParams(nrho=jnp.float64(F.NRHO * EM.knudsen_density(cfg)),
                      temp=jnp.float64(F.TEMP), vstream=jnp.array([EM.stream_speed(cfg), 0.0, 0.0]))
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    dim = static.D
    scale_r = float(params.fnum[0]) / (cfg["window"] * float(params.dt))
    total_steps = cfg["warmup"] + cfg["steps"]
    first = total_steps - cfg["window"]
    geom = shape.geometry(mesh)
    _c, dgeom = jax.jvp(functools.partial(EM._coef_of, cfg=dict(cfg, squash=s),
                                          parameter="squash_y", grid=grid),
                        (jnp.float64(s),), (jnp.float64(1.0),))
    meshes = {"gz": mesh,
              "exact": dataclasses.replace(mesh, body=dataclasses.replace(mesh.body, grazing=0.0))}
    zeros = jnp.zeros_like(state.x), jnp.zeros_like(state.v)
    # tangents started at several steps: the horizon dependence (trunc = the adjoint's)
    starts = {"full": 0, "h600": cfg["warmup"] - 300, "h400": cfg["warmup"] - 100,
              "trunc": cfg["warmup"]}
    tang = {f"gz_{h}": zeros for h in starts}
    tang["exact_trunc"] = zeros
    out = {k: 0.0 for k in tang}
    value = 0.0
    common = (grid, sp, params, key, faces, wall, res, mesh)
    for k0 in range(0, total_steps, BLOCK):
        x0, v0, alive0 = state.x, state.v, state.alive
        (state, cstate), aux = EM._run(state, cstate, *common, static, capacity, k0, True)
        EM._check(aux[1])
        recs = aux[2]
        steps = k0 + jnp.arange(BLOCK, dtype=jnp.int32)
        for name in list(tang):
            m, h = name.split("_")
            if k0 < starts[h]:
                continue
            (x, v, val), (dx, dv, dval) = block_jvp(
                geom, x0, v0, dgeom, *tang[name], alive0, recs, steps, key, grid, faces, wall,
                sp.mass[0], params.dt, meshes[m], static, first, scale_r, dim)
            tang[name] = (dx, dv)
            out[name] += float(dval)
            if name == "gz_full":
                value += float(val)
        del recs, aux
    return value, out


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    seeds = range(int(sys.argv[3]), int(sys.argv[4]))
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    CLASSES[0] = over.pop("classes", True)
    fd_steps = over.pop("fd_steps", [0.05])
    cfg = dict(OS.default_config(), squash=s, **over)
    variants = ("one", "one_pool", "along", "along_pool")
    ensemble_backward.configure("virtual", 1, 2, "staged")
    for i in seeds:
        key = jax.random.key(cfg["seed_start"] + i)
        t0 = time.time()
        terms, info, st = EM.shape_terms(cfg, key, None, "squash_y", 32, 0.2, variants)
        pick = functools.partial(EM.control_gradient, cfg, "squash_y", st["grid"])
        row = {"s": s, "seed": i, "objective": info["objective"],
               "tally": pick(terms["one"].pathwise_tally),
               "score": pick(terms["one"].score), "timing": pick(terms["one"].timing),
               "volume": 0.0 if st["volume"] is None else pick(st["volume"]),
               "variants": {n: {"x": pick(t.pathwise_x), "v": pick(t.pathwise_v)}
                            for n, t in terms.items()},
               "newborn_hit_share": info["newborn_hit_share"]}
        del terms, st
        t1 = time.time()
        value, ex = exact(cfg, s, key)
        row.update({"jvp_objective": value, "exact": ex, "seconds_exact": time.time() - t1})
        row["fd"] = EM.finite_differences(cfg, "squash_y", key, fd_steps)
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
