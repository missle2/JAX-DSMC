"""Direct gate of the collisional grad-lambda machinery: a restitution factor e on the body's
specular reflection, v_out = v - (1 + e)(v.n) n (e = 1 is the forward), changes only the
reflected states -- no geometry, no volume, no hit times.  Adjoint prediction of dD/de:

    sum over window hits of d r/de = scale m (v.n) n_x          (the hit's own reward)
  + sum over captured hits of grad_v lam . (-(v.n) n)          (the v part)
                           + grad_x lam . (-(v.n) n)(dt - t)   (the x part)

with grad lam from the shape adjoint's own fits at the hit slots (single-bounce hits; the
rest counted), against the CRN FD of the windowed drag in e (forward patched).

    PYTHONPATH=. python3 scratch/bias/restitution.py OUT.jsonl SEED0 SEED1 '{overrides}'
"""

import functools
import inspect
import json
import sys
import textwrap
import time

import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape, wall_adjoint  # noqa: E402
from jax_dsmc.collide.ntc import step_v_post  # noqa: E402
from jax_dsmc.ensemble_device import node_gradients  # noqa: E402
from jax_dsmc.rng import STAGE_MOVE, key_for  # noqa: E402
from jax_dsmc.wall_adjoint import as_split  # noqa: E402

RESTITUTION = [1.0]
src = textwrap.dedent(inspect.getsource(wall_adjoint._hit_rows))
assert "v_spec = v - 2.0 * vn * normal" in src
src = src.replace("v_spec = v - 2.0 * vn * normal",
                  "v_spec = v - (1.0 + _RESTITUTION[0]) * vn * normal")
wall_adjoint._RESTITUTION = RESTITUTION
exec(compile(src, wall_adjoint.__file__, "exec"), wall_adjoint.__dict__)  # noqa: S102

FITS = {"one": (0, 32, None, "quadratic_full"), "one_pool_plane": (1, 96, 0.25, "quadratic_full"),
        "one_k16": (0, 16, None, "quadratic_full"), "one_affine": (0, 32, None, "affine"),
        "one_k64": (1, 64, 0.25, "quadratic_full")}
PRED = {}
_gradient_variants = shape.gradient_variants


def _step(carry, xs, geom, base_key, smap, fit, first, scale_r, mass, dt, npc_max, scale):
    x, v, alive = carry
    rec_k, lam_k, step = xs
    dim = x.shape[1]
    events = shape.wall_map(geom, x, v, alive, key_for(base_key, step, STAGE_MOVE), **smap)[5]
    split = as_split(events)
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    head = split.head
    body = (head.face[0] >= 2 * dim) & survivors
    later = jnp.zeros(x.shape[0], bool).at[jnp.where(split.slot >= 0, split.slot, x.shape[0])].max(
        jnp.any(split.tail.face >= 0, axis=0), mode="drop")
    single = body & ~later
    n = head.normal[0]
    vn = jnp.sum(head.v_in[0] * n, axis=1)
    dv = -vn[:, None] * n
    dxx = dv[:, :dim] * (dt - head.time[0])[:, None]
    tally = jnp.sum(jnp.where(single & (step >= first), scale_r * mass * vn * n[:, 0], 0.0))
    queries, _lost = shape._body_hit_slots(events, survivors, dim)
    pool, k, half, basis = fit
    gx, gv = node_gradients(rec_k.x, rec_k.v, rec_k.alive, lam_k, smap["grid"], npc_max, scale,
                            k, basis=basis, direction=n, pool=pool, queries=queries,
                            halfspace=half)
    px = jnp.sum(jnp.where(single[:, None], gx * dxx, 0.0))
    pv = jnp.sum(jnp.where(single[:, None], gv * dv, 0.0))
    out = (tally, px, pv, jnp.sum(single), jnp.sum(body & later))
    return (rec_k.x, step_v_post(rec_k), rec_k.alive), out


@functools.partial(jax.jit, static_argnames=("faces", "static", "fit", "first", "npc_max"))
def predict(records, lam, steps, state0, base_key, grid, faces, wall, mass, dt, mesh, static,
            scale, npc_max, fit, first, scale_r):
    smap = shape._setup(base_key, grid, faces, wall, mass, dt, mesh, 0.0, 0.0, static)["map"]
    body = functools.partial(_step, geom=shape.geometry(mesh), base_key=base_key, smap=smap,
                             fit=fit, first=first, scale_r=scale_r, mass=mass, dt=dt,
                             npc_max=npc_max, scale=scale)
    _, out = lax.scan(body, (state0.x, state0.v, state0.alive), (records, lam, steps))
    return tuple(jnp.sum(o) for o in out)


def gradient_variants(lam_timing, names=None, collision=None, **kw):
    first = int(kw["tally_weight"].shape[0] - jnp.sum(kw["tally_weight"]))
    first = int(kw["steps"][0]) + first
    scale_r = float(kw["tally_cotangent"].momentum[-1, 0])
    for name, fit in FITS.items():
        t, px, pv, n1, nm = predict(
            kw["records"], kw["lam"], kw["steps"], kw["state0"], kw["base_key"], kw["grid"],
            kw["faces"], kw["wall"], kw["mass"], kw["dt"], kw["mesh"], kw["static"],
            kw["scale"], kw["npc_max"], fit, first, scale_r)
        PRED[name] = {"tally": float(t), "x": float(px), "v": float(pv),
                      "single_hits": int(n1), "multi_hits": int(nm)}
    return _gradient_variants(lam_timing, names=("one",), collision=collision, **kw)


def drag(cfg, key, e):
    RESTITUTION[0] = e
    jax.clear_caches()
    _a, _b, diag, _r, tallies, st = EM.forward(cfg, "squash_y", cfg["squash"], key, False)
    EM._check(diag)
    scale_r = float(st["params"].fnum[0]) / (cfg["window"] * float(st["params"].dt))
    return EM.objective_value(cfg, tallies, st["static"].D, scale_r)


def main():
    path = sys.argv[1]
    over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    eps = over.pop("eps", 0.1)
    sk = over.pop("virtual", [2, 4])
    pullback = over.pop("pullback", "virtual")
    cfg = dict(OS.default_config(), squash=1.0, **over)
    ensemble_backward.configure(pullback, sk[0], sk[1], "staged")
    for i in range(int(sys.argv[2]), int(sys.argv[3])):
        key = jax.random.key(cfg["seed_start"] + i)
        t0 = time.time()
        RESTITUTION[0] = 1.0
        jax.clear_caches()
        shape.gradient_variants = gradient_variants
        try:
            _t, info, _s = EM.shape_terms(cfg, key, None, "squash_y", 32, 0.2, ("one",))
        finally:
            shape.gradient_variants = _gradient_variants
        fd = (drag(cfg, key, 1.0 + eps) - drag(cfg, key, 1.0 - eps)) / (2 * eps)
        RESTITUTION[0] = 1.0
        row = {"seed": i, "config": over, "eps": eps, "virtual": sk, "pullback": pullback,
               "objective": info["objective"], "pred": dict(PRED), "fd": fd,
               "seconds": time.time() - t0}
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
