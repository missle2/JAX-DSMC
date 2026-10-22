"""Where the shape step's time goes (local GPU, reduced size as bench.py), 2026-10-22: the
shape step (shape.gradient_variants as EM.shape_terms calls it) on one captured record and
backward, each piece switched off in turn (cost attribution only -- the gradients are wrong):

* ``shipped``;
* ``no_fits``   -- ``node_gradients`` (the grad-lambda fits at the body hits) returns zeros;
* ``no_order``  -- ``order_jump`` (two value queries per hit) returns zeros;
* ``no_pull``   -- the batched pull-back skipped (``vjp`` linearisation kept);
* ``primal``    -- the wall map evaluated without ``jax.vjp`` and nothing pulled back.

    PYTHONPATH=. python3 scratch/shapefast/shape_parts.py [overrides-json]
"""

import functools
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/shapefast")
import optimize as OPT  # noqa: E402
from backward import setup  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

SHIPPED = {name: getattr(shape, name) for name in ("node_gradients", "order_jump", "_pullbacks")}
SHIPPED_VJP = jax.vjp


def zero_fits(rec_x, rec_v, *args, **kwargs):
    del args, kwargs
    return jnp.zeros_like(rec_x), jnp.zeros_like(rec_v)


def zero_order(events, survivors, rec_k, *args):
    del events, survivors, args
    return jnp.zeros(rec_k.x.shape[0])


def zero_pull(vjp, cotangents, geom):
    del vjp
    return [jax.tree.map(jnp.zeros_like, geom) for _ in cotangents]


def primal_vjp(fn, geom, has_aux=False):
    out, aux = fn(geom)
    return out, None, aux


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
    small = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    args, extra = setup(cfg, ls, key, full=True, static_over=small)
    dev, dev_t = EM._backwards(*args)
    kwargs = extra["shape_kwargs"](dev, dev_t)
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    settings = {
        "shipped": {},
        "no_fits": {"node_gradients": zero_fits},
        "no_order": {"order_jump": zero_order},
        "no_pull": {"_pullbacks": functools.partial(zero_pull, geom=extra["geom"])},
        "primal": {
            "_pullbacks": functools.partial(zero_pull, geom=extra["geom"]),
            "vjp": primal_vjp,
        },
    }
    for name, patch in settings.items():
        if only and name not in only:
            continue
        for attr, fn in SHIPPED.items():
            setattr(shape, attr, patch.get(attr, fn))
        shape.jax.vjp = patch.get("vjp", SHIPPED_VJP)
        shape._gradient.clear_cache()
        for label in ("compile", "timed"):
            t0 = time.time()
            terms = shape.gradient_variants(dev_t.lam, names=(cfg["variant"],),
                                            collision=dev.collision, **kwargs)  # fmt: skip
            jax.block_until_ready(terms)
            print(f"{name:10s} {label:8s} {time.time() - t0:7.2f} s", flush=True)
        if name == "shipped":
            t = terms[cfg["variant"]]
            parts = (t.pathwise, t.score, t.silhouette, t.timing, t.order)
            total = sum(np.asarray(jax.tree.leaves(a)[0]) for a in parts if a is not None)
            tag = "small" if small else "full"
            np.save(f"scratch/shapefast/grad_{tag}.npy", total)
            print(f"  replay error {float(t.replay_error):.3g}  |g| {np.linalg.norm(total):.6g}",
                  flush=True)  # fmt: skip
    shape.jax.vjp = SHIPPED_VJP


if __name__ == "__main__":
    main()
