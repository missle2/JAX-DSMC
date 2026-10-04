"""Collisional squash gate, per seed (CRN): the adjoint's terms beside two FDs.

* ``fd``: the full CRN central difference of the windowed drag;
* ``fd_frozen``: the same with every collision cell's flow volume frozen at the base
  shape (EM.build patched) -- ``fd - fd_frozen`` is the cut-volume (NTC 1/V_c) part,
  which the adjoint's ``volume`` term must reproduce;
* the adjoint: tally, score, silhouette, volume, and the x / v parts of every grad_x
  lambda variant, the along-flight ones with (``along*``) and without (``along*_free``)
  the collision term ``-L* lambda`` of the identity.

    PYTHONPATH=. python3 scratch/bias/kn_gate.py OUT.jsonl S SEED0 SEED1 '{overrides}'
"""

import dataclasses
import functools
import json
import sys
import time

import jax

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

_gradient_variants = shape.gradient_variants
_build = EM.build
FROZEN = {}


def gradient_variants(lam_timing, names=None, collision=None, **kw):
    out = _gradient_variants(lam_timing, names=names, collision=collision, **kw)
    if collision is not None and FREE[0]:
        names = ("along", "along_pool", "along_pool_plane")
        free = _gradient_variants(lam_timing, names=names, collision=None, **kw)
        out.update({f"{n}_free": t for n, t in free.items()})
    return out


FREE = [True]


def build_frozen(cfg, parameter, value, steps, body=None):
    out = _build(cfg, parameter, value, steps, body)
    if "volume" in FROZEN:
        grid = out[0]
        if grid.volume.shape != FROZEN["volume"].shape or int(grid.n_sub):
            raise RuntimeError("cell layout changed under the FD step")
        out = (dataclasses.replace(grid, volume=FROZEN["volume"]), *out[1:])
    return out


shape.gradient_variants = gradient_variants


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    seeds = range(int(sys.argv[3]), int(sys.argv[4]))
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    hs = over.pop("fd_steps", [0.05])
    sk = over.pop("virtual", [4, 8])
    tband = over.pop("timing_band", 0.2)
    pullback = over.pop("pullback", "virtual")
    cfg = dict(OS.default_config(), squash=s, **over)
    ensemble_backward.configure(pullback, sk[0], sk[1], "staged")
    variants = tuple(shape.VARIANTS)
    collisional = cfg.get("knudsen", 0.0) > 0.0
    FREE[0] = collisional
    for i in seeds:
        key = jax.random.key(cfg["seed_start"] + i)
        t0 = time.time()
        terms, info, st = EM.shape_terms(cfg, key, None, "squash_y", 32, tband, variants)
        t_adj = time.time() - t0
        pick = functools.partial(EM.control_gradient, cfg, "squash_y", st["grid"])
        one = terms["one"]
        row = {
            "s": s, "seed": i, "config": over, "virtual": sk, "timing_band": tband,
            "pullback": pullback, "objective": info["objective"],
            "tally": pick(one.pathwise_tally), "score": pick(one.score),
            "silhouette": pick(one.silhouette), "timing": pick(one.timing),
            "volume": 0.0 if st["volume"] is None else pick(st["volume"]),
            "order": 0.0 if one.order is None else pick(one.order),
            "variants": {n: {"x": pick(t.pathwise_x), "v": pick(t.pathwise_v),
                             "sil": pick(t.silhouette)}
                         for n, t in terms.items()},
            "newborn_hit_share": info["newborn_hit_share"], "hits": info["hits"],
            "seconds_adjoint": t_adj,
        }
        del terms, st
        row["fd"] = EM.finite_differences(cfg, "squash_y", key, hs)
        row["fd_frozen"] = row["fd"]
        if collisional:
            FROZEN["volume"] = _build(cfg, "squash_y", s, 1)[0].volume
            EM.build = build_frozen
            try:
                row["fd_frozen"] = EM.finite_differences(cfg, "squash_y", key, hs)
            finally:
                EM.build = _build
                FROZEN.clear()
        row["fd_steps"] = hs
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
