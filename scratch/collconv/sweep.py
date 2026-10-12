"""Path 2: the Kn 0.5 squash adjoint's dependence on its hyper-parameters, PAIRED: every
setting runs on the same seed's forward (the forward and replay are identical across
settings; only the backward / fits differ), so differences between settings carry none of
the forward's noise.

Settings: ``{"S": .., "K": .., "k": .., ["basis": ..], ["grazing": ..], ["horizon": M]}`` --
virtual pull-back substeps and partners, the backward fit's neighbour count
(``shape_terms(neighbors=)``) and its basis (``ensemble_backward.VIRTUAL_BASIS``: a
``"cell:..."`` per-cell fit ignores ``k``), the grazing cut-off of the hit derivative (the
forward does not depend on it: paired), and the captured horizon ``M`` (the same forward --
absolute step indices -- captured for its last ``M`` of ``warmup + steps`` steps; the FD is
the configured run's).  Every run records every
``shape.VARIANTS`` entry, extended here by the shape fit's own neighbour counts (unpooled
16 / 64, pooled 48 / 192), and the tally / timing / volume / order terms.  One CRN FD at
``fd_steps`` per seed (shared by the settings).

    PYTHONPATH=. python3 scratch/collconv/sweep.py OUT.jsonl S SEED0 SEED1 '{overrides}'
overrides: the squash config (fnum_scale, knudsen, a_max, ...), ``settings`` (list),
``fd_steps`` (default [0.2]; [] = none), ``timing_band`` (default 10).
"""

import functools
import json
import sys
import time

import jax

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

EXTRA = {
    "one_k16": (0, 16, False),
    "one_k64": (0, 64, False),
    "along_k64": (0, 64, True),
    "one_pool_plane_k48": (1, 48, False, 0.25),
    "one_pool_plane_k192": (1, 192, False, 0.25),
    "along_pool_plane_k48": (1, 48, True, 0.25),
    "along_pool_plane_k192": (1, 192, True, 0.25),
}
DEFAULT_SETTINGS = [{"S": 4, "K": 8, "k": 32}]


def setting_tag(st):
    tag = f"S{st['S']}K{st['K']}" + (st["basis"] if "basis" in st else f"k{st['k']}")
    if "grazing" in st:
        tag += f"_gz{st['grazing']:g}"
    if "horizon" in st:
        tag += f"_h{st['horizon']}"
    return tag


def terms_row(cfg, key, setting, tband):
    basis = setting.get("basis", "quadratic_full")
    if "grazing" in setting:
        cfg = dict(cfg, grazing=float(setting["grazing"]))
    if "horizon" in setting:
        total = cfg["warmup"] + cfg["steps"]
        cfg = dict(cfg, warmup=total - int(setting["horizon"]), steps=int(setting["horizon"]))
    ensemble_backward.configure("virtual", setting["S"], setting["K"], "staged", basis)
    t0 = time.time()
    terms, info, st = EM.shape_terms(
        cfg, key, None, "squash_y", setting["k"], tband, tuple(shape.VARIANTS)
    )
    pick = functools.partial(EM.control_gradient, cfg, "squash_y", st["grid"])
    one = terms["one"]
    out = {
        "objective": info["objective"],
        "tally": pick(one.pathwise_tally),
        "score": pick(one.score),
        "silhouette": pick(one.silhouette),
        "timing": pick(one.timing),
        "volume": 0.0 if st["volume"] is None else pick(st["volume"]),
        "order": 0.0 if one.order is None else pick(one.order),
        "variants": {
            n: {"x": pick(t.pathwise_x), "v": pick(t.pathwise_v), "sil": pick(t.silhouette)}
            for n, t in terms.items()
        },
        "seconds": time.time() - t0,
        "seconds_backward": info["seconds_backward"],
        "seconds_shape": info["seconds_shape"],
    }
    del terms, st
    return out


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    settings = over.pop("settings", DEFAULT_SETTINGS)
    hs = over.pop("fd_steps", [0.2])
    tband = over.pop("timing_band", 10)
    extra = over.pop("extra_variants", True)
    if extra:
        shape.VARIANTS.update({n: v for n, v in EXTRA.items()})
    cfg = dict(OS.default_config(), squash=s, **over)
    for i in range(int(sys.argv[3]), int(sys.argv[4])):
        key = jax.random.key(cfg["seed_start"] + i)
        row = {"s": s, "seed": i, "config": over, "timing_band": tband, "runs": {}}
        t0 = time.time()
        for st in settings:
            row["runs"][setting_tag(st)] = terms_row(cfg, key, st, tband)
        row["fd"] = EM.finite_differences(cfg, "squash_y", key, hs) if hs else []
        row["fd_steps"] = hs
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
