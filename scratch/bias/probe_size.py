"""Timing / memory probe of one shape_terms seed + one forward for a small squash config.

    PYTHONPATH=. python3 scratch/bias/probe_size.py '{"fnum_scale": 4, "warmup": 0, ...}' S K
"""

import json
import sys
import time

import jax

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

over = json.loads(sys.argv[1])
cfg = dict(OS.default_config(), **over)
ensemble_backward.configure("virtual", int(sys.argv[2]), int(sys.argv[3]), "staged")
for rep in range(2):
    t0 = time.time()
    out = EM.forward(cfg, "squash_y", cfg["squash"], jax.random.key(5 + rep), False)
    jax.block_until_ready(out[4])
    t_fwd = time.time() - t0
    t0 = time.time()
    terms, info, st = EM.shape_terms(
        cfg, jax.random.key(5 + rep), None, "squash_y", 32, 0.2, ("one", "along")
    )
    t_sh = time.time() - t0
    mem = jax.devices()[0].memory_stats() or {}
    pick = lambda g: EM.control_gradient(cfg, "squash_y", st["grid"], g)  # noqa: E731
    print(json.dumps({
        "rep": rep, "fwd": round(t_fwd, 1), "shape_terms": round(t_sh, 1),
        "backward": round(info["seconds_backward"], 1), "shape": round(info["seconds_shape"], 1),
        "peak_GB": round(mem.get("peak_bytes_in_use", 0) / 1e9, 2), "alive": info["alive_final"],
        "hits": info["hits"], "D": info["objective"],
        "tally": pick(terms["one"].pathwise_tally),
        "x_one": pick(terms["one"].pathwise_x), "x_along": pick(terms["along"].pathwise_x),
        "v": pick(terms["one"].pathwise_v),
    }), flush=True)
