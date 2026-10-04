"""Paired re-run of the coarse cos(2 theta) gate (grid 1, fnum x4, lane tracker; the current
code's hit_ahead / none runs in scratch/mesh/mode2_coarse_now_*.json, same seeds) with the
hit-ahead class stopped at the slot's next birth.  The stored per-seed CRN FD is reused.
(The fine grid-2 gate does not fit the 4 GB laptop GPU.)

    PYTHONPATH=. python3 scratch/shape_audit/gate_class_fix.py [fixed|current] SEEDS...
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax

from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward, shape

sys.path.insert(0, "scratch/shape_audit")
from hit_ahead_refill_lib import per_step_hits  # noqa: E402

ART = "scratch/mesh/mode2_coarse_now_hit_ahead.json"
_old = shape.hit_ahead


def _later(later_next, xs):
    hit_n, born_n = xs
    later_k = hit_n | (later_next & ~born_n)
    return later_k, later_k


def hit_ahead_own(records, mesh):
    """The class of the slot's OWN particle: a body hit at a later step before the slot's
    next birth (hit[k+1] | later[k+1] & ~born[k+1])."""
    del mesh
    hit = per_step_hits(records)
    born = records.born_face >= 0
    _, out = lax.scan(_later, jnp.zeros(hit.shape[1], bool), (hit[1:], born[1:]), reverse=True)
    return jnp.concatenate((out, jnp.zeros((1, hit.shape[1]), bool)))


mode = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]
if mode == "fixed":
    shape.hit_ahead = hit_ahead_own
art = json.load(open(ART))
cfg = EM.default_config()
cfg.update({k: v for k, v in art["config"].items() if k in cfg})
ensemble_backward.configure("virtual", 1, 2, "staged")
for i in seeds:
    t0 = time.time()
    row = EM.ensemble_gradient(cfg, "mode2", jax.random.key(art["config"]["seed_start"] + i))
    old = art["rows"][i]
    fd = sum(old["fd"]) / len(old["fd"])
    print(json.dumps({
        "mode": mode, "seed": i, "gradient": row["gradient"], "stored": old["gradient"], "fd": fd,
        "tally": row["pathwise_tally"], "x": row["pathwise_x"], "v": row["pathwise_v"],
        "stored_x": old["pathwise_x"], "stored_v": old["pathwise_v"], "s": round(time.time() - t0),
    }), flush=True)
