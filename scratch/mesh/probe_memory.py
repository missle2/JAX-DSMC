"""GPU memory per particle slot: allocator peak after (1) an uncaptured forward, (2) the
captured forward (records of the last `steps` steps), (3) the full shape_terms (replay,
two value-form backwards, shape VJP).  optc settings at a given fnum_scale (arg 1)."""
import dataclasses, json, sys, time
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward

fs = float(sys.argv[1])
ensemble_backward.configure("virtual", 1, 2, "staged")
cfg = O.default_config()
cfg.update(fnum_scale=fs, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls = O.initial_levelset(cfg)
dev = jax.devices()[0]
peak = lambda: dev.memory_stats()["peak_bytes_in_use"]
live = lambda: dev.memory_stats()["bytes_in_use"]
key = jax.random.key(32)
out = {"fnum_scale": fs}
s0, final, diag, rec, tallies, s = EM.forward(cfg, "radius", None, key, False, body=ls)
jax.block_until_ready(final)
out["N_slots"] = int(s["static"].N)
out["alive_end"] = int(np.asarray(final.alive).sum())
out["peak_forward"] = peak()
del s0, final, diag, rec, tallies
s0, final, diag, rec, tallies, s = EM.forward(cfg, "radius", None, key, True, body=ls)
jax.block_until_ready(rec)
out["peak_forward_captured"] = peak()
out["records_bytes"] = int(sum(x.nbytes for x in jax.tree.leaves(rec)))
out["records_fields"] = {k: int(v.nbytes) for k, v in rec._asdict().items() if hasattr(v, "nbytes")} if hasattr(rec, "_asdict") else None
del s0, final, diag, rec, tallies
t0 = time.time()
EM.shape_terms(cfg, key, ls, "radius", cfg["neighbors"], cfg["timing_band"], (cfg["variant"],))
out["peak_shape_terms"] = peak()
out["seconds_shape_terms"] = time.time() - t0
print(json.dumps(out))
