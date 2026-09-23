"""GPU memory of the uncaptured forward alone vs slot count: allocator peak and the
state's own bytes.  Static shapes, so a short run (warmup 100, steps 20) has the same
footprint as the full one; optc settings at fnum_scale = arg 1."""
import json, sys, time
import numpy as np
import jax
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM

fs = float(sys.argv[1])
cfg = O.default_config()
cfg.update(fnum_scale=fs, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05,
           warmup=100, steps=20)
_, ls = O.initial_levelset(cfg)
dev = jax.devices()[0]
t0 = time.time()
s0, final, diag, rec, tallies, s = EM.forward(cfg, "radius", None, jax.random.key(32), False, body=ls)
jax.block_until_ready(final)
st = dev.memory_stats()
print(json.dumps({"fnum_scale": fs, "N_slots": int(s["static"].N),
                  "state_bytes": int(sum(x.nbytes for x in jax.tree.leaves(final))),
                  "peak": st["peak_bytes_in_use"], "live": st["bytes_in_use"],
                  "seconds": round(time.time() - t0, 1)}))
