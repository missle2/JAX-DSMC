"""Where is the adjoint chain's memory peak?  shape_terms with the allocator's peak and
live bytes printed after each stage (forward, replay, backward, timing backward, shape
VJP).  optc settings, lane tracker, fnum_scale = arg 1."""
import json, sys
import jax, jax.numpy as jnp
import numpy as np
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from examples import ensemble_mesh as EM
from jax_dsmc import ensemble_backward

fs = float(sys.argv[1])
import forward as F
F.K_BOUNCE = int(sys.argv[2]) if len(sys.argv) > 2 else F.K_BOUNCE
ensemble_backward.configure("virtual", 1, 2, "staged")
cfg = O.default_config()
cfg.update(fnum_scale=fs, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
_, ls = O.initial_levelset(cfg)
dev = jax.devices()[0]
marks = []
def mark(name, *arrays):
    jax.block_until_ready([a for a in arrays if a is not None])
    st = dev.memory_stats()
    marks.append((name, st["peak_bytes_in_use"], st["bytes_in_use"]))
    print(f"{name:28s} peak {st['peak_bytes_in_use']/1e6:8.1f} MB  live {st['bytes_in_use']/1e6:8.1f} MB", flush=True)
from jax_dsmc import shape as SH
def wrap(mod, name, label):
    fn = getattr(mod, name)
    def inner(*a, **k):
        out = fn(*a, **k)
        mark(label, *jax.tree.leaves(out))
        return out
    setattr(mod, name, inner)
wrap(EM, "forward", "after forward (records)")
wrap(SH, "replay", "after replay")
wrap(EM, "device_backward", "after a device backward")
wrap(SH, "gradient_variants", "after shape gradient")
EM.shape_terms(cfg, jax.random.key(32), ls, "radius", cfg["neighbors"], cfg["timing_band"], (cfg["variant"],))
