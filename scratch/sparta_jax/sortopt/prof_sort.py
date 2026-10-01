"""Per-op profile of one sort variant (host tracer level 3), ops aggregated by HLO name."""
import collections
import glob
import os
import re
import sys

sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp

import bench_sort as B
import sort_variants as V
from sparta_jax import sim
import xspace

name = sys.argv[1] if len(sys.argv) > 1 else "partition"
fn = {"partition": sim.sort_partition, **V.VARIANTS}[name]
cfg, consts, moved = B.prepared()


def body(i, acc):
    s = moved._replace(cell=jnp.where(i < 0, moved.cell + 1, moved.cell))
    out = fn(s, consts, cfg)
    return acc + out.pkey[0] + out.start[-1] + out.scode[0]


g = jax.jit(lambda: jax.lax.fori_loop(0, 100, body, jnp.int32(0)))
jax.block_until_ready(g())
out = f"/tmp/sortprof_{name}"
opts = jax.profiler.ProfileOptions()
opts.host_tracer_level = 3
opts.python_tracer_level = 0
jax.profiler.start_trace(out, profiler_options=opts)
jax.block_until_ready(g())
jax.profiler.stop_trace()
path = glob.glob(os.path.join(out, "plugins", "profile", "*", "*.xplane.pb"))[-1]
buf, planes = xspace.planes(path)
host = next(s for n, s in planes if n == "/host:CPU")
agg = collections.defaultdict(lambda: [0, 0.0])
for _l, _n, _s, dur, stats in xspace.plane_events(buf, host, event_stats=True):
    op = stats.get("hlo_op")
    if op is None or op.startswith(("while", "conditional", "call")):
        continue
    k = re.sub(r"\.\d+$", "", op)
    agg[k][0] += 1
    agg[k][1] += dur / 1e3 / 100
tot = sum(v[1] for v in agg.values())
print(f"{name}: {tot:.1f} us/iter of leaf ops")
for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"  {t:7.1f} us  x{n/100:5.2f}  {k}")
