"""Old (sj_base) vs new (sparta_jax) sim: same seed, same steps -> compare stats and final state."""
import sys
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax/acct")
import sj_base.sim as old  # noqa: E402
from sparta_jax import sim as new  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
res = []
for mod in (old, new):
    cfg, consts, _ = mod.make()
    st = mod.empty_state(cfg)
    st, stats = mod.run(st, 5, jnp.int32(0), consts, cfg, steps)
    res.append((st, mod.as_numpy_stats(stats), cfg))
(a, sa, ca), (b, sb, cb) = res
for k in sa:
    if k in sb and k not in ("nwalk",):
        bad = np.nonzero(sa[k] != sb[k])[0]
        if len(bad):
            print(k, "first diff at step", bad[0], sa[k][bad[0]], sb[k][bad[0]])
        d = np.abs(sa[k].astype(np.int64) - sb[k].astype(np.int64))
        print(f"{k:10s} sum old {sa[k].sum():10d} new {sb[k].sum():10d} max|diff| {d.max()}")
def snap(st, cfg):
    c = np.asarray(st.cell); al = c < cfg.ncell
    key = np.lexsort((np.asarray(st.y)[al], np.asarray(st.x)[al]))
    return {f: np.asarray(getattr(st, f))[al][key] for f in ("x", "y", "vx", "vy", "vz", "cell")}
A, B = snap(a, ca), snap(b, cb)
print("alive", len(A["x"]), len(B["x"]))
if len(A["x"]) == len(B["x"]):
    for f in A:
        print(f, "max|diff|", np.max(np.abs(A[f] - B[f])))
