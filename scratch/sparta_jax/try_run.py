import sys, time, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
from sparta_jax import sim
over = {k: (int(v) if v.isdigit() else v) for k, v in (a.split("=") for a in sys.argv[1:])}
cfg, consts, geo = sim.make(**over)
st = sim.empty_state(cfg)
t0 = time.perf_counter()
st, stats = sim.run(st, 1, jnp.int32(0), consts, cfg, 1000)
jax.block_until_ready(st)
t1 = time.perf_counter()
s = sim.as_numpy_stats(stats)
for k in (9, 99, 499, 999):
    print(k + 1, {n: int(v[k]) for n, v in s.items()})
print("totals", {n: int(v.sum()) for n, v in s.items() if n != "np"}, "max", {n: int(v.max()) for n, v in s.items()}, "first", t1 - t0)
for seed in (2, 3):
    st = sim.empty_state(cfg)
    t0 = time.perf_counter(); st, stats = sim.run(st, seed, jnp.int32(0), consts, cfg, 1000); jax.block_until_ready(st)
    t = time.perf_counter() - t0
    m = sim.moves(stats)
    print(f"run seed {seed}: {t:.3f} s, np {int(stats.np[-1])}, moves {m:.4g}, {m/t:.3g} moves/s", flush=True)
