"""Sort-stage microbenchmark: steady-state input (state after 600 steps, then one step's emit +
move), timed in a fori_loop whose input depends on the loop index (no hoisting).
    run_locked.sh cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 \
        python3 scratch/sparta_jax/sortopt/bench_sort.py [variant ...]"""
import functools
import sys
import time

sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
import jax
import jax.numpy as jnp
import numpy as np

from sparta_jax import sim

REP = 300


def prepared():
    cfg, consts, _ = sim.make()
    st = sim.empty_state(cfg)
    st, _ = sim.run(st, 5, jnp.int32(0), consts, cfg, 600)
    seed = jnp.asarray(5, jnp.uint64)
    st2, inserts, *_ = sim.emit(st, seed, jnp.int32(600), consts, cfg)
    moved, _surf, _mc = sim.move(st2, seed, jnp.int32(600), consts, cfg, inserts)
    return cfg, consts, moved


def timed(fn, moved, consts, cfg):
    def body(i, acc):
        s = moved._replace(cell=jnp.where(i < 0, moved.cell + 1, moved.cell))
        out = fn(s, consts, cfg)
        return acc + out.pkey[0] + out.start[-1] + out.scode[0]

    g = jax.jit(lambda: jax.lax.fori_loop(0, REP, body, jnp.int32(0)))
    jax.block_until_ready(g())
    best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        jax.block_until_ready(g())
        best = min(best, time.perf_counter() - t)
    return best / REP * 1e6


def main():
    cfg, consts, moved = prepared()
    ref = sim.sort_partition(moved, consts, cfg)
    variants = {"partition": sim.sort_partition, "full": sim.sort_full}
    import sort_variants as V  # noqa: PLC0415

    variants.update(V.VARIANTS)
    names = sys.argv[1:] or list(variants)
    nmov = int(jnp.sum(moved.cell[ref.pkey & 0xFFFF] != moved.pkey >> 16))
    print(f"alive {int(ref.start[cfg.ncell])} movers(approx) {nmov}", flush=True)
    for name in names:
        fn = variants[name]
        out = jax.jit(functools.partial(fn, cfg=cfg))(moved, consts)
        # same buckets: starts equal, and every bucket holds the same set of slots
        same_start = bool(jnp.all(out.start == ref.start))
        a = np.sort(np.asarray(out.pkey)); b = np.sort(np.asarray(ref.pkey))
        same_sets = np.array_equal(a, b)
        same_scode = bool(jnp.all(out.scode == ref.scode))
        us = timed(fn, moved, consts, cfg)
        print(f"{name:16s} {us:8.1f} us  start {same_start} buckets {same_sets} scode {same_scode}", flush=True)


if __name__ == "__main__":
    main()
