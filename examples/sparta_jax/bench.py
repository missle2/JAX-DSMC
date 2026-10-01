"""Throughput of sparta_jax on SPARTA's circle deck: 1000 steps from an empty box, one lax.scan.

Moves are counted as SPARTA counts "Particle moves" (particles present at each move, inserts
included).  The first call compiles; the timed calls repeat with other seeds, best of --repeats.

    run_locked.sh gpu python3 examples/sparta_jax/bench.py --tag gpu
    run_locked.sh cpu python3 examples/sparta_jax/bench.py --tag cpu_all
    run_locked.sh cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 \
        python3 examples/sparta_jax/bench.py --tag cpu_1
Options: --set key=value ... (sim.Config overrides, e.g. sort=full tally=1)
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from sparta_jax import sim  # noqa: E402


def parse(kv):
    k, v = kv.split("=")
    if v.lstrip("-").isdigit():
        v = int(v)
    if k == "tally":
        v = bool(v)
    return k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    over = dict(parse(kv) for kv in args.set)
    cfg, consts, _geo = sim.make(**over)
    tally_from = jnp.int32(700) if cfg.tally else None
    rows = []
    for rep in range(args.repeats + 1):
        st = sim.empty_state(cfg)
        jax.block_until_ready(st)
        t0 = time.perf_counter()
        st, stats = sim.run(st, 11 + rep, jnp.int32(0), consts, cfg, args.steps, tally_from)
        jax.block_until_ready((st, stats))
        t = time.perf_counter() - t0
        rows.append({"seconds": t, "moves": sim.moves(stats), "np": int(stats.np[-1])})
        if rep == 0:  # work counts over the second half of the run
            half = {k: v[args.steps // 2 :] for k, v in sim.as_numpy_stats(stats).items()}
            work = {
                f"{k}_{f}": float(getattr(half[k], f)())
                for k in ("np", "nlane", "nwalk", "nscheck", "nscoll")
                for f in ("mean", "max")
            }
        print(f"rep {rep}: {t:.3f} s  moves {rows[-1]['moves']}  np {rows[-1]['np']}", flush=True)
    timed = rows[1:]
    best = min(timed, key=lambda r: r["seconds"])
    out = {
        "tag": args.tag,
        "platform": jax.devices()[0].platform,
        "device": jax.devices()[0].device_kind,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "config": dict(over),
        "sort": cfg.sort,
        "tally": cfg.tally,
        "seconds_first_incl_compile": rows[0]["seconds"],
        "seconds": best["seconds"],
        "seconds_all": [r["seconds"] for r in timed],
        "moves": best["moves"],
        "moves_per_second": best["moves"] / best["seconds"],
        "final_particles": best["np"],
        "work_second_half": work,
    }
    print(json.dumps(out, indent=1))
    if args.tag:
        with open(os.path.join(HERE, "results", f"bench_{args.tag}.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
