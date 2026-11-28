#!/usr/bin/env python3
"""Environment check of a JAX env (and, in a job, of the GPUs and multi-process setup).

    python3 bench/env/check_env.py --import-only      # login node: versions only
    python3 bench/env/check_env.py                    # job: devices, x64, a tiny 3-D step
    srun -n 8 python3 bench/env/check_env.py --distributed   # multi-process + an 8-rank run

Exit status 1 if JAX cannot see an accelerator in a GPU job (JAX_PLATFORMS pins the backend,
so a node whose driver refuses to initialise fails here, not after hours on the CPU).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def visible_gpu_ids():
    """The GPUs this process may use, as JAX ordinals, or None (no GPU restriction set).

    jax.distributed.initialize() under Slurm otherwise picks local_device_ids = [local rank]:
    with `srun --gpus-per-task=1 --gpu-bind=closest` every task sees only ITS GPU as ordinal 0,
    so ranks 1-3 asked for devices 1-3 ("CUDA_ERROR_INVALID_DEVICE: invalid device ordinal",
    Perlmutter check 62133635, 2026-11-27); and one process per node would get one GPU of four.
    """
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        v = os.environ.get(var, "").strip()
        if v:
            return list(range(len(v.split(","))))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--import-only", action="store_true")
    p.add_argument("--distributed", action="store_true")
    a = p.parse_args()
    import jax

    if a.distributed:
        if os.environ.get("JAX_PLATFORMS", "") == "cpu":
            jax.config.update("jax_cpu_collectives_implementation", "gloo")
        cpu = os.environ.get("JAX_PLATFORMS", "") == "cpu"
        jax.distributed.initialize(local_device_ids=None if cpu else visible_gpu_ids())
    import jaxlib

    me = jax.process_index()
    say = print if me == 0 else (lambda *x, **k: None)
    say(f"host {socket.gethostname()} python {sys.version.split()[0]} jax {jax.__version__} jaxlib {jaxlib.__version__}")
    if a.import_only:
        say("import ok")
        return 0
    devs = jax.devices()
    say(f"backend {jax.default_backend()} processes {jax.process_count()} devices {len(devs)}: {devs[0].device_kind}")
    say(f"SLURM_GPUS_ON_NODE={os.environ.get('SLURM_GPUS_ON_NODE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
    want = os.environ.get("JAX_PLATFORMS", "")
    if want and want != "cpu" and jax.default_backend() == "cpu":
        print("no accelerator: JAX fell back to the CPU", file=sys.stderr)
        return 1
    import jax.numpy as jnp

    import sparta_jax  # noqa: F401
    from sparta_jax import decks as D
    from sparta_jax import driver

    x = jnp.ones(4, jnp.float64)
    assert x.dtype == jnp.float64, "x64 off"
    t0 = time.perf_counter()
    d = D.collide(8, 8, 8, steps=20)
    if jax.process_count() > 1 or len(devs) > 1:
        n = len(devs)
        d = D.collide(4 * n if n <= 8 else 32, 8, 8, steps=20)
        res = driver.run_deck_decomposed(d, n, seed=1)
    else:
        res = driver.run_deck(d, seed=1)
    r = res["runs"][-1]
    say(f"collide {d.x}x{d.y}x{d.z} on {res['devices']} device(s): np {r['np_end']} ncoll {r['totals']['ncoll']} "
        f"overflow {r['totals']['overflow']} compile {res['compile_s']:.1f}s total {time.perf_counter() - t0:.1f}s")
    assert r["totals"]["overflow"] == 0 and r["np_end"] == d.n_particles
    say("check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
