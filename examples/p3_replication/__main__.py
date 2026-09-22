"""Run a scaled P3 inflow comparison; outputs a reviewable JSON artifact."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_inflow_jax as ji


def summary(values):
    values = np.asarray(values)
    return {
        "mean": values.mean(axis=0).tolist(),
        "se": (values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist()
        if len(values) > 1
        else None,
    }


def main():  # noqa: PLR0915 -- one CLI driver, both backends
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=("host", "jax"), default="host")
    parser.add_argument("--cells", type=int, default=40)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--reconstruction", choices=("affine", "ballistic"), default="affine")
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--retained", type=int, default=128)
    parser.add_argument("--source-samples", type=int, default=512)
    parser.add_argument("--partners", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--fd-step", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=Path("p3_inflow.json"))
    args = parser.parse_args()
    if args.seeds < 1 or not 0 < args.fd_step < 4:
        parser.error("Use at least one seed and a positive FD step below 4")
    cfg = ei.Config(
        n_initial=args.particles,
        cells=args.cells,
        steps=args.steps,
        retained=args.retained,
        source_samples=args.source_samples,
        partners=args.partners,
        neighbors=args.neighbors,
        reconstruction=args.reconstruction,
    )
    temperatures = np.full((2, 3), 4.0)
    counts = ei.nominal_counts(cfg, temperatures)
    backend = "numpy/scipy host"
    if args.backend == "jax":
        capacity = args.capacity or (cfg.n_initial + cfg.steps * sum(counts))
        compiled_forward = jax.jit(lambda t, key: ji.forward(cfg, t, key, counts, capacity))
        compiled_backward = jax.jit(lambda tr, key: ji.backward(tr, cfg, key))

        def forward(t, seed):
            tr = compiled_forward(t, jax.random.key(seed))
            if bool(tr.invalid):
                raise RuntimeError("Invalid JAX trace: capacity, disjoint count, or temperature")
            return tr

        def backward(tr, seed):
            return np.asarray(compiled_backward(tr, jax.random.key(seed)))

        backend = "jax " + jax.default_backend()
    else:

        def forward(t, seed):
            return ei.forward(cfg, t, seed=seed, injection_counts=counts)

        def backward(tr, seed):
            return ei.backward(tr, seed=seed)

    gradients, differences, objectives, populations = [], [], [], []
    start = time.monotonic()
    for seed in range(args.seeds):
        trace = forward(temperatures, seed)
        gradients.append(backward(trace, 10000 + seed)[:, 0])
        objectives.append(float(trace.objective))
        populations.append(
            int(
                trace.frames.counts[-1].sum()
                if args.backend == "jax"
                else trace.frames[-1].counts.sum()
            )
        )
        fd = []
        for side in (0, 1):
            plus, minus = temperatures.copy(), temperatures.copy()
            plus[side, 0] += args.fd_step
            minus[side, 0] -= args.fd_step
            hi = forward(plus, seed)
            lo = forward(minus, seed)
            fd.append(float((hi.objective - lo.objective) / (2 * args.fd_step)))
        differences.append(fd)
        print(f"seed {seed}: adjoint={gradients[-1]}, FD={fd}", flush=True)
    result = {
        "backend": backend,
        "config": asdict(cfg),
        "seeds": args.seeds,
        "fd_step": args.fd_step,
        "injection_counts": counts,
        "normalization": int(trace.normalization),
        "adjoint_normal_temperature": summary(gradients),
        "finite_difference_normal_temperature": summary(differences),
        "paired_difference": summary(np.asarray(gradients) - differences),
        "objective": summary(objectives),
        "final_population": summary(populations),
        "paper_adjoint_mean": [0.15171883, 0.15157001],
        "paper_resolution_reproduced": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
