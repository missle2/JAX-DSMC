"""Scaled P3 Couette/heat check, including deterministic and regularized FD."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.p3_replication.__main__ import summary
from jax_dsmc import paper_walls as pw


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--case", choices=("couette", "heat"), default="couette")
    parser.add_argument("--score-mode", choices=("paper", "trajectory"), default="paper")
    parser.add_argument(
        "--localize", action="store_true", help="Use P3's hard three-sigma proximity cutoff"
    )
    parser.add_argument("--output", type=Path, default=Path("p3_walls.json"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("Use at least one seed")
    return args


def main():
    args = parse_args()
    cfg = pw.Config(
        particles=args.particles,
        observable=args.case,
        score_mode=args.score_mode,
        localize=args.localize,
    )
    if args.case == "heat":
        cfg = replace(cfg, dt=0.1, epsilon=0.01, initial_std=0.7)
    temp = jnp.ones((2, 3)) if args.case == "couette" else jnp.array([[0.6] * 3, [0.9] * 3])
    theta = jnp.array([4.0]) if args.case == "couette" else jnp.array([0.6, 0.9])

    def value(theta, key, cfg):
        t = temp if args.case == "couette" else jnp.repeat(theta[:, None], 3, axis=1)
        speed = theta[0] if args.case == "couette" else 0.0
        return pw.solve(t, speed, key, cfg)

    grad = jax.jit(jax.value_and_grad(value, has_aux=True), static_argnums=2)
    estimates, fds, regularized_fds, objectives = [], [], [], []
    h = 0.05 if args.case == "couette" else 0.025
    for seed in range(args.seeds):
        key = jax.random.PRNGKey(seed)
        (j, (_, invalid)), gradient = grad(theta, key, cfg)
        if int(invalid):
            raise RuntimeError(f"{int(invalid)} unresolved wall/collision events")
        estimates.append(np.asarray(gradient))
        objectives.append(float(j))
        paired = []
        for epsilon in (0.0, cfg.epsilon):
            result = []
            for i in range(len(theta)):
                plus = value(theta.at[i].add(h), key, replace(cfg, epsilon=epsilon))
                minus = value(theta.at[i].add(-h), key, replace(cfg, epsilon=epsilon))
                if int(plus[1][1]) + int(minus[1][1]):
                    raise RuntimeError("Unresolved events in finite difference")
                result.append(float((plus[0] - minus[0]) / (2 * h)))
            paired.append(result)
        fds.append(paired[0])
        regularized_fds.append(paired[1])
        print(f"seed {seed}: AD={gradient}, deterministic FD={paired[0]}", flush=True)
    result = {
        "backend": jax.default_backend(),
        "config": asdict(cfg),
        "seeds": args.seeds,
        "fd_step": h,
        "adjoint": summary(estimates),
        "deterministic_fd": summary(fds),
        "regularized_fd": summary(regularized_fds),
        "objective": summary(objectives),
        "paper_resolution_reproduced": False,
        "regularization": "paper proximity cutoff"
        if args.localize
        else "all particles; paper uses a three-sigma proximity cutoff",
    }
    if args.case == "couette":
        result.update(paper_couette_adjoint=3.0283, paper_couette_fd=3.0011)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
