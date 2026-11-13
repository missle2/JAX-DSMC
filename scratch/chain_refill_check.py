"""Refresh paths of the shock chains: clones while half are healthy, the pool otherwise."""

import sys

import numpy as np

from examples.learned_collision import train

args = train.parser().parse_args(sys.argv[1:])
theta = train.operators.operator("linear", "shock", **train.model_options(args))[1]
r = train.CaseRunner("K3", "linear", args)
rng = np.random.default_rng(0)
r.drift = 1e-9  # every chain off: the pool refill
r.windows(theta, rng, "pair", 2)
print("all off -> refreshed", r.refreshed, "refilled", r.refilled)
r.drift = 1e9
r.windows(theta, rng, "pair", 2)
print("none off -> refreshed", r.refreshed, "refilled", r.refilled)
