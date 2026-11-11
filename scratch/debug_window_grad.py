"""Bisect the window-loss AD/FD mismatch: steps per block and blocks."""
import dataclasses
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")
import test_learned_window as tw
from examples.learned_collision import windows
from jax_dsmc.collide import relax

states, cstates, keys, ref, case = tw._window()


def loss(log_tau, steps, blocks):
    c = case._replace(static=dataclasses.replace(case.static, n_steps=steps))
    theta = {"log_tau": log_tau, "log_pr": jnp.float64(np.log(2 / 3))}
    return windows.window_loss(theta, states, cstates, keys, ref[:blocks], jnp.ones(2), c,
                               relax.collide, blocks=blocks)


for steps, blocks in [(s, 1) for s in range(6, 21)]:
    f = jax.jit(functools.partial(loss, steps=steps, blocks=blocks))
    t0 = jnp.float64(0.0)
    g = float(jax.grad(f)(t0))
    for h in (1e-5,):
        fd = float((f(t0 + h) - f(t0 - h)) / (2 * h))
        print(f"steps {steps:2d} blocks {blocks}: AD {g:.6e}  FD(h={h:g}) {fd:.6e}", flush=True)
