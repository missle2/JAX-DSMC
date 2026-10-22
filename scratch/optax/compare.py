"""Do our optimiser pieces equal optax's (2026-10-21)?  Synthetic gradients, CPU, seconds.

* Adam: ``optimize.adam_direction`` (eps 1e-30, no eps in the root) against
  ``optax.scale_by_adam(b1, b2, eps=1e-30, eps_root=0)`` over 50 noisy gradient steps.
* L-BFGS: ``optimizers.lbfgs_direction`` with the identity as preconditioner against
  ``optax.scale_by_lbfgs`` (same memory) on a quadratic with exact line search.

    PYTHONPATH=. python3 scratch/optax/compare.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402
import optimizers as OZ  # noqa: E402

jax.config.update("jax_enable_x64", True)


def adam_check(n=500, steps=50, b1=0.9, b2=0.999):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n) * 1e-6  # the drag gradient's scale (per coefficient)
    state = (np.zeros(n), np.zeros(n))
    tx = optax.scale_by_adam(b1=b1, b2=b2, eps=1e-30, eps_root=0.0)
    ost = tx.init(jnp.zeros(n))
    worst = 0.0
    for k in range(steps):
        g = signal + rng.normal(size=n) * 2e-6
        ours, state = O.adam_direction(g, state, k, b1, b2)
        theirs, ost = tx.update(jnp.asarray(g), ost)
        worst = max(worst, float(np.max(np.abs(ours - np.asarray(theirs)) / np.abs(ours).max())))
    return worst


def lbfgs_check(n=12, iters=15, memory=OZ.LBFGS_MEMORY):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(n, n))
    hess = a @ a.T + n * np.eye(n)
    x_star = rng.normal(size=n)

    def grad(x):
        return hess @ (x - x_star)

    tx = optax.scale_by_lbfgs(memory_size=memory)
    x = np.zeros(n)
    pairs, ost = [], tx.init(jnp.zeros(n))
    g = grad(x)
    worst = 0.0
    for _ in range(iters):
        d_ours, _ = OZ.lbfgs_direction(g, pairs, lambda v: v)
        upd, ost = tx.update(jnp.asarray(g), ost, jnp.asarray(x))
        d_theirs = -np.asarray(upd)
        if pairs:  # optax's first step is the scaled gradient too, but scaled differently
            worst = max(worst, float(np.linalg.norm(d_ours - d_theirs) / np.linalg.norm(d_ours)))
        t = -float(g @ d_ours) / float(d_ours @ hess @ d_ours)
        x_new = x + t * d_ours
        g_new = grad(x_new)
        pairs, _ = OZ.add_pair(pairs, x_new - x, g_new - g)
        # optax records the pair from the parameters / gradients it is given next call: feed
        # it the same step so both memories hold identical (s, y)
        x, g = x_new, g_new
        if np.linalg.norm(g) < 1e-10:
            break
    return worst


if __name__ == "__main__":
    print(f"Adam: max relative difference over 50 steps: {adam_check():.2e}")
    print(f"L-BFGS: max relative direction difference: {lbfgs_check():.2e}")
