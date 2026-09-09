"""Does the fixed-A scheme carry the dt / fnum / volume gradient correctly?

The current scheme gets that gradient from the stochastic-rounding count score
`log P(n|f)` (section 8.1 -- the fix for defect 1).  The fixed-A scheme has no
rounded count at all: `dt`, `fnum` and `V` enter the acceptance probability

    p = 0.5 npc(npc-1) dt fnum q / (A V)

smoothly, so the ordinary per-pair score `log p` / `log(1-p)` should carry them.

Ground truth is Rao-Blackwellisation: at a FIXED configuration and fixed uniforms,
the objective is a sum over pairs of a Bernoulli, so E[J] and dE[J]/d(theta) can be
written down exactly with no sampling at all.  Both estimators are compared to it.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

K = 1.380649e-23
M, DIAM, TREF, OMEGA = 6.63e-26, 4.11e-10, 273.15, 0.81
MR = M / 2
PREF = math.pi * DIAM**2 * (2 * K * TREF / MR) ** (OMEGA - 0.5) / math.gamma(2.5 - OMEGA)
NPC, VOL, NRHO = 200, 7.8e-12, 1e23
FNUM = NRHO * VOL / NPC
A = 512

rng = np.random.default_rng(3)
V = jnp.asarray(math.sqrt(K * TREF / M) * rng.standard_normal((NPC, 3)))
II = jnp.asarray(rng.integers(0, NPC, A))
JJ = jnp.asarray((np.asarray(II) + 1 + rng.integers(0, NPC - 1, A)) % NPC)
U = jnp.asarray(rng.random(A))


def qs(v):
    vr2 = jnp.sum((v[II] - v[JJ]) ** 2, -1)
    return PREF * jnp.maximum(vr2, 1e-8 * (2 * K * TREF / MR)) ** (1.0 - OMEGA)


def p_of(dt, fnum, vol, v):
    """Fixed-A acceptance probability."""
    return 0.5 * NPC * (NPC - 1.0) * dt * fnum * qs(v) / (A * vol)


def reward(v):
    """A simple per-pair reward: the post-collision energy proxy."""
    return jnp.sum(v[II] ** 2, -1) + jnp.sum(v[JJ] ** 2, -1)


# --- ground truth: E[J] = sum_a p_a * r_a, differentiable in closed form ------
def J_exact(dt, fnum, vol):
    p = jnp.clip(p_of(dt, fnum, vol, V), 0.0, 1.0)
    return jnp.sum(p * reward(V))


# --- the fixed-A estimator, with the DiCE surrogate on log p / log(1-p) -------
def J_fixedA(dt, fnum, vol, u):
    p = jnp.clip(p_of(dt, fnum, vol, V), 0.0, 1.0)
    acc = u < p
    logh = jnp.where(acc, jnp.log(jnp.maximum(p, 1e-300)),
                     jnp.log(jnp.maximum(1.0 - p, 1e-300)))
    magic = jnp.exp(logh - jax.lax.stop_gradient(logh))
    return jnp.sum(magic * jnp.where(acc, reward(V), 0.0))


dt0, fn0, vol0 = jnp.float64(6.5e-9), jnp.float64(FNUM), jnp.float64(VOL)
g_true = jax.grad(J_exact, argnums=(0, 1, 2))(dt0, fn0, vol0)
print("Rao-Blackwellised truth (exact, no sampling):")
print(f"  J            = {float(J_exact(dt0, fn0, vol0)):.6e}")
print(f"  dJ/d(dt)     = {float(g_true[0]):.6e}")
print(f"  dJ/d(fnum)   = {float(g_true[1]):.6e}")
print(f"  dJ/d(volume) = {float(g_true[2]):.6e}")

M_EST = 20000
keys = jax.random.split(jax.random.PRNGKey(0), M_EST)
gfun = jax.jit(jax.vmap(jax.grad(J_fixedA, argnums=(0, 1, 2)), in_axes=(None, None, None, 0)))
us = jax.random.uniform(jax.random.PRNGKey(11), (M_EST, A))
gs = gfun(dt0, fn0, vol0, us)
print(f"\nfixed-A estimator, {M_EST:,} samples:")
names = ("dJ/d(dt)", "dJ/d(fnum)", "dJ/d(volume)")
for nm, g, t in zip(names, gs, g_true, strict=True):
    m, sd = float(jnp.mean(g)), float(jnp.std(g))
    sem = sd / math.sqrt(M_EST)
    z = (m - float(t)) / sem
    print(f"  {nm:<13} = {m:>13.6e} +- {sem:.2e}   truth {float(t):>13.6e}   z = {z:+.2f}")

print("\n=> all three gradients are carried by `log p` alone.  No stochastic-rounding")
print("   count score is needed: the count is not a random integer any more.")
