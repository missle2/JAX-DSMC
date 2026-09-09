"""Probe G (Claim 2, part 2): the DSMC-realistic case -- a STATIONARY per-step
reward over a sampling window (time-averaged drag / wall heat flux), where the
influence of a decision decays with lag.  Does the naive/causal variance ratio
grow with the window length?

    p = sigmoid(theta);  b_k ~ Bern(p);  eps_k = 2 b_k - 1
    x_{k+1} = alpha x_k + eps_k        (mean-reverting, alpha < 1)
    r_k = x_k                          per-step reward over the window
    E[x_k] -> (2p-1)/(1-alpha) in steady state
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

ALPHA = 0.8

def magic(L): return jnp.exp(L - jax.lax.stop_gradient(L))

def rollout(theta, key, M):
    p = jax.nn.sigmoid(theta)
    b = (random.uniform(key, (M,)) < jax.lax.stop_gradient(p)).astype(jnp.float64)
    L = b*jnp.log(p) + (1-b)*jnp.log1p(-p)
    eps = 2*b - 1
    def st(x, e): x = ALPHA*x + e; return x, x
    _, x = jax.lax.scan(st, 0.0, eps)
    return L, x

def naive(theta, key, M):
    L, r = rollout(theta, key, M); return magic(jnp.sum(L))*jnp.sum(r)
def causal(theta, key, M):
    L, r = rollout(theta, key, M); return jnp.sum(magic(jnp.cumsum(L))*r)

gn = jax.jit(jax.vmap(jax.grad(naive),  in_axes=(None,0,None)), static_argnums=2)
gc = jax.jit(jax.vmap(jax.grad(causal), in_axes=(None,0,None)), static_argnums=2)

theta = 0.4; p = 1/(1+np.exp(-theta)); S = 400_000
# analytic: E[sum_{k=1..M} x_k], x_k = sum_{m<k} alpha^(k-1-m) eps_m
# E[x_k] = (2p-1) * (1-alpha^k)/(1-alpha);  dE/dtheta = 2 p(1-p) * (1-alpha^k)/(1-alpha)
print(f"{'M':>5} {'analytic':>12} {'naive mean':>12} {'naive SE':>10} {'causal mean':>12} "
      f"{'causal SE':>10} {'Var ratio':>10}")
Ms = [4, 8, 16, 32, 64, 128, 256, 512]
rat = []
for M in Ms:
    ks = np.arange(1, M+1)
    exact = 2*p*(1-p) * np.sum((1-ALPHA**ks)/(1-ALPHA))
    keys = random.split(random.key(999), S)
    a = np.asarray(gn(theta, keys, M)); c = np.asarray(gc(theta, keys, M))
    sea, sec = a.std(ddof=1)/np.sqrt(S), c.std(ddof=1)/np.sqrt(S)
    r = a.var(ddof=1)/c.var(ddof=1); rat.append(r)
    print(f"{M:>5} {exact:>12.4f} {a.mean():>12.4f} {sea:>10.4f} {c.mean():>12.4f} "
          f"{sec:>10.4f} {r:>10.3f}   z_naive={(a.mean()-exact)/sea:+.2f} z_causal={(c.mean()-exact)/sec:+.2f}")
sl = np.polyfit(np.log(Ms[2:]), np.log(rat[2:]), 1)[0]
print(f"\nvariance ratio ~ M^{sl:.3f}")
print("ratios:", ", ".join(f"M={m}:{v:.2f}" for m,v in zip(Ms,rat)))
