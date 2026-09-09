"""Probe E (Claim 2): naive (non-causal) DiCE vs causal DiCE on a sequential toy
with per-step rewards.  Bias and variance, as a function of horizon M.

Toy (structurally the DSMC surrogate in miniature):
    p      = sigmoid(theta)                       # theta is the design parameter
    b_k    ~ Bernoulli(p)          k = 0..M-1     # FROZEN discrete decision (like `accept`)
    x_0    = 0 ;  x_{k+1} = x_k + (2 b_k - 1)     # state carried by the scan
    r_k    = x_k                   k = 1..M       # per-step reward (drag / heat flux sample)
    J      = E[ sum_k r_k ]
Analytic:  E[x_k] = k (2p-1)  =>  J = (2p-1) M(M+1)/2
           dJ/dtheta = M(M+1) * p (1-p)
"""
import jax, jax.numpy as jnp, functools
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

def magic(L):
    return jnp.exp(L - jax.lax.stop_gradient(L))

def rollout(theta, key, M):
    """returns per-step log-probs L_k (k=0..M-1) and per-step rewards r_k (k=1..M)."""
    p = jax.nn.sigmoid(theta)
    b = (random.uniform(key, (M,)) < jax.lax.stop_gradient(p)).astype(jnp.float64)  # FROZEN draw
    L = b * jnp.log(p) + (1.0 - b) * jnp.log1p(-p)          # f64[M]
    x = jnp.cumsum(2.0 * b - 1.0)                           # x_1..x_M
    return L, x                                             # r_k = x[k]

def est_naive(theta, key, M):
    L, r = rollout(theta, key, M)
    return magic(jnp.sum(L)) * jnp.sum(r)                   # ONE magic box, spec 11 extended naively

def est_causal(theta, key, M):
    L, r = rollout(theta, key, M)
    Lpre = jnp.cumsum(L)                                    # L_{<=k}
    return jnp.sum(magic(Lpre) * r)                         # causal DiCE

g_naive  = jax.jit(jax.vmap(jax.grad(est_naive),  in_axes=(None, 0, None)), static_argnums=2)
g_causal = jax.jit(jax.vmap(jax.grad(est_causal), in_axes=(None, 0, None)), static_argnums=2)
v_naive  = jax.jit(jax.vmap(est_naive,  in_axes=(None, 0, None)), static_argnums=2)

theta = 0.4
p = 1/(1+np.exp(-theta))

print(f"theta={theta}  p={p:.6f}")
print(f"{'M':>4} {'analytic dJ/dth':>16} {'naive mean':>14} {'naive SE':>11} "
      f"{'causal mean':>14} {'causal SE':>11} {'Var ratio':>11} {'sd ratio':>9}")
S = 400_000
rows = []
for M in [2, 4, 8, 16, 32, 64, 128]:
    exact = M*(M+1) * p*(1-p)
    keys = random.split(random.key(12345), S)
    gn = np.asarray(g_naive(theta, keys, M))
    gc = np.asarray(g_causal(theta, keys, M))
    mn, mc = gn.mean(), gc.mean()
    sen, sec = gn.std(ddof=1)/np.sqrt(S), gc.std(ddof=1)/np.sqrt(S)
    vn, vc = gn.var(ddof=1), gc.var(ddof=1)
    rows.append((M, exact, mn, sen, mc, sec, vn/vc))
    print(f"{M:>4} {exact:>16.4f} {mn:>14.4f} {sen:>11.4f} {mc:>14.4f} {sec:>11.4f} "
          f"{vn/vc:>11.3f} {np.sqrt(vn/vc):>9.3f}")

print("\nbias check (z-score of  mean - analytic  in units of SE):")
for (M, exact, mn, sen, mc, sec, ratio) in rows:
    print(f"  M={M:>4}  naive z={(mn-exact)/sen:>7.2f}   causal z={(mc-exact)/sec:>7.2f}")

print("\nprimal value check (magic box must be identically 1):")
keys = random.split(random.key(7), 5)
for M in [4, 16]:
    vals = np.asarray(v_naive(theta, keys, M))
    L, r = rollout(theta, keys[0], M)
    print(f"  M={M}: surrogate values {vals[:3]}  vs raw sum(r) for seed0 = {float(jnp.sum(r))}")

print("\nvariance-ratio scaling (naive/causal), fitted exponent:")
Ms = np.array([r[0] for r in rows], float); rr = np.array([r[6] for r in rows])
sl = np.polyfit(np.log(Ms[2:]), np.log(rr[2:]), 1)[0]
print(f"  ratio ~ M^{sl:.3f}   (values: " + ", ".join(f"M={int(m)}:{v:.2f}" for m,v in zip(Ms,rr)) + ")")
