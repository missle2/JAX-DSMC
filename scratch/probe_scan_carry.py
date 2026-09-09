"""Probe I: causal DiCE implemented as ONE extra scalar in the 12 scan carry,
using the running logh_acc the state already holds.  Must match the explicit form."""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

Npart, M = 6, 12
def magic(L): return jnp.exp(L - jax.lax.stop_gradient(L))

def step(theta, key, x, logh):
    """toy 'collide': per-particle frozen Bernoulli accept, per-particle log h."""
    p = jax.nn.sigmoid(theta + 0.1*x)
    acc = (random.uniform(key, (Npart,)) < jax.lax.stop_gradient(p)).astype(jnp.float64)
    lh  = acc*jnp.log(p) + (1-acc)*jnp.log1p(-p)
    x   = 0.9*x + (2*acc - 1) + 0.05*p          # pathwise dependence on theta too
    return x, logh + lh

phi = lambda x: x**2                             # per-particle reward density
RHO_OVER_N = 0.3

def terminal_spec(theta, key):                   # spec 11 as written
    x = jnp.zeros(Npart); logh = jnp.zeros(Npart)
    for k in range(M):
        x, logh = step(theta, random.fold_in(key, k), x, logh)
    return RHO_OVER_N*jnp.sum(magic(logh)*phi(x))

def naive_windowed(theta, key):                  # one magic box x summed rewards
    x = jnp.zeros(Npart); logh = jnp.zeros(Npart); R = jnp.zeros(Npart)
    for k in range(M):
        x, logh = step(theta, random.fold_in(key, k), x, logh)
        R = R + phi(x)
    return RHO_OVER_N*jnp.sum(magic(logh)*R)

def causal_explicit(theta, key):                 # sum_k magic(L_{<=k}) r_k, written out
    x = jnp.zeros(Npart); logh = jnp.zeros(Npart); J = 0.0
    for k in range(M):
        x, logh = step(theta, random.fold_in(key, k), x, logh)
        J = J + RHO_OVER_N*jnp.sum(magic(logh)*phi(x))
    return J

def causal_scan_carry(theta, key):               # SAME, as a lax.scan with a scalar carry
    def body(carry, k):
        x, logh, J = carry
        x, logh = step(theta, random.fold_in(key, k), x, logh)
        J = J + RHO_OVER_N*jnp.sum(magic(logh)*phi(x))   # <-- one extra scalar in the carry
        return (x, logh, J), None
    (x, logh, J), _ = jax.lax.scan(body, (jnp.zeros(Npart), jnp.zeros(Npart), 0.0), jnp.arange(M))
    return J

key = random.key(3)
for name, f in [("terminal (spec 11)", terminal_spec), ("naive windowed", naive_windowed),
                ("causal explicit", causal_explicit), ("causal scan-carry", causal_scan_carry)]:
    v, g = jax.value_and_grad(f)(0.3, key)
    print(f"  {name:>20}: value={float(v):+.10f}  grad={float(g):+.10f}")

# primal values must equal the plain (non-magic) objective
def plain_window(theta, key):
    x = jnp.zeros(Npart); logh = jnp.zeros(Npart); J = 0.0
    for k in range(M):
        x, logh = step(theta, random.fold_in(key, k), x, logh)
        J = J + RHO_OVER_N*jnp.sum(phi(x))
    return J
print(f"\n  plain sum_k r_k (no magic)   : {float(plain_window(0.3, key)):+.10f}")

ge = jax.grad(causal_explicit)(0.3, key); gs = jax.grad(causal_scan_carry)(0.3, key)
print(f"  explicit vs scan-carry grad  : {float(ge):.12f} vs {float(gs):.12f}  identical={bool(jnp.allclose(ge,gs,rtol=0,atol=0))}")
