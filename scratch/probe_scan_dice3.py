"""Probe H: does the naive/causal variance ratio SATURATE or GROW with horizon?
Stationary reward (mean-reverting state), M up to 4096, chunked to fit GPU memory.
Also: does an opt-in baseline (spec 11.4) change the picture?"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

ALPHA = 0.8
def magic(L): return jnp.exp(L - jax.lax.stop_gradient(L))

def make(M, baseline):
    def rollout(theta, key):
        p = jax.nn.sigmoid(theta)
        b = (random.uniform(key, (M,)) < jax.lax.stop_gradient(p)).astype(jnp.float64)
        L = b*jnp.log(p) + (1-b)*jnp.log1p(-p)
        _, x = jax.lax.scan(lambda s,e: (ALPHA*s+e,)*2, 0.0, 2*b-1)
        return L, x - baseline
    def naive(theta, key):
        L, r = rollout(theta, key); return magic(jnp.sum(L))*jnp.sum(r)
    def causal(theta, key):
        L, r = rollout(theta, key); return jnp.sum(magic(jnp.cumsum(L))*r)
    return (jax.jit(jax.vmap(jax.grad(naive),  in_axes=(None,0))),
            jax.jit(jax.vmap(jax.grad(causal), in_axes=(None,0))))

def stats(gf, theta, S, M, chunk):
    tot=n=0; s1=s2=0.0
    seed = 0
    while n < S:
        c = min(chunk, S-n)
        keys = random.split(random.key(4242+seed), c); seed += 1
        g = np.asarray(gf(theta, keys))
        s1 += g.sum(); s2 += (g**2).sum(); n += c
    m = s1/n; v = (s2 - n*m*m)/(n-1)
    return m, v, np.sqrt(v/n), n

theta = 0.4; p = 1/(1+np.exp(-theta))
for baseline_on in (False, True):
    mu = (2*p-1)/(1-ALPHA)          # steady-state mean reward
    b = mu if baseline_on else 0.0
    print(f"\n===== baseline {'ON (b = steady-state mean = %.4f)'%mu if baseline_on else 'OFF (spec 11.4 default)'} =====")
    print(f"{'M':>6} {'analytic':>12} {'naive mean':>12} {'causal mean':>12} {'Var ratio':>10} {'SE ratio':>9}")
    Ms = [8, 32, 128, 512, 1024, 2048, 4096]
    rat = []
    for M in Ms:
        ks = np.arange(1, M+1)
        exact = 2*p*(1-p)*np.sum((1-ALPHA**ks)/(1-ALPHA))   # unaffected by a constant baseline
        S = 400_000 if M <= 128 else (200_000 if M <= 512 else 60_000)
        chunk = max(1, min(S, int(3e7 // M)))
        gn, gc = make(M, b)
        mn, vn, sen, _ = stats(gn, theta, S, M, chunk)
        mc, vc, sec, _ = stats(gc, theta, S, M, chunk)
        rat.append(vn/vc)
        print(f"{M:>6} {exact:>12.3f} {mn:>12.3f} {mc:>12.3f} {vn/vc:>10.3f} {np.sqrt(vn/vc):>9.3f}"
              f"   z_n={(mn-exact)/sen:+6.2f} z_c={(mc-exact)/sec:+6.2f}  S={S}")
    sl = np.polyfit(np.log(Ms[-4:]), np.log(rat[-4:]), 1)[0]
    print(f"  tail scaling: ratio ~ M^{sl:+.3f}   (saturating if ~0)")
