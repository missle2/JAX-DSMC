"""CLAIM 1 probes: safe_sqrt gradient magnitude, composite chain-rule behaviour,
harm demonstration, EPS_D dimensionality."""

import jax, jax.numpy as jnp, numpy as np

jax.config.update("jax_enable_x64", True)

EPS = 1e-30
EPS_D = 1e-6


def safe_sqrt(x, eps=EPS):
    ok = x > eps
    x_safe = jnp.where(ok, x, 1.0)
    return jnp.where(ok, jnp.sqrt(x_safe), 0.0)


def safe_div(num, den, eps=EPS, fallback=0.0):
    ok = jnp.abs(den) > eps
    den_safe = jnp.where(ok, den, 1.0)
    return jnp.where(ok, num / den_safe, fallback)


def safe_norm(v, eps=EPS):
    return safe_sqrt(jnp.sum(v * v), eps)


def safe_pow(x, p, eps=EPS):
    ok = x > eps
    x_safe = jnp.where(ok, x, 1.0)
    return jnp.where(ok, jnp.exp(p * jnp.log(x_safe)), 0.0)


print("=" * 78)
print("1A. d/dx safe_sqrt(x) as x -> eps+  (eps = 1e-30)")
print("=" * 78)
g = jax.grad(lambda x: safe_sqrt(x))
print(f"{'x':>12} {'safe_sqrt':>14} {'d/dx':>14} {'analytic 1/(2sqrt x)':>22}")
for e in [-30, -29, -28, -26, -24, -20, -16, -12, -8, -4, 0]:
    x = 10.0**e
    gg = float(g(x))
    print(f"{x:12.1e} {float(safe_sqrt(x)):14.4e} {gg:14.4e} {1 / (2 * np.sqrt(x)):22.4e}")
# exactly at / below clamp
for x in [1e-30, 1.0000001e-30, 9.9e-31, 0.0]:
    print(f"  x={x:.6e}: val={float(safe_sqrt(x)):.4e} grad={float(g(x)):.4e}")
print()
print("second derivative at just-above-clamp:")
g2 = jax.grad(g)
for e in [-30, -28, -24, -16, -8, 0]:
    x = 10.0**e
    print(f"  x={x:.1e}  d2/dx2 = {float(g2(x)):.4e}   (analytic {-0.25 * x**-1.5:.4e})")

print()
print("=" * 78)
print("1A-bis. THE COMPOSITE: d/dv of safe_norm(v) = safe_sqrt(sum v*v)")
print("  (this is how safe_sqrt is actually reached in the spec)")
print("=" * 78)
gn = jax.grad(lambda v: safe_norm(v))
print(f"{'|v|':>12} {'sum v*v':>12} {'||grad_v||':>14}")
for e in [-16, -14, -12, -10, -8, -4, 0, 2]:
    s = 10.0**e
    v = jnp.array([s, 0.0, 0.0])
    gv = gn(v)
    print(f"{s:12.1e} {float(jnp.sum(v * v)):12.1e} {float(jnp.linalg.norm(gv)):14.6e}")
v = jnp.array([1e-16, 0.0, 0.0])  # sum v*v = 1e-32 < eps -> clamped branch
print("  below clamp |v|=1e-16 (sum=1e-32 < eps):", np.asarray(gn(v)))

print()
print("=" * 78)
print("1A-ter. d/dv of q = safe_pow(sum v*v, 1-omega)  (the NTC cross-section)")
print("=" * 78)
for omega in [0.5, 0.75, 1.0, 1.2]:
    f = lambda v: safe_pow(jnp.sum(v * v), 1.0 - omega)
    gq = jax.grad(f)
    row = []
    for e in [-15, -12, -8, -4, 0, 2]:
        s = 10.0**e
        v = jnp.array([s, 0.0, 0.0])
        row.append((s, float(f(v)), float(jnp.linalg.norm(gq(v)))))
    print(f"omega={omega}")
    for s, val, gm in row:
        print(f"   |vr|={s:8.1e}  q={val:12.4e}  ||dq/dvr||={gm:12.4e}")
