import sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scratch/ls")
import levelset_old as OLD
from jax_dsmc.mesh import levelset as L
lo, m = L.grid_for_box(np.zeros(2), np.full(2, 10.0), 0.0625)
ls = L.from_analytic(np.array([5.0, 5.0]), 3.0, lo, 0.0625, m, n_sample=16)
c = np.array([5.0, 5.0]); r = 3.0
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
n = 20000
u = jax.random.normal(k1, (n, 2)); u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
v = (jax.random.normal(k2, (n, 3)) * 400.0).at[:, 2].set(0.0)
rad = r + jax.random.uniform(k3, (n,), minval=-0.05, maxval=0.6)
x = c + rad[:, None] * u
a = np.asarray(jax.jit(L.hit_body)(x, v, ls, None)); b = np.asarray(jax.jit(OLD.hit_body)(x, v, ls, None))
both = np.isfinite(a) & np.isfinite(b) & (a > 0)
sp = np.linalg.norm(np.asarray(v)[:, :2], axis=1)
vh = np.asarray(v)[:, :2] / sp[:, None]
sa, sb = a * sp, b * sp
pa, ga = L.value_and_grad(ls, jnp.asarray(np.asarray(x) + sa[:, None] * vh))
pb, gb = L.value_and_grad(ls, jnp.asarray(np.asarray(x) + sb[:, None] * vh))
ea = np.abs(np.asarray(pa) / np.sum(np.asarray(ga) * vh, 1))
eb = np.abs(np.asarray(pb) / np.sum(np.asarray(gb) * vh, 1))
d = np.abs(sa - sb)
i = np.argsort(-d[both])[:8]
idx = np.nonzero(both)[0][i]
for j in idx:
    print(f"s {sa[j]:.6f} ds {d[j]:.2e} est.err new {ea[j]:.2e} old {eb[j]:.2e} slope {float(np.sum(np.asarray(ga[j])*vh[j])):.3f}")
print("median est err new", np.median(ea[both]), "old", np.median(eb[both]), "max", ea[both].max(), eb[both].max())
