import sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scratch/ls")
import levelset_old as OLD
from jax_dsmc.mesh import levelset as L
lo, m = L.grid_for_box(np.full(3, -2.0), np.full(3, 2.0), 0.1)
ls = L.from_analytic(np.zeros(3), 1.0, lo, 0.1, m, n_sample=16)
key = jax.random.PRNGKey(hash((3, "surface")) % 1000)
k1, k2, k3, k4 = jax.random.split(key, 4)
n = 8000
u = jax.random.normal(k1, (n, 3)); u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
v = jax.random.normal(k2, (n, 3)) * 400.0
x = L.project(ls, u)
a = np.asarray(L.hit_body(x, v, ls, None)); b = np.asarray(OLD.hit_body(x, v, ls, None))
idx = np.nonzero(np.isfinite(a) != np.isfinite(b))[0]
print(len(idx))
phi0, g0 = L.value_and_grad(ls, x)
vh = v / jnp.linalg.norm(v, axis=1, keepdims=True)
for i in idx[:8]:
    print(i, a[i], b[i], float(phi0[i]), float(jnp.dot(vh[i], g0[i])))
# also per-ray single evaluation
for i in idx[:4]:
    print("single", L.hit_body(x[i:i+1], v[i:i+1], ls, None), OLD.hit_body(x[i:i+1], v[i:i+1], ls, None))
