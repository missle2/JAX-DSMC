import sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax_dsmc.mesh import levelset as L
lo, m = L.grid_for_box(np.full(3, -2.0), np.full(3, 2.0), 0.1)
ls = L.from_analytic(np.zeros(3), 1.0, lo, 0.1, m, n_sample=16)
u = jax.random.normal(jax.random.PRNGKey(0), (8000, 3)); u = u/jnp.linalg.norm(u,axis=1,keepdims=True)
x = L.project(ls, u)
f1 = jax.jit(L.value_and_grad)
p1, g1 = f1(ls, x)
take = jnp.arange(0, 8000, 2)
p2, g2 = f1(ls, x[take])
p3 = jax.jit(lambda ls, x, t: L.value_and_grad(ls, x[t])[0])(ls, x, take)
p4 = jax.jit(lambda ls, x: L.value_and_grad(ls, x)[0] * 1.0)(ls, x)
p5 = jax.jit(L.value)(ls, x)
print("sub vs full", np.abs(np.asarray(p1)[::2]-np.asarray(p2)).max(), (np.sign(p1[::2])!=np.sign(p2)).sum())
print("gathered-in-jit", np.abs(np.asarray(p1)[::2]-np.asarray(p3)).max())
print("fused", np.abs(np.asarray(p1)-np.asarray(p4)).max(), "value vs vg", np.abs(np.asarray(p1)-np.asarray(p5)).max(), (np.sign(p1)!=np.sign(p5)).sum())
