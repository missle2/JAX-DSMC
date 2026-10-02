import sys, math
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np
import test_mesh_levelset as T
from jax_dsmc.mesh import levelset as L
ls = T._perturbed(2)
kx, kd, ks = jax.random.split(jax.random.PRNGKey(5), 3)
x = jax.random.uniform(kx, (20000, 2), minval=-3.0, maxval=13.0)
lower = L.block_min_table(ls, 0)[L._table_index(ls, x, 0)]
val = L.value(ls, x)
blk, w, dw = L._block(ls, x)
bmin = blk.reshape(20000, -1).min(axis=1)
bad = np.nonzero(np.asarray(val < lower))[0]
print(len(bad), ls.coef.shape, ls.lo)
for i in bad[:5]:
    print(x[i], (x[i]-ls.lo)/ls.h, float(val[i]), float(lower[i]), float(bmin[i]), w[0][i], w[1][i], float(jnp.sum(w[0][i])))
