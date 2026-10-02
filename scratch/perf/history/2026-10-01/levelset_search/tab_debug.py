import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh.types import LevelSet
c = jnp.array([[5., 3, 1, 4, 7, 2, 8, 6]]*1).T * jnp.ones((1, 8))  # [8, 8]
c = jnp.asarray(np.random.default_rng(0).normal(size=(8, 8)))
ls = LevelSet(coef=c, lo=jnp.zeros(2), h=1.0)
t = L.block_min_table(ls, 0)
print(t.shape)
x = jnp.asarray(np.random.default_rng(1).uniform(-3, 11, size=(2000, 2)))
blk, w, dw = L._block(ls, x)
bmin = blk.reshape(2000, -1).min(axis=1)
lower = t[L._table_index(ls, x, 0)]
bad = np.nonzero(np.asarray(bmin) != np.asarray(lower))[0]
print(len(bad), x[bad[:3]], bmin[bad[:3]], lower[bad[:3]])
