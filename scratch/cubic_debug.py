import sys
sys.path.insert(0, "tests")
import numpy as np
import test_relax_cubic as t
from jax_dsmc.collide import relax
from jax_dsmc import sort
box = t._heat_box()
state, cs, grid, sp, params, static, key = box
table = sort.build_table(state, grid)
seg, row, n, u, u_loc, pi, theta_c = relax.local_state(state, table, grid, (True,))
_, n_s, _, th = relax.safe_cells(n, pi, theta_c)
phi = relax.cell_features(n, n_s, u, th, grid, sp, params, (True,))
d = relax.correction(t._theta(0.3), phi, "linear")
print("phi", np.asarray(phi)[0]); print("delta", np.asarray(d)[:, 0])
print("phi full", np.asarray(phi).mean(0))
import jax.numpy as jnp
for beta in (0.0, 0.05, 0.1, -0.1):
    th = t._theta(0.0); th["A"] = th["A"].at[2, 0].set(beta / float(np.asarray(phi)[:, 0].mean()))
    out = t._step(th, box)
    print(beta, "q,pxx", t._moments(out), "delta2", np.asarray(relax.correction(th, phi, "linear"))[2, :2])
print("start q,pxx", t._moments(box[0]))
