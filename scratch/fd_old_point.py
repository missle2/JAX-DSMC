"""The old test's operating point (old basis A = 0.03) mapped to the rate basis."""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")
import test_learned_window2d as t  # noqa: E402

from examples.learned_collision import operators  # noqa: E402
from examples.learned_collision import windows2d as w2  # noqa: E402

case, px, pv, pa, target = t._shear()
fn, theta, _ = operators.operator("linear", case.kind)
a = jnp.zeros_like(theta["A"]).at[0, :-1].set(-0.03)  # r_s e^{-0.03 phi}, r_q unchanged
theta = dict(theta, A=a)
states, cstates, keys = t._replicas(case, px, pv, pa)
weights = jnp.ones(len(w2.LOSS_FIELDS[case.kind]))
bias = jax.random.normal(jax.random.PRNGKey(5), (2, 2, len(weights)))
f = jax.jit(functools.partial(t._objective, states=states, cstates=cstates, keys=keys,
                              target=jnp.asarray(target), weights=weights, bias=bias,
                              case=case, fn=fn, estimator="bias"))
g = np.asarray(jax.jit(jax.grad(f))(theta)["A"])
h = 1e-7
for i, j in ((0, 0), (1, 0), (0, 6)):
    e = jnp.zeros_like(a).at[i, j].set(1.0)
    fd = (float(f(dict(theta, A=a + h * e))) - float(f(dict(theta, A=a - h * e)))) / (2 * h)
    print(f"A[{i},{j}]: AD {g[i, j]:.6f} FD {fd:.6f} diff {g[i, j] - fd:+.2e}")
print(f"old-basis dL/dA_tau[0,0] = {-(g[0, 0] + g[1, 0]):.6f}")
