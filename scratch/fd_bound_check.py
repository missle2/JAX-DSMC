"""Which change breaks the linear shear window AD = FD check: the bound feature or the basis?"""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")
import test_learned_window2d as t  # noqa: E402

from examples.learned_collision import operators  # noqa: E402
from examples.learned_collision import windows2d as w2  # noqa: E402


def check(a_fill, zero_bound):
    case, px, pv, pa, target = t._shear()
    fn, theta, _ = operators.operator("linear", case.kind)
    a = jnp.full_like(theta["A"], a_fill)
    if zero_bound:
        a = a.at[:, -1].set(0.0)
    theta = dict(theta, A=a)
    states, cstates, keys = t._replicas(case, px, pv, pa)
    weights = jnp.ones(len(w2.LOSS_FIELDS[case.kind]))
    bias = jax.random.normal(jax.random.PRNGKey(5), (2, 2, len(weights)))
    f = jax.jit(functools.partial(t._objective, states=states, cstates=cstates, keys=keys,
                                  target=jnp.asarray(target), weights=weights, bias=bias,
                                  case=case, fn=fn, estimator="bias"))
    g = float(np.asarray(jax.jit(jax.grad(f))(theta)["A"]).ravel()[0])
    e = (jnp.arange(a.size) == 0).reshape(a.shape)
    for h in (1e-7, 1e-6, 1e-5):
        fu = float(f(dict(theta, A=a + h * e)))
        fd = float(f(dict(theta, A=a - h * e)))
        print(f"A={a_fill:+.2f} zero_bound={zero_bound} h={h:g}: AD {g:.6f} FD {(fu - fd) / (2 * h):.6f}")


for fill, zb in ((0.03, True), (-0.03, False), (-0.03, True), (0.03, False)):
    check(fill, zb)
