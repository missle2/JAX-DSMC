"""Is the shock-window AD/FD mismatch a flip (FD across a discrete event)?"""
import functools, sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np
import test_learned_window2d as t
from examples.learned_collision import operators, windows2d as w2

case, px, pv, pa, target = t._shock()
fn, theta, _ = operators.operator("scalar", case.kind)
states, cstates, keys = t._replicas(case, px, pv, pa)
weights = jnp.ones(2)
bias = jnp.zeros((2, case.n_coarse, 2))
f = jax.jit(functools.partial(t._objective, states=states, cstates=cstates, keys=keys,
            target=jnp.asarray(target), weights=weights, bias=bias, case=case, fn=fn, estimator="pair"))
g = jax.jit(jax.grad(f))(theta)
print("AD", float(g["log_tau"]))
for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    up = dict(theta, log_tau=theta["log_tau"] + h); dn = dict(theta, log_tau=theta["log_tau"] - h)
    fu, f0, fd_ = float(f(up)), float(f(theta)), float(f(dn))
    print(f"h {h:.0e}  central {(fu-fd_)/(2*h): .6f}  fwd {(fu-f0)/h: .6f}  bwd {(f0-fd_)/h: .6f}")
