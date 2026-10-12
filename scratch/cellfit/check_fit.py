"""Unit checks of jax_dsmc.cell_fit: polynomial reproduction, agreement with a numpy
least-squares reference per cell, the adaptive / loo fallbacks, the quad basis matching
ensemble_device._design's column space, and the slope evaluation."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import cell_fit, ensemble_device

D, C, P = 2, 7, 64
rng = np.random.default_rng(0)
counts = np.array([60, 40, 25, 14, 9, 3, 0])
valid = np.arange(P)[None, :] < counts[:, None]
scale = jnp.asarray([0.1, 0.1, 2.0, 2.0, 2.0])
x = rng.uniform(0, 0.1, (C, P, D)) + np.arange(C)[:, None, None] * 0.1
v = rng.normal(0, 2.0, (C, P, 3)) + 1.5


def poly_field(x, v, poly):
    o = np.concatenate((x / 0.1, v / 2.0), -1)
    terms = cell_fit.monomials(poly, D)
    coef = np.random.default_rng(1).normal(size=len(terms))
    return sum(c * np.prod([o[..., a] for a in t], axis=0) for c, t in zip(coef, terms))


print("columns", {p: len(cell_fit.monomials(p, D)) for p in cell_fit.POLYS})
assert len(cell_fit.monomials("quad", D)) == ensemble_device._design(jnp.zeros((1, 1, 5)), "quadratic_full").shape[-1]

for poly in cell_fit.POLYS:
    lam = poly_field(x, v, poly)
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), scale, f"cell:{poly}:ridge0")
    c = np.repeat(np.arange(C), P)
    got = np.asarray(cell_fit.evaluate(f, jnp.asarray(c), jnp.asarray(x.reshape(-1, D)), jnp.asarray(v.reshape(-1, 3)), scale, f"cell:{poly}")).reshape(C, P)
    p = len(cell_fit.monomials(poly, D))
    ok = counts > p
    err = np.abs(got - lam)[valid & ok[:, None]].max() / np.abs(lam).max()
    print(f"reproduce {poly:9s} p={p:2d}  max rel err (cells with count > p) {err:.1e}")
    assert err < 1e-8

# numpy lstsq reference on noisy data, unweighted
lam = rng.normal(size=(C, P))
for poly in ("quad", "cubicv"):
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), scale, f"cell:{poly}:ridge0")
    worst = 0
    for ci in range(C):
        n = counts[ci]
        p = len(cell_fit.monomials(poly, D))
        if n <= p:
            continue
        z = np.concatenate((x[ci, :n], v[ci, :n]), -1) / np.asarray(scale)
        o = z - z.mean(0)
        A = np.asarray(cell_fit.features(jnp.asarray(o), poly))
        beta = np.linalg.lstsq(A, lam[ci, :n], rcond=None)[0]
        worst = max(worst, np.abs(beta - np.asarray(f.beta[ci])).max() / np.abs(beta).max())
    print(f"lstsq {poly}: max rel coef err {worst:.1e}")
    assert worst < 1e-8

# adaptive: cell with 9 nodes (D=2: orders 0,1,2 -> cum 1, 6, 14) at adapt 1.5 -> order 1
f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), scale, "cell:quad:adapt1.5")
order = np.asarray(cell_fit.orders("quad", D))
for ci in range(C):
    nz = np.abs(np.asarray(f.beta[ci])) > 0
    print(f"adapt1.5 count {counts[ci]:2d}: highest order with nonzero coef", order[nz].max() if nz.any() else None)
f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), scale, "cell:cubicv:loo")
order = np.asarray(cell_fit.orders("cubicv", D))
for ci in range(C):
    nz = np.abs(np.asarray(f.beta[ci])) > 0
    print(f"loo noise count {counts[ci]:2d}: order", order[nz].max() if nz.any() else None)
lam2 = poly_field(x, v, "quad") + 1e-3 * rng.normal(size=(C, P))
f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam2), scale, "cell:cubicv:loo")
for ci in range(C):
    nz = np.abs(np.asarray(f.beta[ci])) > 1e-12
    print(f"loo quad-field count {counts[ci]:2d}: order", order[nz].max() if nz.any() else None)

# slopes: analytic derivative of a known polynomial
lam = poly_field(x, v, "quad")
f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), scale, "cell:quad:ridge0")
qx, qv = jnp.asarray(x[0, :5]), jnp.asarray(v[0, :5])
_, gx, gv = cell_fit.evaluate(f, jnp.zeros(5, jnp.int32), qx, qv, scale, "cell:quad:ridge0", gradients=True)
fn = lambda xx, vv: jnp.asarray(poly_field(np.asarray(xx), np.asarray(vv), "quad"))
eps = 1e-6
num = []
for a in range(5):
    dz = np.zeros(5); dz[a] = eps
    up = poly_field(np.asarray(qx) + dz[:2], np.asarray(qv) + dz[2:], "quad")
    dn = poly_field(np.asarray(qx) - dz[:2], np.asarray(qv) - dz[2:], "quad")
    num.append((up - dn) / (2 * eps))
num = np.stack(num, 1)
got = np.concatenate((np.asarray(gx), np.asarray(gv)), 1)
print("slope max rel err", np.abs(got - num).max() / np.abs(num).max())
assert np.abs(got - num).max() / np.abs(num).max() < 1e-6
# value columns
lam_r = np.stack((lam, 2 * lam), -1)
f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam_r), scale, "cell:quad")
out = cell_fit.evaluate(f, jnp.zeros(5, jnp.int32), qx, qv, scale, "cell:quad")
print("columns shape", out.shape, "ratio", np.asarray(out[:, 1] / out[:, 0]))
print("ALL OK")
