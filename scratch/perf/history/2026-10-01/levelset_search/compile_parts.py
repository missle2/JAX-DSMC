import sys, time, functools
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scratch/ls")
import levelset_old as OLD
from jax_dsmc.mesh import levelset as L
lo, m = L.grid_for_box(np.zeros(2), np.full(2, 20.0), 0.0625)
ls = L.from_analytic(np.array([10.0, 10.0]), 3.0, lo, 0.0625, m, n_sample=16)
n = 60256
x = jnp.zeros((n, 2)); v = jnp.ones((n, 3)); r = jnp.ones(n)
def ct(f, *a):
    t = time.perf_counter(); lo_ = jax.jit(f).lower(*a); t1 = time.perf_counter(); c = lo_.compile(); t2 = time.perf_counter()
    return f"lower {t1-t:.2f} compile {t2-t1:.2f} hlo {len(lo_.as_text())}"
for name, mod in (("old", OLD), ("new", L)):
    print(name, "hit rem", ct(lambda x, v, r, ls: mod.hit_body(x, v, ls, r), x, v, r, ls))
    print(name, "hit none", ct(lambda x, v, ls: mod.hit_body(x, v, ls, None), x, v, ls))
    print(name, "inside", ct(lambda x, ls: mod.inside(ls, x), x, ls))
    g = jax.grad(lambda c, x, v, r, ls: jnp.sum(jnp.where(jnp.isfinite(t := mod.hit_body(x, v, ls.__class__(c, ls.lo, ls.h, ls.n_sample, ls.n_bisect), r)), t, 0.0)))
    print(name, "grad hit", ct(g, ls.coef, x, v, r, ls))
    print(name, "samples-only" if name=="new" else "", "")
if True:
    ds = jnp.ones(n); ph = jnp.ones(n)
    print("new _samples", ct(lambda x, v, d, p: L._samples(ls, x, v[:, :2], d, p), x, v, ds, ph))
    print("new _search_rows", ct(lambda x, v, s, d: L._search_rows(ls, x, v[:, :2], s, d), x, v, ds, ds))
    print("new table", ct(lambda x, d, ls: L._hit_candidates(ls, x, d, d > 0, True), x, ds, ls))
