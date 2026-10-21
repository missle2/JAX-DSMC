"""The culprit's second leg (after its reflection at the M 10 nose): the hit search from the
reflection point and phi along the leg (scratch/insidebody/culprit.py's numbers)."""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

RUN = "examples/mesh/results/optimize/circle_long_kn0.5_M10_S4_tr2_tan_sm8_free_g32_w500"
with open(f"{RUN}/history.json") as fh:
    cfg = dict(O.default_config(), **json.load(fh)["config"])
_g, ls = O.initial_levelset(cfg)
ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{RUN}/coef_68.npy")))
x0 = np.array([0.33523155, 5.06590921])
v0 = np.array([3468.05011994, 166.9773404])
v1 = np.array([429.4581106, 3445.40546204])
dt = 2.5e-5
t_hit = float(L.hit_body(jnp.asarray(x0[None]), jnp.asarray(v0[None]), ls, jnp.asarray([dt]))[0])
xh = x0 + v0 * t_hit
phi_h, g_h = L.value_and_grad(ls, jnp.asarray(xh[None]))
n = np.asarray(g_h)[0] / np.linalg.norm(np.asarray(g_h)[0])
rem = dt - t_hit
print(f"hit t {t_hit:.4e}, point {xh}, phi {float(phi_h[0]):.2e}, normal {n}, v1.n {v1 @ n:.1f}")
t2 = float(L.hit_body(jnp.asarray(xh[None]), jnp.asarray(v1[None]), ls, jnp.asarray([rem]))[0])
print(f"second leg: remaining {rem:.4e}, hit_body -> {t2:.4e}, leg length {np.linalg.norm(v1) * rem:.4f}, "
      f"ds = leg/{ls.n_sample} = {np.linalg.norm(v1) * rem / ls.n_sample:.4f}")
s = np.linspace(0, 1, 2001)[:, None]
leg = xh[None] + s * v1[None] * rem
ph = np.asarray(L.value(ls, jnp.asarray(leg)))
sign = np.sign(ph)
changes = np.nonzero(np.diff(sign))[0]
print("phi along the leg: start", ph[0], "end", ph[-1], "max", ph.max(), "at s", float(s[ph.argmax(), 0]))
print("sign changes at s:", [float(s[i, 0]) for i in changes][:10])
k = np.arange(1, ls.n_sample + 1) / ls.n_sample
ph_k = np.asarray(L.value(ls, jnp.asarray(xh[None] + k[:, None] * v1[None] * rem)))
print("the 16 samples:", np.array2string(ph_k, precision=2))
# the nose: phi's zero set near the hit point
xs = np.linspace(xh[0] - 0.15, xh[0] + 0.15, 7)
print("nose geometry: phi on y = hit y and x around the hit:",
      np.array2string(np.asarray(L.value(ls, jnp.asarray(np.c_[xs, np.full_like(xs, xh[1])]))),
                      precision=3))
