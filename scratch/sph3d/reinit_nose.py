"""Does reinitialisation alone make the M 10 sphere run's nose lobed (2026-10-26)?  The
iterate-60 level set (round nose sections) and a stretched copy whose nose is thinner (the
x axis of the coefficients scaled by 1.5 about the base, the radius kept: a nose ~1.5 knots in
radius, as at iterate 80) reinitialised 1, 2, 5 and 10 times with no step; per pass the nose
stations' diag / axis extent ratio and piece counts (scratch/sph3d/nose.py's measure)."""

import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy import ndimage

sys.path.insert(0, "scratch/sph3d")
import shape3d as S  # noqa: E402
from shape3d import O  # noqa: E402

TAG = "sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300"
out = Path(O.RESULTS) / TAG
cfg = dict(O.default_config(), **json.loads((out / "history.json").read_text())["config"])
_g, ls0 = O.initial_levelset(cfg)
fine = S.PhiGrid(ls0, [(-3.4, -1.0), (-0.6, 0.6), (-0.6, 0.6)], 0.01)


def nose(coef):
    phi = fine.phi(coef)
    res = []
    for i in range(fine.n[0]):
        sec = phi[i] < 0
        if not sec.any():
            continue
        yz = np.argwhere(sec) * fine.h + fine.lo[1:]
        d = yz - yz.mean(0)
        r = np.linalg.norm(d, axis=1)
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
        axial = r[(ang < 10) | (ang > 80)].max(initial=0)
        diag = r[(ang > 35) & (ang < 55)].max(initial=0)
        res.append((fine.lo[0] + i * fine.h, np.sqrt(sec.sum() / np.pi) * fine.h,
                    diag / max(axial, 1e-9), ndimage.label(sec)[1]))  # fmt: skip
    tip = res[0][0]
    near = [q for q in res if q[0] < tip + 0.3]
    lobed = max(abs(q[2] - 1.0) for q in near if q[1] > 0.02)
    pieces = max(q[3] for q in near)
    pieces3d = ndimage.label(phi < 0)[1]
    return f"tip {tip:+.2f} r(tip+0.1) {next(q[1] for q in res if q[0] >= tip + 0.1):.3f} " \
           f"max|diag/axis-1| {lobed:.2f} section pieces {pieces} 3-D pieces {pieces3d}"


def stretched(coef):
    """x scaled by 1.5 about x = 1 (the base): resample the coefficients."""
    xk = float(ls0.lo[0]) + float(ls0.h) * np.arange(coef.shape[0])
    src = 1.0 + (xk - 1.0) / 1.5
    idx = (src - float(ls0.lo[0])) / float(ls0.h)
    return ndimage.map_coordinates(coef, np.meshgrid(idx, np.arange(coef.shape[1]),
                                   np.arange(coef.shape[2]), indexing="ij"), order=1,
                                   mode="nearest")  # fmt: skip


for name, coef in (("it60", np.load(out / "coef_60.npy")),
                   ("it60 stretched", stretched(np.load(out / "coef_60.npy")))):  # fmt: skip
    ls = dataclasses.replace(ls0, coef=jnp.asarray(coef))
    print(f"{name:15s} pass 0: {nose(coef)}", flush=True)
    for p in range(1, 11):
        ls = O.L.reinitialise(ls)
        if p in (1, 2, 5, 10):
            print(f"{name:15s} pass {p}: {nose(np.asarray(ls.coef))}", flush=True)
