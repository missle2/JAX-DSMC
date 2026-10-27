"""The M 10 sphere drag run's nose (2026-10-26): y-z cross-sections of phi near the tip,
equivalent radius and |grad phi| on the zero set per station, the lobes' orientation, and the
nose's history (tip x, pieces, the thinnest station) over the iterates.

    PYTHONPATH=. python3 scratch/sph3d/nose.py [TAG]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, "scratch/sph3d")
import shape3d as S  # noqa: E402
from shape3d import O  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300"
out = Path(O.RESULTS) / TAG
hist = json.loads((out / "history.json").read_text())
cfg = dict(O.default_config(), **hist["config"])
_grid, ls0 = O.initial_levelset(cfg)
h_knot = float(ls0.h)
fine = S.PhiGrid(ls0, [(-3.0, -1.0), (-0.6, 0.6), (-0.6, 0.6)], 0.01)
n = len(hist["rows"])


def stations(phi, g):
    """Per x station: inside area, equivalent radius, |grad phi| median on the contour, the
    ratio of the section's extent along the diagonals to along the axes (lobes: > 1 on the
    diagonals, < 1 on the axes) and the piece count."""
    grads = np.gradient(phi, g.h)
    gn = np.sqrt(sum(q**2 for q in grads))
    rows = []
    for i in range(0, g.n[0], 5):
        sec = phi[i] < 0
        if not sec.any():
            continue
        x = g.lo[0] + i * g.h
        area = sec.sum() * g.h**2
        edge = sec & ~ndimage.binary_erosion(sec)
        yz = np.argwhere(sec) * g.h + g.lo[1:]
        c = yz.mean(0)
        d = yz - c
        r = np.linalg.norm(d, axis=1)
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
        axial = r[(ang < 10) | (ang > 80)].max(initial=0)
        diag = r[(ang > 35) & (ang < 55)].max(initial=0)
        rows.append((x, area, np.sqrt(area / np.pi), float(np.median(gn[i][edge])),
                     diag / max(axial, 1e-9), ndimage.label(sec)[1]))  # fmt: skip
    return rows


for k in sorted({0, 10, 20, 30, 40, 60, 80, n}):
    p = out / f"coef_{k}.npy"
    if not p.exists():
        continue
    phi = fine.phi(np.load(p))
    st = stations(phi, fine)
    if not st:
        print(f"k {k}: no body in the window")
        continue
    tip = st[0][0]
    print(f"k {k:3d}: tip x {tip:+.2f}  " + "  ".join(
        f"[x {x:+.2f} r {r:.3f} |gphi| {gp:.2f} diag/axis {q:.2f} pcs {pc}]"
        for x, _a, r, gp, q, pc in st[:6]), flush=True)  # fmt: skip
    if k == n:
        fig, axes = plt.subplots(1, 6, figsize=(16, 3))
        for ax, (x, *_r) in zip(axes, st[:6], strict=False):
            i = int(round((x - fine.lo[0]) / fine.h))
            ext = (fine.lo[1], fine.lo[1] + fine.h * (fine.n[1] - 1),
                   fine.lo[2], fine.lo[2] + fine.h * (fine.n[2] - 1))  # fmt: skip
            ax.imshow(phi[i].T, origin="lower", extent=ext, cmap="RdBu", vmin=-0.2, vmax=0.2)
            ax.contour(np.linspace(ext[0], ext[1], fine.n[1]),
                       np.linspace(ext[2], ext[3], fine.n[2]), phi[i].T, levels=[0.0],
                       colors="k", linewidths=1)  # fmt: skip
            # knot lines
            for kk in np.arange(-0.6, 0.61, h_knot):
                ax.axvline(kk + (float(ls0.lo[1]) % h_knot), color="0.6", lw=0.4)
                ax.axhline(kk + (float(ls0.lo[2]) % h_knot), color="0.6", lw=0.4)
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_title(f"x = {x:+.2f}", fontsize=9)
            ax.set_aspect("equal")
        fig.suptitle(f"{TAG} iterate {k}: y-z sections of phi at the nose (grey: knot lines)")
        fig.tight_layout()
        fig.savefig("scratch/sph3d/nose_sections.png", dpi=90)
