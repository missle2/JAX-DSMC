"""shape.volume_cotangent against central differences of the forward's own cut volumes on the
fitgate M 10 circle (circle_long, level set h 0.0625, control ``radius``), random per-cell
weights and weights on the cut cells only (2026-11-01: is the geometric half of the cut-volume
term right where the M 10 bias is?).

    PYTHONPATH=. python3 scratch/fitgate/check_volume_c10.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/fitgate")
import variant_gate as V  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import shape  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    cfg, parameter, *_ = V.gate_config("c10_spec")
    r0 = float(EM.nominal(cfg, parameter))
    grid, mesh, *_ = EM.build(cfg, parameter, r0, 10)
    n = grid.volume.shape[0]
    cut_cells = np.zeros(n, bool)
    cut_cells[: len(mesh.cell_type)] = np.asarray(mesh.cell_type) == cut.CUT
    rng = np.random.default_rng(0)
    sets = {
        "random all": rng.normal(size=n),
        "random cut": np.where(cut_cells, rng.normal(size=n), 0.0),
        "ones cut": cut_cells.astype(float),
    }
    vols = {}
    for h in (0.02, 0.01, 0.005):
        for sg in (1, -1):
            g2, *_ = EM.build(cfg, parameter, r0 + sg * h, 10)
            assert g2.volume.shape[0] == n, "cell count changed"
            vols[(h, sg)] = np.asarray(g2.volume)
    for name, w in sets.items():
        cot = shape.volume_cotangent(shape.geometry(mesh), mesh, grid, jnp.asarray(w))
        adj = float(EM.control_gradient(cfg, parameter, grid, cot))
        for h in (0.02, 0.01, 0.005):
            fd = float(np.dot(w, (vols[(h, 1)] - vols[(h, -1)]) / (2 * h)))
            print(f"{name:10s} h={h}: VJP {adj:.6g}  FD {fd:.6g}  ratio {adj / fd:.4f}", flush=True)
    print(f"R {r0}  cells {n}  cut {int(cut_cells.sum())}  n_sub {int(grid.n_sub)}")


if __name__ == "__main__":
    main()
