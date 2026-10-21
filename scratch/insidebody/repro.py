"""Reproduce the 2026-10-21 ``inside_body fired`` of optrules2 kn0.5_M10_tr2 iterate 68: the
warm start (warm/seed<i>.npz, the gas iterate 67 left) on coef_68, settle + captured steps
(no backward), the per-step inside_body counts of both phases and the carried particles the
level set calls outside but the cut-cell polygon calls inside.

    PYTHONPATH=. python3 scratch/insidebody/repro.py RUN_DIR ITERATE SEED [fnum_mult]
"""

import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.grid import cell_of  # noqa: E402
from jax_dsmc.mesh import geometry as mesh_geometry  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402


def main():
    run, it, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(f"{run}/history.json") as fh:
        cfg = dict(O.default_config(), **json.load(fh)["config"])
    _grid, ls = O.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    key = jax.random.key(cfg["seed_start"] + seed)
    state, cstate = O.load_warm(O._warm_path(run, seed))
    grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
        cfg, "radius", EM.nominal(cfg, "radius"), cfg["steps"], ls
    )
    flow = FlowParams(nrho=jnp.float64(EM.F.NRHO * EM.knudsen_density(cfg)),
                      temp=jnp.float64(EM.F.TEMP),
                      vstream=jnp.array([EM.stream_speed(cfg), 0.0, 0.0]))  # fmt: skip
    fresh = (initialize.initial_state(key, flow, grid, sp, static, n_alive=0),
             initialize.initial_collide_state(flow, grid, sp, static))  # fmt: skip
    x, alive = state.x, state.alive
    phi_in = L.inside(mesh.body, x) & alive
    poly_in = mesh_geometry.inside_polygon(x, cell_of(x, grid), mesh, grid, alive)
    print(f"carried alive {int(alive.sum())}: level set inside {int(phi_in.sum())}, polygon "
          f"inside {int(poly_in.sum())}, polygon-only {int((poly_in & ~phi_in).sum())}, "
          f"level-set-only {int((phi_in & ~poly_in).sum())}", flush=True)  # fmt: skip
    state, cstate = EM._carried(state, cstate, mesh, fresh, grid)
    common = (grid, sp, params, key, faces, wall, res, mesh)
    settle = int(cfg["restart_steps"]) or 500
    warm = dataclasses.replace(static, n_steps=settle)
    (state, cstate), aux = EM._run(state, cstate, *common, warm, capacity,
                                   cfg["warmup"] - settle, False)  # fmt: skip
    ins = np.asarray(aux[1].inside_body)
    print(f"settle: inside_body total {int(ins.sum())}, steps {np.nonzero(ins)[0][:20].tolist()}",
          flush=True)  # fmt: skip
    (_final, _c), aux = EM._run(state, cstate, *common, static, capacity, cfg["warmup"], False)
    ins = np.asarray(aux[1].inside_body)
    print(f"capture: inside_body total {int(ins.sum())}, steps {np.nonzero(ins)[0][:20].tolist()}",
          flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
