"""Diffuse normal-score term on the capsule's constructed rays vs the level-set resolution."""
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")
import test_mesh_adjoint as T  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import shape  # noqa: E402
from jax_dsmc.boundary import BC, Faces, WallParams  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.mesh import cut, levelset  # noqa: E402

vbar = np.sqrt(np.pi * K_BOLTZ * T.T_WALL / (2 * T.MASS))
h = float(sys.argv[1])
ns = int(sys.argv[2])
K = int(sys.argv[3])
grazing = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
cfg = EM.default_config()
cfg.update({"case": "plate", "body": "levelset", "acc": 1.0, "levelset_samples": ns, "levelset_h": h, "grazing": grazing})
grid = T._grid2()
coef, ls = EM.levelset_coef(jnp.float64(1.0), cfg, "width", grid)
import dataclasses  # noqa: E402

ls = dataclasses.replace(ls, grazing=grazing)
mesh = cut.build_mesh(ls, grid, radius=1, sub=4)
faces = Faces.uniform(BC.SPECULAR, 2)
wall = WallParams.uniform(2, T.T_WALL, acc=0.0, n_models=1).with_model(2, 0, temp=T.T_WALL, acc=1.0)
db = 0.002
b = jnp.arange(-3.0, 3.0, db)
n = b.shape[0]
x = jnp.stack((jnp.full(n, 5.0 - 0.4 - 0.5), 5.0 + b), axis=1)
v = jnp.zeros((n, 3)).at[:, 0].set(500.0)
dt = 0.9 / 500.0
alive = jnp.ones(n, bool)
static = T.SimStatic(N=n, D=2, S=1, G=1, A_max=2, K_bounce=4, n_steps=1)
setup = shape._setup(jax.random.PRNGKey(3), grid, faces, wall, T.MASS, dt, mesh, 0.0, 0.0, static)["map"]
scale = 1e-3
reward = functools.partial(T._drag, mass=T.MASS, scale=scale)
geom = shape.geometry(mesh)
_, vjp_w = jax.vjp(lambda w: EM.levelset_coef(w, cfg, "width", grid)[0], jnp.float64(1.0))
ang = np.linspace(0, np.pi, 200)
tip = np.stack([5.0 + 0.4 * np.cos(ang), 5.0 + 1.0 + 0.4 * np.sin(ang)], 1)
err = np.abs(np.asarray(levelset.value(ls, jnp.asarray(tip)))).max()


def step(key):
    f = functools.partial(shape.wall_map, x=x, v=v, alive=alive, key=key, **setup)
    primal, vjp = jax.vjp(f, geom)
    events = primal[5]
    r_k = reward(events, None, None, 0)
    to_go = shape._to_go(jnp.zeros(n), r_k)
    zeros = shape._zero_cotangent(primal)
    cot_events = zeros[5]._replace(normal=shape.normal_score(events, to_go[:-1], wall, mesh, T.MASS, 2))
    (d_score,) = vjp((zeros[0], zeros[1], zeros[2], to_go[0], zeros[4], cot_events))
    return vjp_w(d_score)[0]


step_j = jax.jit(step)
vals = np.array([float(step_j(jax.random.PRNGKey(1000 + k))) for k in range(K)])
ref = 2 / db * T.MASS * scale * vbar
print("draws:", np.array2string(vals / ref, precision=2))
print(f"h_phi {h} grazing {grazing}: score {vals.mean():.4e} ± {vals.std(ddof=1)/np.sqrt(K):.1e} vs analytic {ref:.4e} ({vals.mean()/ref*100:.1f} ± {vals.std(ddof=1)/np.sqrt(K)/ref*100:.1f} %)  cap surface error {err:.2e}", flush=True)
