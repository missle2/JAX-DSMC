"""Ridge sensitivity with a per-column (Marquardt) ridge instead of the mean-diagonal one."""
import sys

import jax
import numpy as np

from examples import ensemble_inlet_shock as eis
from jax_dsmc import ensemble_backward as eb


def call(self, query, gradient=False):
    from scipy.spatial import cKDTree

    query = np.asarray(query) / self.scale
    tree = cKDTree(self.points)
    dist, index = tree.query(query, k=self.k)
    dist, index = dist.reshape(len(query), self.k), index.reshape(len(query), self.k)
    pts = self.points[index]
    design = self._design(pts - query[:, None], query, pts)
    weights = 1 / (1 + (dist / np.maximum(dist[:, -1:], 1e-12)) ** 2)
    matrix = np.einsum("nki,nk,nkj->nij", design, weights, design)
    diag = np.einsum("nii->ni", matrix).copy()
    diag[:, 0] = 0.0
    matrix += eb.RIDGE_REL * diag[:, :, None] * np.eye(design.shape[2])
    rhs = np.einsum("nki,nk,nk->ni", design, weights, self.values[index])
    beta = np.linalg.solve(matrix, rhs[..., None])[:, :, 0]
    if gradient:
        return beta[:, 0], beta[:, 1 : 1 + self.points.shape[1]] / self.scale
    return beta[:, 0]


if sys.argv[3] == "marquardt":
    eb.LocalFit.__call__ = call
particles, steps = int(sys.argv[1]), int(sys.argv[2])
geometry = eis.inlet_shock_geometry(particles, 8, 0.05)
key = jax.random.key(32)
for ridge in (1e-10, 1e-8, 1e-6, 1e-4):
    eb.RIDGE_REL = ridge
    j, g, _ = eis.ensemble_gradient("temp", key, geometry, 8, 512, steps, "certified", particles, 32, False, True)
    print(f"{sys.argv[3]} RIDGE_REL={ridge:g} J={j:.10g} g={g:.10g}", flush=True)
