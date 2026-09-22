"""Where are the host fits ill-conditioned?  Records k (nodes used) and the normal-matrix
condition number of every LocalFit call during one backward of the inlet case."""
import sys

import jax
import numpy as np

from examples import ensemble_inlet_shock as eis
from jax_dsmc import ensemble_backward as eb

orig_call = eb.LocalFit.__call__
stats = []


def call(self, query, gradient=False):
    from scipy.spatial import cKDTree

    q = np.asarray(query) / self.scale
    tree = cKDTree(self.points)
    dist, index = tree.query(q, k=self.k)
    dist, index = dist.reshape(len(q), self.k), index.reshape(len(q), self.k)
    pts = self.points[index]
    design = self._design(pts - q[:, None], q, pts)
    weights = 1 / (1 + (dist / np.maximum(dist[:, -1:], 1e-12)) ** 2)
    matrix = np.einsum("nki,nk,nkj->nij", design, weights, design)
    d = np.sqrt(np.einsum("nii->ni", matrix))
    d = np.where(d > 0, d, 1.0)
    scaled = matrix / d[:, :, None] / d[:, None, :]
    cond = np.linalg.cond(scaled)
    stats.append((self.k, len(self.points), design.shape[2], cond))
    return orig_call(self, query, gradient)


eb.LocalFit.__call__ = call
particles, steps = int(sys.argv[1]), int(sys.argv[2])
geometry = eis.inlet_shock_geometry(particles, 8, 0.05)
key = jax.random.key(32)
j, g, _ = eis.ensemble_gradient("temp", key, geometry, 8, 512, steps, "certified", particles, 32, False, True)
ks = np.array([s[0] for s in stats]); ns = np.array([s[1] for s in stats]); cols = stats[0][2]
conds = np.concatenate([s[3] for s in stats])
print(f"fits={len(stats)} columns={cols} queries={len(conds)}")
print("nodes per fit: min", ns.min(), "median", np.median(ns), "max", ns.max(), " fits with nodes<columns:", (ns < cols).sum())
for t in (1e6, 1e8, 1e10, 1e12):
    print(f"queries with cond>{t:g}: {(conds > t).mean():.3%}")
