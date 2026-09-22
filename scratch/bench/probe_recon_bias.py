"""Interpolant bias in isolation: collisionless inlet (q0 = 0), where the exact backward
field is the ballistic pull-back of the terminal reward.  Force the local-affine
reconstruction of that known field and compare the source gradient with the exact one.
Everything else (fixed counts, source samples, escaped births) is identical."""

import sys

import numpy as np

from jax_dsmc import ensemble_inflow as ei

R = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N0 = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
cfg = ei.Config(n_initial=N0, cells=40, dt=0.025, steps=20, retained=R, source_samples=1995,
                partners=1, neighbors=32, q0=0.0)
T = np.full((2, 3), 4.0)
exact, affine = [], []
original = ei._reconstruct
for seed in range(3):
    tr = ei.forward(cfg, T, seed=seed, injection_counts=ei.nominal_counts(cfg, T))
    ei._reconstruct = original
    exact.append(ei.backward(tr, seed=seed)[:, 0])
    # force the affine fit of the (exactly known) field, as the collisional run must do
    ei._reconstruct = lambda t, v, c, n, r, f, s: ei.LocalAffine(t, v, c.neighbors, s)
    affine.append(ei.backward(tr, seed=seed)[:, 0])
ei._reconstruct = original
exact, affine = np.array(exact), np.array(affine)
print(f"R={R}: exact G = {exact.mean(0).round(4)} +- {(exact.std(0, ddof=1)/np.sqrt(3)).round(4)}")
print(f"R={R}: affine G = {affine.mean(0).round(4)} +- {(affine.std(0, ddof=1)/np.sqrt(3)).round(4)}"
      f"   ratio {(affine.mean(0)/exact.mean(0)).round(3)}")
