"""The shock loss on the domain's low cosine modes (windows2d.shock_basis, 2026-11-12)."""

from __future__ import annotations

import types

import numpy as np

from examples.learned_collision import windows2d as w2


def test_basis_is_orthonormal_and_row_zero_is_the_integral():
    b = np.asarray(w2.shock_basis(15, 15))
    np.testing.assert_allclose(b @ b.T, np.eye(15), atol=1e-12)
    d = np.random.default_rng(0).normal(size=(3, 15, 2))
    case = types.SimpleNamespace(kind="shock", n_coarse=15, modes=1)
    p = np.asarray(w2.project(d, case))
    np.testing.assert_allclose(p[:, 0], d.sum(axis=1) / np.sqrt(15), rtol=1e-12)


def test_all_modes_give_the_per_cell_loss_back():
    d = np.random.default_rng(1).normal(size=(3, 15, 2))
    full = types.SimpleNamespace(kind="shock", n_coarse=15, modes=15)
    cells = types.SimpleNamespace(kind="shock", n_coarse=15, modes=0)
    np.testing.assert_allclose(np.sum(np.asarray(w2.project(d, full)) ** 2),
                               np.sum(np.asarray(w2.project(d, cells)) ** 2), rtol=1e-12)  # fmt: skip
    assert w2.loss_rows(full) == 15 and w2.loss_rows(cells) == 15
    assert w2.loss_rows(types.SimpleNamespace(kind="shock", n_coarse=15, modes=2)) == 2
