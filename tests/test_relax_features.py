"""The signed correction features (2026-11-14: ``div_u``, ``t_strain_t``) and models of
fewer inputs: a compressive cell reads ``div_u < 0``, an expanding one ``> 0``; a model
fitted on the first nine features (features_v2) runs unchanged on the eleven-feature
operator -- the same step as an eleven-input model with zero weights on the new two."""

from __future__ import annotations

import dataclasses
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import test_relax
import test_relax_bgk

from jax_dsmc import sort
from jax_dsmc.collide import relax

LINEAR = test_relax_bgk.LINEAR
DIV_U = relax.FEATURES.index("div_u")


def _sound_box(cells=8, npc=2000, amplitude=60.0):
    """A periodic 1-D box with the longitudinal wave ``u_x = A sin(2 pi x / L)``: ``div u =
    A k cos(k x)``, compression where the cosine is negative."""
    state, cs, grid, sp, params, static, key = test_relax._box(n_part=cells * npc,
                                                               cells=(cells,))  # fmt: skip
    x = np.asarray(state.x)[:, 0]
    length = float(grid.hi[0] - grid.lo[0])
    v = np.asarray(state.v).copy()
    v[:, 0] += amplitude * np.sin(2 * np.pi * x / length)
    return dataclasses.replace(state, v=jnp.asarray(v)), cs, grid, sp, params, static, key


def _features(box):
    state, _, grid, sp, params, _, _ = box
    table = sort.build_table(state, grid)
    _, _, n, u, _, pi, theta_c = relax.local_state(state, table, grid, (True,))
    _, n_s, _, theta_s = relax.safe_cells(n, pi, theta_c)
    return np.asarray(relax.cell_features(n, n_s, u, theta_s, grid, sp, params, (True,)))


def test_divergence_feature_has_the_sign_of_the_compression():
    box = _sound_box()
    phi = _features(box)
    assert phi.shape[1] == len(relax.FEATURES) and np.isfinite(phi).all()
    cells = phi.shape[0]
    centres = (np.arange(cells) + 0.5) / cells
    expected = np.cos(2 * np.pi * centres)
    strong = np.abs(expected) > 0.5
    assert (np.sign(phi[strong, DIV_U]) == np.sign(expected[strong])).all()


def test_a_nine_input_model_reads_the_first_nine_features():
    """Same noise: a 9-input linear model and an 11-input one with the same first nine
    columns (the last two zero) give the same step, bitwise."""
    box = test_relax._shear_box(cells=8, npc=500)
    state, cs, grid, sp, params, static, key = box
    tau = test_relax._tau_s(state, grid, sp, params)
    rng = np.random.default_rng(0)
    a9 = 0.05 * rng.standard_normal((2, 9))
    nine = dict(relax.init_correction("linear", n_features=9), A=jnp.asarray(a9))
    a11 = np.concatenate([a9, np.zeros((2, 2))], axis=1)
    eleven = dict(relax.init_correction("linear"), A=jnp.asarray(a11))
    outs = []
    for theta in (nine, eleven):
        (out, _), _ = test_relax_bgk._run(LINEAR, state, cs, grid, sp, params, static, key, 2,
                                          0.2 * tau, theta, moments=False)  # fmt: skip
        outs.append(np.asarray(out.v))
    np.testing.assert_array_equal(outs[0], outs[1])
    assert relax.correction_inputs(nine) == 9
    assert relax.correction_inputs(eleven) == len(relax.FEATURES)
