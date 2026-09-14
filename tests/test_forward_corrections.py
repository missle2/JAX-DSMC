"""Independent regression cases from the forward review."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import inflow
from jax_dsmc.boundary import BC, Faces
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import GridSpec, ParticleState


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_face_area_uses_physical_transverse_measure(dim):
    grid = GridSpec(lo=jnp.zeros(dim), hi=jnp.arange(2, 2 + dim, dtype=float),
                    n=(1,) * dim, transverse=jnp.float64(0.01 if dim < 3 else 1))
    for axis in range(dim):
        expected = float(grid.transverse) * np.prod(np.delete(np.asarray(grid.hi), axis))
        assert float(inflow.face_area(grid, axis)) == pytest.approx(expected)


def test_two_faces_allocate_actual_births_with_capacity_larger_than_storage():
    n = 40
    grid = GridSpec(lo=jnp.zeros(1), hi=jnp.ones(1), n=(1,))
    state = ParticleState(x=jnp.zeros((n, 1)), v=jnp.zeros((n, 3)),
                          species=jnp.zeros(n, jnp.int32), alive=jnp.zeros(n, bool),
                          logh_acc=jnp.zeros(n))
    # Unit normal thermal variance, mean flux 1/sqrt(2 pi); exactly ten births/face.
    res = inflow.InflowParams.uniform(1, 10 * np.sqrt(2 * np.pi), 1.)
    faces = Faces((BC.INFLOW, BC.INFLOW))
    out, inserted, shortfall = inflow.inject(state, grid, faces, res, K_BOLTZ,
                                            1., 1., jax.random.key(3), capacity=60)
    assert int(inserted) == 20
    assert int(out.alive.sum()) == 20
    assert float(shortfall) == 0
