"""Spatial domain decomposition (spec section 20.2).

These need more than one device.  On a single-GPU machine, run them against
forced-host devices::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/test_decomp.py -q

Fake devices cannot measure speed -- they time-share one processor -- but they
exercise every collective and every capacity, which is what these assert.

**Particle conservation is the load-bearing assertion.**  Both bugs found while
building this module lost or duplicated particles while reporting no overflow and
raising nothing: padding indices clamped to slot 0 (which made a padded scatter
fight a real one), and arrivals left uncompacted in two blocks when the free-slot
run is one block.  Neither is visible in a moment profile at plausible statistics;
counting is the only thing that catches them.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from examples.turbulence.case import initial_particles, setup  # noqa: E402
from jax_dsmc import decomp, parallel  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.simulate import run  # noqa: E402
from jax_dsmc.state import ParticleState  # noqa: E402

multidevice = pytest.mark.skipif(
    len(jax.devices()) < 2, reason="needs >= 2 devices (see module docstring)"
)


def _slab_packed_state(st0, grid, n_part, p, per, dim):
    """Regroup a global state into ``p`` equal per-device blocks, padded."""
    slab = np.asarray(decomp.slab_of(st0.x, grid, p))
    xs, vs, sps, al, lg = [], [], [], [], []
    for r in range(p):
        idx = np.where(slab == r)[0]
        assert idx.size <= per, f"slab {r} holds {idx.size} > capacity {per}"
        pad = per - idx.size
        xs.append(np.concatenate([np.asarray(st0.x)[idx], np.zeros((pad, dim))]))
        vs.append(np.concatenate([np.asarray(st0.v)[idx], np.zeros((pad, 3))]))
        sps.append(np.zeros(per, np.int32))
        al.append(np.concatenate([np.ones(idx.size, bool), np.zeros(pad, bool)]))
        lg.append(np.zeros(per))
    return ParticleState(
        x=jnp.asarray(np.concatenate(xs)),
        v=jnp.asarray(np.concatenate(vs)),
        species=jnp.asarray(np.concatenate(sps)),
        alive=jnp.asarray(np.concatenate(al)),
        logh_acc=jnp.asarray(np.concatenate(lg)),
    )


def _run_pair(dim, side, p, steps=10, npc=40):
    sp, grid, static, params, cstate, info = setup(
        dim=dim, cells_per_side=side, npc=npc, n_steps=steps
    )
    static = dataclasses.replace(static, n_steps=steps)
    key = jax.random.PRNGKey(0)
    st0 = initial_particles(key, grid, static, info)
    (s1, _c1), (m1, _d1) = jax.jit(
        lambda s, c: run(s, c, grid, sp, params, static, key, bc=BC.PERIODIC)
    )(st0, cstate)

    mesh = parallel.device_mesh(p)
    per = decomp.suggest_capacity(info["N"], p, headroom=1.6)
    m_max = decomp.suggest_migration_buffer(per, side // p)
    st_d = _slab_packed_state(st0, grid, info["N"], p, per, dim)
    with mesh:
        st_d = parallel.shard_state(st_d, mesh)
        (s2, _c2), (m2, _d2, mig) = decomp.run_decomposed(
            st_d, cstate, grid, sp, params, static, key, mesh, BC.PERIODIC, steps, m_max
        )
    return info, m1, m2, s1, s2, mig


@multidevice
@pytest.mark.parametrize(("dim", "side"), [(1, 64), (2, 16), (3, 8)])
def test_particles_are_conserved(dim, side):
    """No particle is lost or duplicated by migration, in any dimension."""
    p = min(4, len(jax.devices()))
    info, _m1, _m2, _s1, s2, mig = _run_pair(dim, side, p)
    assert int(jnp.sum(s2.alive)) == info["N"]
    assert int(jnp.sum(mig["overflow_send"])) == 0
    assert int(jnp.sum(mig["overflow_recv"])) == 0
    # A CFL-limited step can only reach an adjacent slab.  A non-zero count here
    # means the timestep is too large for the decomposition, and the nearest-
    # neighbour exchange would then be silently losing those particles.
    assert int(jnp.sum(mig["far_jumps"])) == 0


@multidevice
@pytest.mark.parametrize(("dim", "side"), [(1, 64), (2, 16)])
def test_density_matches_single_device(dim, side):
    """Mean density is exact: it counts particles, and both runs hold the same set.

    The temperature is *not* asserted equal.  Decomposed runs fold the device index
    into the key, so the slabs draw independent streams and the two runs are
    statistically equivalent rather than identical (see
    :func:`jax_dsmc.decomp.run_decomposed`).
    """
    p = min(4, len(jax.devices()))
    _info, m1, m2, _s1, _s2, _mig = _run_pair(dim, side, p)
    assert float(jnp.mean(m2.nrho[-1])) == pytest.approx(float(jnp.mean(m1.nrho[-1])), rel=1e-12)


@multidevice
def test_migration_actually_happens():
    """Guard against a decomposition that is accidentally a no-op."""
    p = min(4, len(jax.devices()))
    _info, _m1, _m2, _s1, _s2, mig = _run_pair(1, 64, p)
    assert float(jnp.mean(mig["sent"])) > 0.0


def test_slab_of_partitions_cells_exactly():
    """Every cell belongs to exactly one slab, and slabs are contiguous in x."""
    sp, grid, static, params, cstate, info = setup(dim=1, cells_per_side=64, npc=10)
    del sp, static, params, cstate
    x = jnp.linspace(0.0, float(grid.hi[0]), 1000, endpoint=False)[:, None]
    s = np.asarray(decomp.slab_of(x, grid, 4))
    assert s.min() == 0 and s.max() == 3
    assert np.all(np.diff(s) >= 0)  # contiguous and ordered in x
    assert set(np.unique(s)) == {0, 1, 2, 3}
    del info


def test_slab_of_rejects_indivisible_grid():
    sp, grid, *_ = setup(dim=1, cells_per_side=64, npc=10)
    del sp
    with pytest.raises(ValueError, match="does not divide"):
        decomp.slab_of(jnp.zeros((3, 1)), grid, 5)
