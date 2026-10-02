"""Multi-device execution (spec section 20).

**What this module does and does not claim.**  It shards the particle arrays over
a one-dimensional device mesh and lets XLA insert the collectives.  That is a
*correct* parallelisation of the existing step -- the program is unchanged, so the
physics is unchanged -- but it is not the parallelisation a production DSMC code
ends up with, and the difference is worth stating before anyone reads a speed-up
number off it.

Per stage:

``move``, ``boundary``
    element-wise on particles.  Perfectly parallel, no communication.
``collide``
    per-cell, and cells are independent.  Parallel once each cell's particles are
    on one device.
``sample``
    ``segment_sum`` over cells; a reduction, so one all-reduce per step.
``sort``
    **the problem.**  ``build_table`` is a global ``argsort`` over all ``N``
    particles.  Sharded, that is a distributed sort -- all-to-all communication
    every step, and it is what will dominate as devices are added.

**Measured, not predicted.**  Sharding the particle axis and inspecting the
resulting shardings through one step gives::

    x            PartitionSpec('dev')     <- sharded
    table.cell   PartitionSpec('dev')     <- sharded (cell_of is element-wise)
    table.order  PartitionSpec()          <- REPLICATED
    table.counts PartitionSpec()          <- REPLICATED
    moments.temp PartitionSpec()          <- REPLICATED

The global ``argsort`` collapses the sharding, and everything downstream that
indexes through ``order`` is then replicated: **every device redundantly performs
the whole collision**.  On four (forced-host) devices the collide stage took
6.8 / 13.0 / 25.2 ms at 1 / 2 / 4 devices on a fixed problem -- growing in
proportion to the device count, which is the signature of replicated work sharing
one processor, not of a distributed computation.

So particle-axis sharding is not a parallelisation of DSMC at all.  The parallel
axis of the collision operator is the **cell** axis (section 8.6), and putting
work on it requires each device's particles to be co-located with its cells --
that is, spatial domain decomposition.  :mod:`jax_dsmc.decomp` implements it, and
this module is kept as the reference point that shows why it was necessary.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

jax.config.update("jax_enable_x64", True)

AXIS = "dev"


def device_mesh(n_devices: int | None = None) -> Mesh:
    """A 1-D mesh over the first ``n_devices`` devices.  Host-side setup."""
    devs = jax.devices()
    if n_devices is not None:
        if n_devices > len(devs):
            raise ValueError(f"asked for {n_devices} devices, {len(devs)} present")
        devs = devs[:n_devices]
    return Mesh(np.asarray(devs).reshape(len(devs)), (AXIS,))


def particle_sharding(mesh: Mesh) -> NamedSharding:
    """Shard along the particle axis; every other axis replicated."""
    return NamedSharding(mesh, PartitionSpec(AXIS))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec())


def shard_state(state, mesh: Mesh):
    """Place a :class:`ParticleState` with its leading axis split over the mesh.

    ``N`` must divide the device count exactly.  An uneven split is not merely
    slower: the per-device shapes differ, so the compiled program differs per
    device, which defeats the static-shape contract the whole solver rests on.
    """
    import dataclasses  # noqa: PLC0415

    sh = particle_sharding(mesh)
    n = state.x.shape[0]
    ndev = mesh.devices.size
    if n % ndev:
        raise ValueError(f"N = {n:,} does not divide over {ndev} devices")
    fields = {
        f.name: jax.device_put(getattr(state, f.name), sh)
        for f in dataclasses.fields(state)
        if getattr(state, f.name) is not None
    }
    return dataclasses.replace(state, **fields)


def shard_cells(x, mesh: Mesh):
    """Replicate a per-cell array.  Cell arrays are small next to particle arrays.

    ``C`` is typically ``N/npc``, so at ``npc = 50`` the cell arrays are 2 % of the
    particle arrays and replicating them costs little, while sharding them would
    force a gather in every ``segment_sum``.
    """
    return jax.device_put(x, replicated(mesh))
