"""Collision models (spec sections 7 and 8).

``vss.py`` is the *model* -- the rate, the acceptance probability, the score, and
the scattering law.  ``ntc.py`` is the *driver* -- how many pairs to attempt, which
pairs, and in what order.  The split is deliberate: adding a collision model (VHS,
Larsen-Borgnakke, chemistry) should not touch the driver, and changing the pairing
scheme should not touch the physics.
"""

from jax_dsmc.collide.ntc import (
    attempt_count,
    collide,
    decision_weights,
    pair_ntc,
    pair_perm,
    thermal_majorant,
    vremax_init,
)
from jax_dsmc.collide.vss import (
    acceptance_prob,
    collision_rate,
    frame_is_well_defined,
    log_h,
    make_prefactor,
    reduced_mass,
    scatter,
    species_params,
)

__all__ = [
    "acceptance_prob",
    "attempt_count",
    "collide",
    "collision_rate",
    "decision_weights",
    "frame_is_well_defined",
    "log_h",
    "make_prefactor",
    "pair_ntc",
    "pair_perm",
    "reduced_mass",
    "scatter",
    "species_params",
    "thermal_majorant",
    "vremax_init",
]
