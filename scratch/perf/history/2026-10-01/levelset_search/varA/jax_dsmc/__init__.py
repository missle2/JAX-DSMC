"""JAX-DSMC — differentiable Direct Simulation Monte Carlo.

Spec: ``docs/PSEUDOCODE.md`` (revision 3) is the authoritative contract.

Importing this package enables ``jax_enable_x64`` **before any array is
created** (section 0.3): the VSS energy-conservation test asserts ~1e-14
relative agreement and the adjoint oracles compare against hand-coded recursions
to machine precision, neither of which f32 can deliver.  Every submodule repeats
the call so that importing one directly is equally safe.

Only the foundation modules are re-exported here.  The physics modules
(``grid``, ``sampling``, ``sort``, ``collide.*``) are imported explicitly by
their users, so that a failure in one does not poison ``import jax_dsmc``.
"""

import jax

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout

from jax_dsmc.constants import (  # noqa: E402  (must follow the x64 config call)
    EPS,
    EPS_D,
    EPS_REL_VR2,
    EPS_U,
    K_BOLTZ,
)
from jax_dsmc.rng import (  # noqa: E402  (foundational: no heavy deps, used everywhere)
    STAGE_ACCEPT,
    STAGE_INFLOW,
    STAGE_INIT,
    STAGE_MOVE,
    STAGE_NTC_ROUND,
    STAGE_PAIR,
    STAGE_SCATTER,
    STAGE_WALL,
    key_for,
    key_for_slot,
)
from jax_dsmc.safe import (  # noqa: E402
    safe_div,
    safe_log,
    safe_norm,
    safe_normalize,
    safe_pow,
    safe_recip_count,
    safe_sqrt,
)
from jax_dsmc.state import (  # noqa: E402
    CellTable,
    CollideState,
    Diagnostics,
    FlowParams,
    GridSpec,
    ParticleState,
    SimParams,
    SimStatic,
    SpeciesParams,
    check_fnum_uniform,
    num_cells_of,
)

__all__ = [
    "EPS",
    "EPS_D",
    "EPS_REL_VR2",
    "EPS_U",
    "K_BOLTZ",
    "STAGE_ACCEPT",
    "STAGE_INFLOW",
    "STAGE_INIT",
    "STAGE_MOVE",
    "STAGE_NTC_ROUND",
    "STAGE_PAIR",
    "STAGE_SCATTER",
    "STAGE_WALL",
    "CellTable",
    "CollideState",
    "Diagnostics",
    "FlowParams",
    "GridSpec",
    "ParticleState",
    "SimParams",
    "SimStatic",
    "SpeciesParams",
    "check_fnum_uniform",
    "key_for",
    "key_for_slot",
    "num_cells_of",
    "safe_div",
    "safe_log",
    "safe_norm",
    "safe_normalize",
    "safe_pow",
    "safe_recip_count",
    "safe_sqrt",
]
