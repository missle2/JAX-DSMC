"""Checked host boundary for pure compiled solver results.

Device kernels return diagnostics without raising inside JIT. Applications must
check complete runs before reporting profiles, objectives or gradients.
"""

import numpy as np


def validate_inputs(sp, params, static):
    if static.S != 1 or static.G != 1 or np.asarray(sp.mass).size != 1:
        raise ValueError("Only one monatomic species/group is supported")
    weights = np.asarray(params.fnum)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(weights != weights.flat[0]):
        raise ValueError("Finite positive uniform fnum is required; reweighting is not implemented")
    if static.majorant_mode == "certified":
        omega = np.asarray(sp.omega)
        if np.any((omega < 0.5) | (omega > 1)):
            raise ValueError("Certified VSS bound requires 0.5 <= omega <= 1")


def require_valid(diagnostics, migration=None, *, gradients=False):
    failures = {}
    for name in (
        "attempts_clipped",
        "majorant_violations",
        "wall_recrossed",
        "inflow_shortfall",
        "configuration_invalid",
    ):
        value = np.asarray(getattr(diagnostics, name))
        if not np.isfinite(value).all() or np.any(value != 0):
            failures[name] = float(np.sum(value))
    margin = np.asarray(diagnostics.min_majorant_margin)
    if np.isnan(margin).any() or np.any(margin < 1):
        failures["min_majorant_margin"] = float(np.min(margin))
    if migration is not None:
        for name in ("overflow_send", "overflow_recv", "far_jumps", "wrong_owner"):
            if name in migration and np.any(np.asarray(migration[name]) != 0):
                failures[name] = int(np.sum(migration[name]))
    if gradients and np.any(np.asarray(diagnostics.count_support_boundary) != 0):
        failures["count_support_boundary"] = int(np.sum(diagnostics.count_support_boundary))
    if failures:
        raise RuntimeError(f"Invalid DSMC trajectory: {failures}")
