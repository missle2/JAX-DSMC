"""Paired ensemble-gradient equivalence, distinct from frozen-history FD checks."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Acceptance:
    absolute: float = 0.02
    relative: float = 0.05
    alpha: float = 0.05
    comparisons: int = 1


DEFAULT_ACCEPTANCE = Acceptance()


def paired_report(gradients, finite_differences, *, scale=1.0, acceptance=DEFAULT_ACCEPTANCE):
    """Predeclared Student-t equivalence interval; no optional-stopping rule.

    Rows are independent seeds, not correlated snapshots within one simulation.
    The caller must separately establish FD-step and discretization convergence.
    """
    g, f = np.asarray(gradients, float) * scale, np.asarray(finite_differences, float) * scale
    if g.shape != f.shape or g.ndim < 1 or len(g) < 2:
        raise ValueError("Matching arrays with at least two independent seeds required")
    if not np.isfinite(g).all() or not np.isfinite(f).all():
        raise ValueError("Invalid gradients/finite differences cannot be dropped")
    d = g - f
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d))
    critical = student_t.ppf(1 - acceptance.alpha / (2 * acceptance.comparisons), len(d) - 1)
    lo, hi = d.mean(axis=0) - critical * se, d.mean(axis=0) + critical * se
    band = acceptance.absolute + acceptance.relative * np.abs(f.mean(axis=0))
    equivalent = (lo >= -band) & (hi <= band)
    different = (lo > band) | (hi < -band)
    status = np.where(equivalent, "equivalent", np.where(different, "different", "inconclusive"))
    return {
        "seeds": len(g),
        "gradient_mean": g.mean(axis=0).tolist(),
        "fd_mean": f.mean(axis=0).tolist(),
        "difference_mean": d.mean(axis=0).tolist(),
        "difference_se": se.tolist(),
        "interval_low": lo.tolist(),
        "interval_high": hi.tolist(),
        "equivalence_band": band.tolist(),
        "status": status.tolist(),
        "convergence_established": False,
    }


def adjacent_refinement_report(estimates, *, scale=1.0, acceptance=DEFAULT_ACCEPTANCE):
    """Compare paired estimates at adjacent refinements; finest is the last axis."""
    x = np.asarray(estimates)
    if x.ndim < 2 or x.shape[-1] < 3:
        raise ValueError("At least three refinement levels are required")
    return [
        paired_report(x[..., i], x[..., i + 1], scale=scale, acceptance=acceptance)
        for i in range(x.shape[-1] - 1)
    ]
