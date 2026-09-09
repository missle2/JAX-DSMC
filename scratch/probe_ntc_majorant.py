"""Probe: how much does SPARTA's live vremax ratchet under-collide vs dsmcFoam+'s stale-max?

SPARTA (collide_vss.cpp:225-231) ratchets the majorant inside the candidate loop and tests
each candidate against the UPDATED value:
    vremax = max(vremax, vre);  accept if vre/vremax >= U
dsmcFoam+ (noTimeCounter.C:231-246) tests every candidate against the value captured BEFORE
the loop, publishing the new max only for the next step. Its inline rationale: N_pairs was
itself computed from the old majorant, so raising it mid-loop leaves you with too few
candidates AND a lower acceptance rate.

This sweeps majorant tightness to show when the difference actually bites. q is drawn from a
realistic VHS distribution -- q = C * |v_r|^(2-2w) with v_r the relative speed of two
Maxwellian particles -- rather than something uniform.

Drove PSEUDOCODE section 18.1 and plan decision 9.  Run: source ~/activate_jax_env.sh
"""

import numpy as np

rng = np.random.default_rng(0)
N_CELLS, N_ATT, OMEGA = 20000, 16, 0.75


def sample_q(n_cells, n_att, omega, rng):
    """q = sigma_T * v_r = |v_r|^(2-2*omega), v_r from two Maxwellians (unit thermal speed)."""
    vr = rng.normal(size=(n_cells, n_att, 3)) - rng.normal(size=(n_cells, n_att, 3))
    return np.linalg.norm(vr, axis=-1) ** (2.0 - 2.0 * omega)


def run(q, u, sigma0, live):
    """Return accepted-candidate count under the live or stale rule."""
    if not live:
        return int((u < np.clip(q / sigma0[:, None], 0.0, 1.0)).sum())
    acc, S = 0, sigma0.copy()
    for a in range(q.shape[1]):  # sequential ratchet
        S = np.maximum(S, q[:, a])
        acc += int((u[:, a] < np.clip(q[:, a] / S, 0.0, 1.0)).sum())
    return acc


q = sample_q(N_CELLS, N_ATT, OMEGA, rng)
u = rng.uniform(size=q.shape)
qmax = q.max()

print(f"omega={OMEGA}  cells={N_CELLS}  attempts/cell={N_ATT}")
print(f"q: mean={q.mean():.3f}  p99={np.percentile(q, 99):.3f}  max={qmax:.3f}\n")
print(f"{'Sigma0/max(q)':>14} {'ratchet fires':>14} {'stale':>9} {'live':>9} {'deficit':>9}")
print("-" * 60)

for frac in (2.00, 1.20, 1.00, 0.80, 0.60, 0.40, 0.20):
    sigma0 = np.full(N_CELLS, frac * qmax)
    fires = float((q > sigma0[:, None]).any(axis=1).mean())
    stale = run(q, u, sigma0, live=False)
    live = run(q, u, sigma0, live=True)
    print(
        f"{frac:>14.2f} {fires:>13.1%} {stale:>9d} {live:>9d} "
        f"{(stale - live) / max(stale, 1):>8.2%}"
    )

print("""
Reading it: when Sigma0 is a true majorant the ratchet never fires and the two rules agree
exactly -- as they must. The gap opens precisely when Sigma0 is too small, which is the
regime NTC actually runs in during warm-up and whenever the flow heats up, since vremax is
seeded from a coarse estimate and only ever adjusts upward.

Both rules converge to the same physics once vremax has settled. The stale rule is preferred
here not mainly for accuracy but because it makes Sigma constant within a step, which is the
fixed-majorant assumption [P2]'s score-term derivation is built on -- and it takes vremax out
of the attempt-scan carry.""")
