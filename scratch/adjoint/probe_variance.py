"""Does our score estimator obey [P2]'s scaling laws, at [P2]'s particle count?

[P2] section 6 does report this variance and budgets for it:

* `main.tex:1425` -- average `M_s = 100` i.i.d. runs for *both* the AD and FD gradient;
* `main.tex:1431` -- random error is `O((sqrt(N M_s) |Delta m|)^-1)`, and `Delta m = 0.1`
  (a **10 percent** step) is "nearly optimal for N = 1e6 Maxwellian particles, M_s = 100";
* `main.tex:1502` -- both standard deviations grow as `O(sqrt(M))` in the step count.

So the question is not "why is ours noisy" but **"is ours noisy in the way theirs is,
and does it come good at their budget"**.

**The objective has to be [P2]'s.**  Their `phi` is `T_y`.  Ours must not be a
*collision invariant*: VSS conserves pair energy exactly and the specular wall
conserves it too, so `sum |v|^2` cannot depend on any collision-model parameter at
all -- the pathwise gradient of it with respect to `alpha` measures **1.5e-13**.
That is a fine zero-check (section 0 below) and useless as a variance signal.
`v_y^2` is not invariant: collisions redistribute energy between directions, which is
exactly why [P2] chose `T_y`.

**How N is grown here matters.**  Raising `npc` raises the NTC attempt count in
proportion (`n ~ 0.5 npc (npc-1) Sigma dt fnum / V` with `fnum ~ 1/N`, so `n ~ npc`),
and `A_max` is the *trip count of a scan* -- 25 000 slots would be intractable to
compile, let alone differentiate.  Widening the **transverse** direction instead adds
cells and particles at fixed `npc`, so `A_max` stays put.  A normal shock is a 1-D
structure; transverse cells are independent samples of it, which is the same
averaging [P2] gets from a larger 0-D ensemble.

Unlike mode A, none of this needs a flip-free step: `Delta m = 0.1` *wants* decisions
to flip.  The small case in `examples/shock_adjoint` is small because a fixed-key
finite difference has no valid step size above `N C M ~ 1e7` (section 13.4.1); that
constraint does not apply here, and carrying it over was a mistake.
"""

import functools
import sys
import time

sys.path.insert(0, ".")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa  # noqa: E402
from jax_dsmc import adjoint  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide import vremax_init  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.initialize import initial_state  # noqa: E402
from jax_dsmc.state import CollideState  # noqa: E402

print("devices:", jax.devices(), flush=True)
TH = sa.reference_theta()
NAMES = ("temp", "ux", "alpha", "diam")
DELTA_M = 0.1  # [P2] main.tex:1431


def value(theta, base_key, cs, phi_kind, scored):
    flow, sp, grid, params, static, lam = sa.build(theta, cs)
    st = initial_state(base_key, flow, grid, sp, static)
    c = num_cells(grid)
    cst = CollideState(
        vremax=vremax_init(
            jnp.full((c,), theta["temp"]),
            sp.omega[0, 0],
            sp.tref[0, 0],
            sp.mr[0, 0],
            sp.prefactor[0, 0],
            c,
            1,
        )
    )
    if phi_kind == "invariant":
        phi = adjoint.phi_kinetic  # a collision invariant -- zero-check only
    elif phi_kind == "windowed":
        phi = functools.partial(
            adjoint.phi_windowed,
            inner=functools.partial(adjoint.phi_power, component=1, power=2),
            axis=0,
            center=cs.window_mfp * lam,
            width=cs.width_mfp * lam,
        )
    else:  # "ty" -- [P2]'s objective
        phi = functools.partial(adjoint.phi_power, component=1, power=2)
    j, _ = adjoint.terminal_objective(
        st,
        cst,
        grid,
        sp,
        params,
        static,
        base_key,
        phi,
        rho=1.0,
        bc=BC.SPECULAR,
        baseline="mean",
        scored=scored,
    )
    return j


def _run_batched(fn, keys, chunk):
    """Apply a vmapped `fn` over `keys`, halving the batch on out-of-memory.

    The reverse pass's peak is many times the particle arrays themselves, and it is
    not worth predicting: a 512-key batch at `N = 2000` asks XLA for 4.0 GiB, and a
    16 000-particle case asked for 8.8 GiB.  Catching the failure and backing off is
    more robust than any formula, and it reports the batch it settled on.
    """
    while chunk >= 1:
        try:
            f = jax.jit(jax.vmap(fn, in_axes=(None, 0)))
            outs = [
                jax.block_until_ready(f(TH, keys[i : i + chunk]))
                for i in range(0, len(keys), chunk)
            ]
            return (
                jax.tree.map(lambda *xs: jnp.concatenate(xs), *outs) if len(outs) > 1 else outs[0]
            ), chunk
        except Exception as exc:  # noqa: BLE001 -- XLA raises a plain RuntimeError
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
            print(f"      (OOM; retrying at chunk={chunk})", flush=True)
    return None, 0  # caller decides; the reverse pass does not fit on this card


#: Measured peak reverse-pass memory, bytes per (particle x key).  From the observed
#: failure: 8.58 GiB requested at chunk=125, N=16000 -> 8.58e9/(125*16000) = 4290.
#: Used to pick a starting batch instead of discovering it by repeated compiles.
BYTES_PER_PARTICLE_KEY = 4290
GPU_BUDGET = 2.6e9


def auto_chunk(cs):
    return max(1, int(GPU_BUDGET / (BYTES_PER_PARTICLE_KEY * cs.n_part)))


def grads(cs, phi_kind, scored, keys, chunk=None):
    fn = functools.partial(jax.grad(value), cs=cs, phi_kind=phi_kind, scored=scored)
    return _run_batched(fn, keys, auto_chunk(cs) if chunk is None else chunk)[0]


def values(cs, phi_kind, keys, theta, chunk=256):
    fn = functools.partial(value, cs=cs, phi_kind=phi_kind, scored=True)
    while chunk >= 1:
        try:
            f = jax.jit(jax.vmap(fn, in_axes=(None, 0)))
            outs = [
                np.asarray(jax.block_until_ready(f(theta, keys[i : i + chunk])))
                for i in range(0, len(keys), chunk)
            ]
            return np.concatenate(outs)
        except Exception as exc:  # noqa: BLE001
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            chunk //= 2
    raise RuntimeError("forward does not fit")


def case_of(n_transverse, npc=50, steps=16):
    """D = 2, widened transversely.  `A_max` is calibrated but stays ~constant."""
    return sa.calibrate(
        sa.case(2, npc=npc, n_steps=steps, n_x=40, transverse=n_transverse, a_max=256)
    )


def report(cs, phi_kind, keys, label, chunk=None):
    gp = grads(cs, phi_kind, False, keys, chunk)
    gf = grads(cs, phi_kind, True, keys, chunk)
    if gp is None or gf is None:
        print(f"\n  {label}: reverse pass does not fit on this device -- SKIPPED", flush=True)
        return None
    print(f"\n  {label}   ({len(keys)} keys)", flush=True)
    print(
        f"    {'param':>7} {'pathwise mean':>15} {'full mean':>15} {'sd(full)':>12} "
        f"{'sd(score)':>12} {'SE %':>8}"
    )
    rec = {}
    for name in NAMES:
        a, b = np.asarray(gp[name]), np.asarray(gf[name])
        sd = b.std(ddof=1)
        rec[name] = (b.mean(), sd)
        se_pct = 100 * sd / np.sqrt(len(b)) / abs(b.mean()) if b.mean() else float("nan")
        print(
            f"    {name:>7} {a.mean():15.5e} {b.mean():15.5e} {sd:12.3e} "
            f"{(b - a).std(ddof=1):12.3e} {se_pct:8.1f}"
        )
    return rec


NK = 128
keys = jax.random.split(jax.random.PRNGKey(101), NK)

print("\n=== 0. zero-check: sum |v|^2 is a collision invariant ===")
print("  VSS conserves pair energy and the specular wall conserves it, so dJ/d(any")
print("  collision parameter) must be EXACTLY zero, pathwise and in expectation.")
cs0 = case_of(8)
k0 = jax.random.split(jax.random.PRNGKey(3), 32)
gp0 = grads(cs0, "invariant", False, k0)
gf0 = grads(cs0, "invariant", True, k0)
for n in ("alpha", "diam", "omega"):
    a, b = np.asarray(gp0[n]), np.asarray(gf0[n])
    print(
        f"    {n:>7}  max|pathwise| {np.abs(a).max():.3e}    "
        f"scored {b.mean():.3e} +- {b.std(ddof=1) / np.sqrt(len(b)):.3e}",
        flush=True,
    )

print("\n=== A. sd vs N, objective = T_y ([P2]'s).   Predicted: sd ~ N^-0.5 ===")
rows = {}
for nt in (2, 8, 32, 128):
    cs = case_of(nt)
    t0 = time.time()
    rec = report(
        cs,
        "ty",
        keys,
        f"transverse={nt}  C={cs.n_cells}  N={cs.n_part}  A_max={cs.a_max}",
    )
    if rec is not None:
        rows[cs.n_part] = rec
    print(f"    ({time.time() - t0:.0f}s)", flush=True)

print("\n  fitted exponent of sd ~ N^p    ([P2] predicts p = -0.5)")
ns = np.array(sorted(rows))
for name in NAMES:
    sd = np.array([rows[n][name][1] for n in ns])
    p_fit = np.polyfit(np.log(ns), np.log(sd), 1)[0]
    print(f"    {name:>7}  p = {p_fit:6.3f}    sd: " + " ".join(f"{v:9.2e}" for v in sd))

print("\n=== B. at [P2]'s budget: N ~ 1e6, M_s = 100, Delta m = 0.1 ===")
# N ~ 1e6 does not fit the REVERSE pass on a 4 GB card even one key at a time
# (~4290 bytes per particle-key), so this runs at the largest N that does.  The
# forward side is far cheaper and is not the constraint.
cs = case_of(128)
k100 = jax.random.split(jax.random.PRNGKey(7), 100)
ch = auto_chunk(cs)
gf = grads(cs, "ty", True, k100, ch)
print(
    f"  N = {cs.n_part}, C = {cs.n_cells}, M_s = 100  ->  N*M_s = {cs.n_part * 100:.1e} "
    f"([P2] uses 1e8)"
)
print(f"  {'param':>7} {'AD mean':>15} {'AD SE':>12} {'FD mean':>15} {'FD SE':>12} {'|z|':>7}")
for name in NAMES:
    b = np.asarray(gf[name])
    ad, ad_se = b.mean(), b.std(ddof=1) / np.sqrt(len(b))
    jp = values(cs, "ty", k100, sa.perturb(TH, name, DELTA_M), max(ch * 8, 8))
    jm = values(cs, "ty", k100, sa.perturb(TH, name, -DELTA_M), max(ch * 8, 8))
    d = (jp - jm) / (2.0 * DELTA_M * float(TH[name]))
    fd, fd_se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    z = (fd - ad) / np.hypot(fd_se, ad_se)
    print(
        f"  {name:>7} {ad:15.5e} {ad_se:12.3e} {fd:15.5e} {fd_se:12.3e} {abs(z):7.2f}",
        flush=True,
    )

print("\n=== C. does the spatial window cost variance? ===")
cs = case_of(32)
for kind in ("ty", "windowed"):
    report(cs, kind, keys, f"phi = {kind}")

print("\n=== D. and the step count?   ([P2]: sd ~ sqrt(M)) ===")
for steps in (4, 8, 16, 32):
    cs = case_of(32, steps=steps)
    gf = grads(cs, "ty", True, keys)
    print(
        f"  M={steps:>3}: "
        + "  ".join(f"{n}: sd {np.asarray(gf[n]).std(ddof=1):.3e}" for n in NAMES),
        flush=True,
    )
