"""Why is the reverse pass 17x the forward, when the papers report ~0.6x?

Three candidate explanations, each separable by measurement:

1. **Launch-bound forward.**  At N = 2000 the forward pass is a stream of tiny
   kernels and is latency-limited, not compute-limited.  The reverse pass has more
   work per kernel, so the *ratio* is inflated by smallness.  -> ratio should fall
   as N grows.
2. **Rematerialisation.**  `remat_policy="nothing_saveable"` recomputes the whole
   step body in the backward pass.  -> a cheaper policy should cut the ratio.
3. **Structural.**  [P1]'s 0.6x is a *hand-written* backward that replays only the
   recorded collisions: no sort, no moments, no rejected-candidate scatter, just
   stored 6x6 rotations.  AD must transpose everything the forward did, including
   work that contributes nothing.  -> an irreducible floor, visible as the ratio
   plateauing above 1 even when (1) and (2) are exhausted.
"""

import sys
import time

sys.path.insert(0, ".")
import dataclasses  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from examples import shock_adjoint as sa  # noqa: E402

print("devices:", jax.devices(), flush=True)
TH = sa.reference_theta()


def timed(fn, *args, reps=3):
    jax.block_until_ready(fn(*args))
    t = []
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(fn(*args))
        t.append(time.time() - t0)
    return min(t)


def ratio(cs, key, label):
    fwd = timed(sa._jit_value, TH, cs, key, True)
    rev = timed(sa._jit_value_and_grad, TH, cs, key, True)
    print(
        f"  {label:<34} fwd {1e3 * fwd:8.2f} ms   fwd+rev {1e3 * rev:8.2f} ms   "
        f"ratio {rev / fwd:6.2f}",
        flush=True,
    )
    return rev / fwd


print("\n=== 1. ratio vs problem size (one key, D=1, A_max calibrated) ===")
for npc, n_x, steps in ((20, 8, 4), (50, 40, 16), (100, 80, 16), (200, 160, 16), (400, 160, 16)):
    cs = sa.calibrate(sa.case(1, npc=npc, n_steps=steps, n_x=n_x, transverse=2, a_max=64))
    ratio(cs, jax.random.PRNGKey(0), f"N={cs.n_part:>7} C={cs.n_cells:>4} A={cs.a_max:>5} M={steps}")

print("\n=== 1b. ratio vs A_max at FIXED N -- the scan-residual hypothesis ===")
# Reverse-mode through `lax.scan` must keep the carry at EVERY iteration.  The carry
# here is the whole (N+1, 3) velocity array, so the backward pass touches A_max times
# more memory than the forward, which overwrites in place.  If that is the dominant
# cost, the ratio should grow roughly linearly in A_max at fixed N.
fixed = sa.calibrate(sa.case(1, npc=100, n_steps=8, n_x=80, transverse=2, a_max=64))
for mult in (1, 2, 4, 8):
    cs = dataclasses.replace(fixed, a_max=fixed.a_max * mult)
    ratio(cs, jax.random.PRNGKey(0), f"A_max={cs.a_max:>5} (x{mult}), N={cs.n_part}")

print("\n=== 2. ratio vs remat policy (N=8000, D=1) ===")
base = sa.calibrate(sa.case(1, npc=100, n_steps=16, n_x=80, transverse=2, a_max=64))
for policy in (
    "nothing_saveable",
    "everything_saveable",
    "checkpoint_dots",
    "checkpoint_dots_with_no_batch_dims",
):
    if not hasattr(jax.checkpoint_policies, policy):
        print(f"  {policy:<34} (not available in this JAX)")
        continue
    cs = dataclasses.replace(base, remat=policy)
    try:
        ratio(cs, jax.random.PRNGKey(0), policy)
    except Exception as exc:  # noqa: BLE001
        print(f"  {policy:<34} FAILED: {type(exc).__name__}: {str(exc)[:90]}")

print("\n=== 3. how much of the forward is each stage (N=8000, D=1) ===")
from jax_dsmc import sample, sort  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.collide import vremax_init  # noqa: E402
from jax_dsmc.collide.ntc import collide  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.initialize import initial_state  # noqa: E402
from jax_dsmc.move import move  # noqa: E402
from jax_dsmc.state import CollideState  # noqa: E402

flow, sp, grid, params, static, lam = sa.build(TH, base)
key = jax.random.PRNGKey(0)
st = initial_state(key, flow, grid, sp, static)
C = num_cells(grid)
cst = CollideState(
    vremax=vremax_init(
        jnp.full((C,), TH["temp"]), sp.omega[0, 0], sp.tref[0, 0], sp.mr[0, 0],
        sp.prefactor[0, 0], C, 1,
    )
)
tab = sort.build_table(st, grid)
mom = sample.moments(st, tab, grid, params, sp)

f_move = jax.jit(lambda s: move(s, grid, params.dt, BC.SPECULAR))
f_sort = jax.jit(lambda s: sort.build_table(s, grid))
f_mom = jax.jit(lambda s, t: sample.moments(s, t, grid, params, sp))
f_col = jax.jit(
    lambda s, t, c, mt: collide(s, t, c, grid, sp, params, static, mt, 0, key)
)
print(f"  move     {1e3 * timed(f_move, st):8.3f} ms")
print(f"  sort     {1e3 * timed(f_sort, st):8.3f} ms")
print(f"  moments  {1e3 * timed(f_mom, st, tab):8.3f} ms")
print(f"  collide  {1e3 * timed(f_col, st, tab, cst, mom.temp):8.3f} ms")
print(f"  (A_max = {static.A_max}, C = {C}, N = {static.N})")
