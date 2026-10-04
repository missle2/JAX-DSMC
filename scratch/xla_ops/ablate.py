"""Upper bounds for algorithmic options: the one-core forward's steady-state step time with
one stage removed or made trivial (monkeypatched here, the library untouched).  The
physics of an ablated run is wrong on purpose -- the number is what that stage costs at
most, i.e. the most a restructuring of it could save.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false PYTHONPATH=. \
        taskset -c 3 python3 scratch/xla_ops/ablate.py VARIANT [polygon|levelset]

Variants:
  base            unchanged
  no_entrant_move entrants placed by x += v frac, no tracker call of their own (inject
                  track=False): the bound for moving entrants inside the main move
  cheap_sampler   entrant velocities from plain normals (no Halley inverse CDF)
  no_temperature  cell temperature a constant (no per-step pass)
  slotsN          N particle slots instead of 60000 (the same run while it fits)
  no_body         the embedded body removed (mesh=None): the bound for every
                  surface-search / routing change
"""

import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/perf")
sys.path.insert(0, "examples/mesh")
import bench_forward as B  # noqa: E402
import forward as F  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import inflow, initialize, sample, simulate  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

def _open_run(
    state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, step0, capture,
    simulate,
):
    return simulate.run(
        state, cstate, grid, sp, params, static, key, faces=faces, wall=wall, res=res,
        capacity=capacity, emit_moments="none", step0=step0, mesh=mesh, capture=capture,
    )


variant = sys.argv[1]
body = sys.argv[2] if len(sys.argv) > 2 else "polygon"

if variant == "no_entrant_move":
    _inject = inflow.inject

    def _untracked(*a, **k):
        k["track"] = False
        return _inject(*a, **k)

    inflow.inject = _untracked
elif variant == "cheap_sampler":

    def _normals(key, n, res, axis, side, mass):
        f = 2 * axis + side
        scale = jnp.sqrt(1.380649e-23 * res.temp[f] / mass)
        k1, k2 = jax.random.split(key)
        v = res.vstream[f] + scale * jax.random.normal(k1, (n, 3))
        # the flux-weighted (Rayleigh) normal speed of a resting gas, plus the stream
        u = jax.random.uniform(k2, (n,), minval=1e-12)
        vn = scale * jnp.sqrt(-2.0 * jnp.log(u)) + jnp.abs(res.vstream[f][axis])
        v = v.at[:, axis].set(vn * (1.0 if side == 0 else -1.0))
        return v, jnp.ones(n, bool)

    inflow.sample_face_velocities = _normals
elif variant == "no_temperature":
    sample.cell_temperature = lambda state, table, grid, sp: jnp.full(
        table.counts.shape, F.TEMP
    )
elif variant.startswith("slots"):  # e.g. slots52000: fewer padded slots, same physics
    F.CASES["circle"]["capacity"] = int(variant[5:])
elif variant != "base" and variant != "no_body":
    raise SystemExit(f"unknown variant {variant}")

grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(body)
run = EM._run
if variant == "no_body":
    # the surface tallies need a body: an open box without them (their cost is in the
    # bound too, a few per cent of the tracker at most)
    mesh = None
    run = jax.jit(
        functools.partial(_open_run, simulate=simulate),
        static_argnames=("faces", "static", "capacity", "capture"),
    )
flow = FlowParams(
    nrho=jnp.float64(F.NRHO), temp=jnp.float64(F.TEMP), vstream=jnp.array([F.VSTREAM, 0.0, 0.0])
)
key = jax.random.key(0)
best = []
for _rep in range(2):
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    times = []
    for k0 in range(0, B.STEPS, B.BLOCK):
        t0 = time.perf_counter()
        (state, cstate), _ = run(
            state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, k0,
            False,
        )
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)
    best.append(times)
t = np.min(np.array(best), axis=0)
steady = np.min(t[-4:]) / B.BLOCK * 1e3
print(
    f"{variant:16s} {body:8s} steady {steady:.2f} ms/step  run {t[1:].sum() + t[0]:.2f} s  "
    f"alive {int(state.alive.sum())}"
)
