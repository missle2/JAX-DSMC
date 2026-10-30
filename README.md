# JAX-DSMC, branch `sparta-match`: sparta_jax

This branch holds only **sparta_jax**: a non-differentiable JAX forward that follows SPARTA's
circle deck routine for routine, used to separate XLA's cost from our own algorithmic
additions. The differentiable DSMC code (`jax_dsmc`, the adjoint, the mesh examples, the
docs) lives on `main`; nothing on this branch imports it.

Everything is described in [`sparta_jax/README.md`](sparta_jax/README.md): the deck, the
validation against SPARTA, the timings and the per-stage breakdown.

| Path | Contents |
|---|---|
| `sparta_jax/` | the simulation (`geometry.py`, `sim.py`, `rng.py`) and the SPARTA deck |
| `examples/sparta_jax/` | benchmark, profile, validation, emit check, figures; `results/` holds the outputs |
| `tests/test_sparta_jax.py` | the tests (`pytest tests`, about 15 s) |
| `scratch/sparta_jax/` | the work-accounting and sort experiments behind the README's tables (scripts use absolute paths; some compared against `main` and need a `main` checkout next to this one) |

Figures: `examples/sparta_jax/results/timings.png` (whole-run times against SPARTA and the
main code) and `examples/sparta_jax/results/step_breakdown.png` (where one step goes, stage by
stage and operation by operation, against SPARTA's own timers).
