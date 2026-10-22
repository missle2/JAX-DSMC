# Examples

One directory per case, each holding its driver, its figures, and its own notes.  Run everything
**from the repository root**.  The **current** work is `mesh/` (embedded surfaces, ensemble-adjoint
shape gradients, level-set shape optimisation, the CURC cluster study) and the benchmarks that
validate its gradient (`gradient_bench/`); the milestone cases M1-M7 below are **reference**
(passed gates, kept as regression evidence), and `p2_replication/`, `p3_replication/` and the
score-estimator variance study are **historical** cross-checks of the adjoint papers.

```bash
python3 -m examples.bkw --help
python3 -m examples.shock --help
python3 -m examples.turbulence --help
python3 -m examples.scaling --help
python3 -m examples.shock_adjoint --help
python3 -m examples.couette --help
python3 -m examples.gradient_bench --help
python3 -m examples.mesh.optimize --help     # flags = default_config() keys, foo_bar -> --foo-bar
```

| directory | status | what it demonstrates |
|---|---|---|
| `mesh/` | **current** | SPARTA-style embedded surfaces (cut cells, level-set bodies); forward duplicate of SPARTA's `circle` / `sphere`; shape gradients by the ensemble adjoint; shape optimisation (`optimize.py`); `mesh/cluster/` = the CURC Alpine study |
| `gradient_bench/` | current / reference | the estimator benchmark and the production ensemble adjoint (route (b)) on Couette, inlet shock and relaxation; artifact directory of the mesh gradient gates |
| `couette/` | reference (M4, M7) | wall boundary conditions: shear viscosity vs Chapman-Enskog, wall-parameter gradients |
| `shock/` | reference (M2) | periodic transport: shock formation, a Riemann problem against the exact Euler solution, **Bird §12.11 normal-shock structure vs Bird + Alsmeyer**, two-sided-inflow shock |
| `bkw/` | reference (M1) | 0-D relaxation against the exact Bobylev-Krook-Wu solution |
| `turbulence/` | reference (M2) | decaying turbulence; the **Taylor-Green viscosity gate (+0.38 % vs Chapman-Enskog)** |
| `shock_adjoint/` | reference / historical (M6) | `jax.grad` (pathwise) vs finite differences on Bird's shock, `D = 1, 2, 3`; the score-estimator variance study (`VARIANCE.md`, superseded) |
| `scaling/` | reference | weak and strong scaling over multiple devices |
| `p2_replication/`, `p3_replication/` | historical | scaled re-implementations of the adjoint papers' numerical tests |

Top-level scripts (`ensemble_mesh.py`, `ensemble_couette.py`, `ensemble_inlet_shock.py`,
`ensemble_relaxation.py`, `gradient_convergence.py`, `shape_planar_oracle.py`,
`validation_matrix.py`) are the gradient runners and helpers that `gradient_bench/` and `mesh/` use.

Figures land in `<case>/figures/`. Compiled programs are cached in `examples/.jax_cache/`
(gitignored), so the second run of any problem size skips compilation — measured 4.0 s → 0.8 s.

## Two things that will mislead you, in every case here

**Seeds.** Every quantity these examples report is a random variable. The BKW decay rate has a
spread of ~13 % at `N = 1e4`; a handful of seeds shows convincing trends that are pure noise. That
happened three separate times during development, at 1, 5 and 8 seeds. Use 16+, and read the
standard error, not the mean alone.

**Which mean free path.** Three different definitions appear in the shock literature and they
differ by up to 60 % for argon. Quoting a thickness against the wrong one produced a 24 % false
discrepancy against Bird here. `shock/README.md` §3 has the table; `PSEUDOCODE.md` §21.1 has the
derivation. Reproducing a single number quoted in a reference is not verification — two plausible
formulas both did.

**Noise floors, and the opposite conclusions they lead to.** Every per-cell quantity carries a
floor set by `npc`. On a *scalar* it is constant while the signal decays, so it biases fitted rates
and `turbulence/validate.energy_history` subtracts it. On a *spectrum* subtracting it is unsafe —
near the crossover `E - noise` fluctuates through zero and the surviving shells set the slope — so
the floor is plotted, never removed, and the floor is lowered instead (`--ensemble M` averages M
velocity fields; deterministic ICs make that exactly `M x npc`). After shell binning the floor also
*rises* as `k^(D-1)`, so it can pass for a cascade.

**The floor is usually not the binding limit.** The 1-D spectrum was rerun with 16 seeds, the floor
fell by exactly 16x, and the verdict did not move — two other things bound it. A single-mode initial
condition imprints its own symmetry on `E(k)` (measured: a factor-12 odd/even comb at shells 90x
clear of the floor, so use `--ic broadband` for any spectrum), and `Re = 61` puts dissipation at
`k ~ 10` against a peak at `k = 2`. Check the IC and `Re` first; particle count third.

**Resolution, and what a discrepancy means.** DSMC is not a continuum solver. Its shocks have a
real thickness of order the mean free path and its contact discontinuities really do diffuse —
those are physics, not error, and an exact Euler reference has neither. Comparisons here are made
on wave *positions* and plateau values, with the discontinuities masked out, and the per-case
notes say which is which.

## Output policy: fields, not particles

The default output is **rendered fields**, matching SPARTA's `dump image`. Writing particle state
per frame is the obvious thing to reach for in a particle method and it is the wrong default,
because its cost scales with `N` — the one number this project intends to push to `1e9`–`1e10`:

| output | bytes/frame | scales with |
|---|---|---|
| field PNG, 48×32×24 cells | **50 KB** | cell count only |
| particle `.vtp`, `N = 1e6`, already subsampled 50× | 1.7 MB | `N` |
| particle state, `N = 1e6`, unsubsampled | ~85 MB | `N` |

`jax_dsmc.viz.FieldMovie` is that path. In 3-D it writes **three PNGs per frame** — the `x = 0`,
`y = 0` and `z = 0` faces — because one slice of a 3-D flow says almost nothing about the other two
directions. `viz.PLANES_MID` selects mid-planes instead and `viz.Plane(axis, index=...)` names any
other axis-aligned cut. Colour limits are held fixed across a series; per-frame autoscaling makes
an animation unreadable, because the colours stop meaning a value.

Particles remain available via `output.VTKSeries.add_particles` — a separate, opt-in call with a
frame-stable subsample.

**A finite difference is only as good as its flip guard.** At frozen randomness `J` is a *step
function* of any parameter entering a discrete decision, so an FD estimate means nothing unless
the two evaluations took the same discrete path. `Diagnostics.decision_checksum` fingerprints
that path, and it had to cover more than the obvious: the specular **boundary fold index** (whose
parity flips `v`'s sign, and which is invisible in the cell index because both sides of the wall
fold into cell 0), `scatter`'s **degenerate-frame branch**, and — the one that actually bit —
**no decision may be weighted by zero**, or an entire cell drops out of the fingerprint.
`shock_adjoint/README.md` and `PSEUDOCODE.md` §13.4.1 have the story; the symptom was FD = 346
against a true 127 with the guard reporting no flips.
