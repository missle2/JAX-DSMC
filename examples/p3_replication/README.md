# P3 reference experiments

Status: **reference / historical** (2026-09).  Scaled re-implementations of the algorithms of the
P3 paper (host NumPy/SciPy retained-ensemble inlet reference, JAX wall solver) used to
cross-check the production ensemble adjoint (route (b)); nothing in the optimisation path
depends on them.  Not maintained beyond the commands below.

See [implementation changes and validation limits](../../docs/ADJOINT_CORRECTIONS.md).
These are scaled checks of P3's algorithms; the saved runs do not reproduce the
paper's large-population confidence intervals.

Activate `/home/aakash/jax_dsmc/activate_jax_env.sh`, then run from the
repository root. The wall solver uses JAX; the retained-ensemble inlet reference
uses NumPy/SciPy on the host.

```sh
python -m examples.p3_replication.walls --case couette --localize --particles 2000 --seeds 8
python -m examples.p3_replication.walls --case heat --localize --particles 2000 --seeds 8
OPENBLAS_NUM_THREADS=1 python -m examples.p3_replication --particles 4000 \
  --retained 512 --source-samples 1995 --partners 8 --seeds 4 \
  --fd-step .1 --reconstruction ballistic
```

Use `--output path.json` to select an artifact destination. Wall runs report both
deterministic and regularized finite differences. Inlet finite differences hold
nominal injection counts fixed and share the initial sample, collision seeds and
source seeds. The inlet API returns all six temperature derivatives; the CLI
compares the two normal-temperature derivatives used in P3 Test 3.

`--localize` applies P3's hard three-sigma proximity selection. The default wall
regularization applies to every particle. `--score-mode paper` selects the
terminal-label comparison estimator; `trajectory` scores the complete objective.
Neither differentiates hard cell memberships.

`--reconstruction affine` is the direct local-affine inlet field from the paper.
`ballistic` fits the residual after subtracting known collisionless transport.
Reservoir and reconstruction dependence remain visible in the saved results.

## Route (b) on P3's model, and the paper's Test-3 value

`jax_dsmc/ensemble_particle.py` is the particle-carried quadrature of the same ensemble
adjoint (review §5.1–5.2): no retained frames, interpolation only at the collided
partners' ghosts.  Its verification experiments are one CLI:

```sh
python3 -m examples.p3_replication.route_b_experiments two-step --seeds 240 --a-seeds 16
python3 -m examples.p3_replication.route_b_experiments compare --n0 2000 --seeds 16
python3 -m examples.p3_replication.route_b_experiments scale --n0 16000 --seeds 32
python3 -m examples.p3_replication.route_b_experiments scale --n0 16000 --seeds 128 --seed-start 32
python3 -m examples.p3_replication.route_b_experiments fd --n0 16000 --seeds 1600 --workers 10
python3 -m examples.p3_replication.route_b_report        # figures/route_b_agreement.png
```

Artifacts live in `results/route_b/`.  **The paper's 0.1517 is not a target for this
transcription**: the transcribed forward differs from the paper's at the inlet (final
population +1.7 %, wall-cell temperature 4.2 vs 3.73 — review §4.12), and the
transcription's own gradient is 0.144–0.147 by a 1600-seed finite difference and by both
adjoint routes.  `report.py` (`figures/p3_inlet_convergence.png`) covers the retained-
frame budget sweep; the affine-vs-ballistic gap there is the interpolant's Jensen bias
(`route_b_experiments recon-bias`).
