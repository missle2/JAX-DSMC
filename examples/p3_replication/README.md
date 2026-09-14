# P3 reference experiments

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
