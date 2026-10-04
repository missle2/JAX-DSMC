# Adjoint fit cost (2026-10-02)

The value backwards' local quadratic fits (`ensemble_device.field_query`), measured on the
laptop GPU (RTX 3050 Ti, f64 at 1/64 rate).

* `seed_basis.py` -- one optimiser seed's two value backwards (`EM._backwards`, optc at
  fnum x 4, virtual 1 x 2), interleaved per basis.  Collisionless (`KN=0`): every fit is
  weighted by zero attempts, lambda identical across bases -- the fits are pure cost there
  (a skip-when-zero is ruled out: memory "keep collision path exercised").  Collisional
  (`KN=0.5`, `seed_basis_kn0.5.log`): quadratic_full 93.4 / 92.4 s, affine 67.7 / 67.9 s
  -> affine 1.37x cheaper end to end.
* `bench_fit.py` -- one 4096-query chunk (P 256, k 32, D 2): LU 5.87 ms (front 2.0, assembly
  2.5, solve 1.2), unrolled Cholesky 5.15, sqrt(w) two-operand Gram + unrolled 4.99, affine
  4.6 (LU) / 2.75 (upper-triangle Gram + unrolled).  Assembly is not flop-bound: ~2.5 ms
  at 6 and at 14 columns.
* `bench_rhs.py` -- R value fields through one fit (one k-nearest search, one normal matrix,
  one factorisation): einsum + LU 5.88 (R=1) / 8.38 (R=2); weighted-sum right-hand side +
  shared unrolled Cholesky 5.21 (R=1) / 5.61 (R=2), i.e. 2.8 ms per field against 5.88.
  Agreement with separate LU fits 2.6e-13 (CPU check; the bench's printed "diff" compares
  different random data and is meaningless).

The two value backwards of a seed (hit rewards, timing rewards) are linear in their rewards
and share record, key, partners and node sets, so one backward with lambda f64[N, 2] gives
the same results as two, with every fit carrying two right-hand sides.

## Built (2026-10-02): the two-column backward

`ensemble_device.backward_virtual` with `λ f64[N, R]`, `field_query` value columns and the
unrolled Cholesky `_spd_solve`; `examples/ensemble_mesh._backwards` as one `R = 2` call
(PSEUDOCODE 9.7.1, HANDOFF 1.10).

* `gate_columns.py save|compare` (KN=0.5, PULL=1x2): two backwards 90.5 -> 43.3 s; lam vs
  the old code max |diff| / rms 2.9e-7 (`LU=1`: the column sweep with the old LU, 1.3e-7 --
  the gap is rounding amplified by ill-conditioned fits, zero at the last step, 9e-10 one
  step back, then a plateau).  Reference: `gate_kn0.5_1x2.npz`.
* `scratch/perf/regress.py --compare final --save columns`: worst gradient term 3.65e-9,
  objective bitwise; warm seed backward 65 -> 29 s, total 99 -> 64 s.
* Tests: `tests/test_ensemble_device.py` (solver vs LU, field columns, sweep columns).
