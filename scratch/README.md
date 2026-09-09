# scratch/

Throwaway probes and one-off numerical experiments. Not part of the package, not imported by
anything, no stability guarantees. Kept in the repo (rather than a temp dir) so the reasoning
behind a design decision can be re-run and checked later. Excluded from `ruff` lint and format.

Promoted, curated checks live in `docs/phase0_verification.py` — that one stays green and becomes
the unit-test suite.

Run everything with the project env: `source ~/activate_jax_env.sh`

## Design decisions these settled

| Probe | Question | Outcome |
|---|---|---|
| `probe_ntc_majorant.py` | Does SPARTA's live `vremax` ratchet under-collide vs. dsmcFoam+'s stale-max rule? | **Zero** difference when Σ is a true majorant; 9–21% only while Σ climbs from its seed. Moved the argument for the stale rule onto [P2]'s fixed-Σ requirement rather than accuracy. §18.1 |
| `probe_A_matrix_rank.py` | Is [P1]'s `B = Aᵀ = A⁻¹` literally true? | No — `A` is rank-4 singular, `A⁻¹` does not exist. `B = Aᵀ` holds; the involution is a property of the *map*. §17.1 |
| `probe_grad_claim1*.py` | Does `n = floor(f)` drop the `dt`/`fnum`/`volume` gradients? | **Yes, exactly zero.** Fixed by stochastic rounding + `log P(n\|f)` score. Baseline gives 84× variance reduction. §8.1 |
| `probe_grad_claim2.py` | Does `move→collide` give the same adjoint recursion as [P3]? | No — transport term sits *outside* `B_k`, not inside. §0.2, §12.1 |
| `probe_scan_clobber.py`, `probe_scan_mechanism.py` | Do inactive cells corrupt the attempt scan? | **Yes, 176/300 runs broke momentum.** Mechanism is the stale second `.set`, not a duplicate-index race. §8.4.1 |
| `adjoint/probe_split.py` | Which parameters reach `J` pathwise, and which only through a frozen decision? | `temp`/`vstream`/`alpha` pathwise; `diam`/`omega`/`tref`/`nrho` **exactly 0.0** pathwise — score-only. Decides which FD mode applies. §23.1 |
| `adjoint/probe_p1_frame.py` | Does [P1]'s `A` describe *our* `scatter`? | **No.** Lab-frame `sigma` reproduces `A` to 3.5e-18; SPARTA's `v_r`-frame sampler is **rank 6**, off by 53–81%, at Maxwell molecules. §17.1.1, gap U10 |
| `adjoint/probe_jump.py`, `probe_locate.py`, `probe_step4.py`, `probe_chk.py` | Why did FD read 346 against a true 127 with the flip guard silent? | The checksum weighted index 0 by **exactly zero**, so all of cell 0 — the cell at the wall — vanished from it. Also found two missing decisions. §13.4.1 |
| `adjoint/probe_alpha.py` | Is `alpha`'s FD gap cancellation or truncation? | **Truncation**: clean `O(h^2)`, at the floor by `h = 1.1e-5`. The step cap was too loose, not the gradient wrong. §23.2 |
| `adjoint/probe_cost.py` | Why did the reverse pass measure 17x the forward? | **A `vmap` artifact.** Un-batched it is 3.1–5.0x and flat in `N`. Scan-residual and remat-policy hypotheses both **refuted**. §23.4 |
| `adjoint/probe_baseline.py` | Does §11.4's baseline give 84x here? | **No — ~1.2x** on a windowed spatial objective, where the reward's within-run spread dominates. The 84x was the count score in isolation. §11.4 |
| `adjoint/probe_variance.py` | Does the estimator follow [P2]'s `1/sqrt(N M_s)` law? | **Yes.** Fitted `sd ~ N^p` gives `p` = -0.508 / -0.464 / -0.495 / -0.499 for `temp`/`ux`/`alpha`/`diam` over a 64x range in `N`. Also: at [P2]'s `Delta m = 0.1` the AD/FD gap is **truncation**, not noise; and the spatial window costs **signal** (up to 6.3x SNR), not variance. §23.3 |
| `adjoint/probe_split_variance.py` | Is `alpha`'s error bar pathwise or score noise? | **Pathwise, 3-5x over score**, correlation ~0. This is why §11.4's baseline gives only ~1.2x: it can only attack the score half, which is ~19% of the sd. More particles is the only lever. §23.3 |
| `p2_repl.log`, `p2_alpha_sweep.log`, `p2_gases.log` | Can we reproduce [P2] section 6, and is `dJ/d(alpha)` real? | **Yes to both.** Section 6.1 replicates (plateau height grows with beta as `O(Delta m^2)` predicts; beta=0 exact to 4-5 decimals). `dJ/d(alpha)` changes sign at `alpha = sqrt(2)` — argon sits 1.0 % away, which is why its ensemble gradient reads zero. Use helium. See `examples/p2_replication/README.md` |
| **All of the above are consolidated** by `python3 -m examples.shock_adjoint.variance_report` into `examples/shock_adjoint/VARIANCE.md` + 5 figures. It parses these logs, so they stay the source of truth. | | |
| `adjoint/probe_gas_alpha.py` | Does helium's 12.6x better alpha conditioning survive in the 2-D SHOCK, not just 0-D? | **Partly.** `|AD|/SE` goes 0.8 (Ar) -> 2.7 (He): a 3.4x realised gain against 12.6x analytic, vs 15.7x realised in 0-D. Marginally usable, not comfortably. Bigger finding: helium's FD sits at 0.57 sigma, so `SE_FD/SE_AD = 8.4x` -- the first place [P2]'s claimed ~10x AD advantage actually appears. `examples/shock_adjoint/README.md` |
| `adjoint/probe_M_scaling.py` | Is `sd ~ sqrt(M)` even the right comparison here? | **No** — [P2]'s 0-D case relaxes to equilibrium so the signal saturates; in a developing shock the gradient itself grows with `M`, so the *relative* spread is what to read. §23.3 |
| `adjoint/rereport.py` | Regenerate the mode-A report and figures from saved rows | no solver runs; the reason `__main__.py` pickles *before* it plots |
| `probe_scan_fix.py`, `probe_scan_cost.py` | Does the scratch-row fix work, and what does it cost? | 176/300 → 0/300; 1.02–1.11× wall time, 24 bytes. §8.4.1 |
| `probe_scan_dice*.py` | Is naive (non-causal) DiCE variance horizon-scaling? | **No** — saturates at ~3×, flat over `M = 8…4096`. Adopted causal form anyway (one scalar), but without the "architectural" framing. §11.5 |
| `probe_num_c1_safe.py`, `probe_num_c1_harm.py` | Is `eps=1e-30` harmful, and where? | `safe_norm` is bounded by 1.0 *regardless of eps*; the real offender is `safe_pow`. `EPS_D` as an absolute cut gives order-unity divergence between unit systems. §2.3, §7.4 |
| `probe_num_c2a_zero.py`, `probe_num_c2d_variance2.py` | Do majorant violations matter, and does variance grow with Σ? | Violations **bias** the gradient (75% low at 15% rate). Variance **falls** with Σ — opposite of the hypothesis. §7.2.1, §7.2.2 |
| `probe_num_c2e_vremax.py` | Does never-decaying `vremax` degrade the gradient? | No — it improves slightly. Real cost is 2.29× compute and a silent `A_max` overrun. §8.2, §16.2 |
| `probe_num_c3_*.py` | Can FD ever agree with AD once cell binning exists? | Yes, in a 5.5-decade window — but `h_crit ~ 5.8/(N·C·M)`, so the window **closes** at production scale. §13.4.1 |
