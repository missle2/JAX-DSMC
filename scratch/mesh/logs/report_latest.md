
### circle (4 runs, 60 s/run on gpu)

| gate | measurement | ours | SPARTA | z / criterion | pass |
|---|---|---|---|---|---|
| FG1 | diagnostics (sum over runs) | stencil_overrun=0, walk_exhausted=0, inside_body=0, wall_recrossed=0, attempts_clipped=0, inflow_shortfall=0, inflow_tracking_exhausted=0 | 0 | all zero | True |
| FG2 | final Np | 47594 ± 61 | 47683 ± 58 | z = -1.05, z(tail mean) = -0.19, max\|z\| last 300 = 1.30 | True |
| FG2 | Nscoll/step, last 300 | 136.2 | 140.5 | z(tail mean) = -3.08, max\|z\| = 3.81 | |
| FG3 | total drag fx [N/m] | 3.69e-20 ± 7.4e-23 | 3.679e-20 ± 3.7e-23 | z = 1.36 (+0.31 %) | True |
| FG3 | per-element pressure | rms z 1.31, max 5.71, 93 % within 2 | | hits/step 139.1 vs 139.0 | |
| FG4 | nrho per cell (310 cells) | mean 0.6571 | mean 0.6565 | rms z 1.21, 90 % within 2 | True |
| FG4 | u per cell (310 cells) | mean 192.9 | mean 192.8 | rms z 1.20, 90 % within 2 | True |
| FG4 | temp per cell (310 cells) | mean 334.4 | mean 334.2 | rms z 1.19, 92 % within 2 | True |
| FG7 polygon vs analytic | Np / Nscoll / fx | 47594 / 139.1 / 3.69e-20 | 47429 / 138.6 / 3.709e-20 | rel -0.35 / -0.4 / +0.51 % | |
| walk vs stencil | Np / Nscoll / fx | 47594 / 139.1 / 3.69e-20 | 47444 / 138.6 / 3.705e-20 | rel -0.32 / -0.3 / +0.41 % | |

### sphere (4 runs, 292 s/run on gpu)

| gate | measurement | ours | SPARTA | z / criterion | pass |
|---|---|---|---|---|---|
| FG1 | diagnostics (sum over runs) | stencil_overrun=0, walk_exhausted=0, inside_body=0, wall_recrossed=0, attempts_clipped=0, inflow_shortfall=0, inflow_tracking_exhausted=0 | 0 | all zero | True |
| FG2 | final Np | 80757 ± 57 | 80642 ± 105 | z = 0.97, z(tail mean) = 1.01, max\|z\| last 300 = 1.53 | True |
| FG2 | Nscoll/step, last 300 | 177.0 | 174.6 | z(tail mean) = 1.60, max\|z\| = 3.70 | |
| FG3 | total drag fx [N] | 1.626e-20 ± 4.5e-23 | 1.628e-20 ± 4.3e-23 | z = -0.30 (-0.11 %) | True |
| FG3 | per-element pressure | rms z 1.19, max 4.81, 92 % within 2 | | hits/step 174.9 vs 175.2 | |
| FG4 | nrho per cell (7696 cells) | mean 0.671 | mean 0.6707 | rms z 1.24, 91 % within 2 | True |
| FG4 | u per cell (7696 cells) | mean 211.7 | mean 211.6 | rms z 1.23, 91 % within 2 | True |
| FG4 | temp per cell (7696 cells) | mean 319.3 | mean 319.2 | rms z 1.24, 91 % within 2 | True |
| FG7 polygon vs analytic | Np / Nscoll / fx | 80757 / 174.9 / 1.626e-20 | 80589 / 175.5 / 1.639e-20 | rel -0.21 / +0.3 / +0.75 % | |
| walk vs stencil | Np / Nscoll / fx | 80757 / 174.9 / 1.626e-20 | 80643 / 174.5 / 1.631e-20 | rel -0.14 / -0.3 / +0.29 % | |

### G3a window score (1-D closed box, windowed impulse; 12 seeds)

| W | FD ± SE (sd) | analytic | eps-score mean ± SE, eps = 0.02 | eps = 0.01 | score sd / FD sd |
|---|---|---|---|---|---|
| 1 | -0.0492 ± 0.0131 (0.0453) | -0.0504 | +0.0136 ± 0.0519 (z +0.3) | -0.0133 ± 0.0812 (z -0.2) | 4.0 / 6.2 |
| 10 | -0.0458 ± 0.0033 (0.0113) | -0.0494 | +0.0064 ± 0.0235 (z +0.3) | -0.0264 ± 0.0210 (z -1.3) | 7.2 / 6.4 |
| 100 | -0.0503 ± 0.0015 (0.0052) | -0.0497 | -0.0011 ± 0.0099 (z -0.1) | +0.0195 ± 0.0121 (z +1.6) | 6.5 / 8.0 |

### Adjoint gates G1-G3 and the shape cases (ensemble adjoint vs CRN FD)

| gate | case | seeds | J | ensemble ± SE | FD ± SE (h) | analytic / truncated FD | z | pass |
|---|---|---|---|---|---|---|---|---|
| G1 | Couette windowed v_y^2, speed | 48 | 1.239 | +0.491 ± 0.025 | +0.4925 ± 0.016 (0.0125) |  | -0.05 | True |
| G1 | Couette windowed v_y^2, speed [one-sided ghost fits] | 48 | 1.239 | +0.4906 ± 0.025 | +0.4925 ± 0.016 (0.0125) |  | -0.06 | True |
| G2 | Couette windowed shear, speed | 48 | -0.01624 | -0.01771 ± 0.0014 | -0.01646 ± 0.00075 (0.0125) |  | -0.77 | True |
| G2 | Couette windowed shear, speed [one-sided ghost fits] | 48 | -0.01624 | -0.01774 ± 0.0014 | -0.01646 ± 0.00075 (0.0125) |  | -0.80 | True |
| G2 | Couette windowed shear, T_left | 48 | -0.01624 | -0.002489 ± 0.00064 | -0.003224 ± 0.0015 (0.1) |  | +0.46 | True |
| G2 | Couette windowed shear, T_left [one-sided ghost fits] | 48 | -0.01624 | -0.002508 ± 0.00064 | -0.003224 ± 0.0015 (0.1) |  | +0.45 | True |
| G3 | embedded plane position, windowed v_y^2 | 32 | 1.438 | -0.3717 ± 0.067 (pathwise +0.07348, score +0, timing -0.4452) | -0.0796 ± 0.1 (0.04) | truncated FD -0.1493 ± 0.037 (256 seeds), z = -2.91 | -2.41 | False |
| G3 | embedded plane position, windowed v_y^2 [two-sided fit, no timing term: superseded] | 32 | 1.438 | +0.7601 ± 0.12 (pathwise +0.7601, score +0, timing +0) | -0.0796 ± 0.1 (0.04) | truncated FD -0.1493 ± 0.037 (256 seeds), z = +7.23 | +5.35 | False |
| G3 | embedded plane position, windowed v_y^2 | 32 | 1.438 | -0.3717 ± 0.067 (pathwise +0.07348, score +0, timing -0.4452) | -0.0796 ± 0.1 (0.04) | truncated FD -0.1493 ± 0.037 (256 seeds), z = -2.91 | -2.41 | False |
| G3 | embedded plane position, windowed shear | 32 | -0.01806 | -0.001785 ± 0.005 (pathwise -0.001703, score +0, timing -8.124e-05) | +0.04312 ± 0.01 (0.01) | truncated FD +0.00173 ± 0.0026 (64 seeds), z = -0.63 | -3.97 | True |
| G3 | embedded plane position, windowed shear [two-sided fit, no timing term: superseded] | 32 | -0.01806 | -0.01217 ± 0.0054 (pathwise -0.01217, score +0, timing +0) | +0.04312 ± 0.01 (0.01) | truncated FD +0.00173 ± 0.0026 (64 seeds), z = -2.33 | -4.81 | False |
| G3 | embedded plane position, windowed shear | 32 | -0.01806 | -0.001785 ± 0.005 (pathwise -0.001703, score +0, timing -8.124e-05) | +0.04312 ± 0.01 (0.01) | truncated FD +0.00173 ± 0.0026 (64 seeds), z = -0.63 | -3.97 | True |
| G3 | embedded plane position, windowed shear | 32 | -0.01814 | +0.0141 ± 0.014 (pathwise +0.01162, score +0, timing +0.002482) | +0.01881 ± 0.0081 (0.01) | truncated FD +0.01696 ± 0.0023 (64 seeds), z = -0.21 | -0.30 | True |
| D | analytic circle radius, windowed drag [mixed-pool ∇λ fits] | 48 | 3.697e-20 | +1.44e-20 ± 8.9e-22 (pathwise +1.44e-20, score +0, timing +7.059e-22) | +1.395e-20 ± 2e-22 (0.05) | +1.232e-20 (F/r) | +0.49 | True |
| D | analytic circle radius, windowed drag [mixed-pool ∇λ fits] | 35 | 3.701e-20 | +1.452e-20 ± 1.1e-21 (pathwise +1.452e-20, score +0, timing +9.598e-22) | +1.407e-20 ± 2.2e-22 (0.05) | +1.234e-20 (F/r) | +0.40 | True |
| D | analytic circle centre_x, windowed drag [mixed-pool ∇λ fits] | 32 | 3.7e-20 | -8.496e-22 ± 6.3e-22 (pathwise -8.496e-22, score +0, timing -6.028e-22) | -6.878e-23 ± 1.7e-22 (0.1) | +0 (F/r) | -1.20 | True |
| LS2 | level-set circle radius, windowed drag [mixed-pool ∇λ fits] | 4 | 3.68e-20 | +1.207e-20 ± 8.9e-22 (pathwise +1.207e-20, score +0, timing -3.233e-22) | +1.398e-20 ± 1e-21 (0.05) | +1.227e-20 (F/r) | -1.39 | True |
| LS2 | level-set circle cos 2θ mode, windowed drag [mixed-pool ∇λ fits] | 4 | 3.68e-20 | -1.587e-20 ± 3e-22 (pathwise -1.587e-20, score +0, timing +1.531e-22) | -1.571e-20 ± 5.9e-22 (0.05) | +0 (F/r) | -0.23 | True |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [score form, hit-time cut-off only] [mixed-pool ∇λ fits] | 4 | 1.872e-20 | +1.274e-20 ± 1.6e-21 (pathwise -1.541e-22, score +7.041e-21, timing +6.46e-22, silhouette +5.853e-21) | +1.169e-20 ± 9.6e-22 (0.05) | +0 (F/r) | +0.57 | True |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [score form + emission-side floor] [mixed-pool ∇λ fits] | 4 | 1.872e-20 | +1.228e-20 ± 1.3e-21 (pathwise -1.541e-22, score +6.577e-21, timing +6.46e-22, silhouette +5.853e-21) | +1.169e-20 ± 9.6e-22 (0.05) | +0 (F/r) | +0.36 | True |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [mixed-pool ∇λ fits] | 4 | 1.872e-20 | +1.191e-20 ± 6.8e-22 (pathwise +6.039e-21, score +2.265e-23, timing +6.46e-22, silhouette +5.853e-21) | +1.169e-20 ± 9.6e-22 (0.05) | +0 (F/r) | +0.19 | True |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] | 4 | 1.861e-20 | +1.337e-20 ± 3.6e-22 (pathwise +1.359e-20, score +0, timing +1.85e-22, silhouette -2.201e-22) | +1.355e-20 ± 3.8e-22 (0.05) | +0 (F/r) | -0.35 | True |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [mixed-pool ∇λ fits] | 4 | 1.861e-20 | +9.452e-21 ± 6.7e-22 (pathwise +9.672e-21, score +0, timing +1.85e-22, silhouette -2.201e-22) | +1.355e-20 ± 3.8e-22 (0.05) | +0 (F/r) | -5.32 | False |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 4 | 1.859e-20 | +1.104e-20 ± 2.5e-22 (pathwise -2.022e-22, score +5.783e-21, timing +2.783e-22, silhouette +5.463e-21) | +1.305e-20 ± 1.4e-21 (0.05) | +0 (F/r) | -1.41 | True |
| GP | diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 2 | 1.872e-20 | +1.039e-20 ± 7.5e-22 (pathwise +1.026e-20, score +0, timing -3.734e-22, silhouette +1.295e-22) | +1.354e-20 ± 9.6e-22 (0.05) | +0 (F/r) | -2.59 | False |
| GD | diffuse level-set circle radius, windowed drag (hybrid normal + silhouette terms) | 4 | 4.142e-20 | +1.312e-20 ± 5.2e-22 (pathwise +5.139e-21, score -2.04e-22, timing +3.586e-22, silhouette +8.183e-21) | +1.128e-20 ± 1.6e-21 (0.05) | +1.381e-20 (F/r) | +1.10 | True |

Fit bias on the plane (replica-exact value gradients, 3 seeds, x/v-part of the windowed shear estimator): true +0.0030 ± 0.0007; one-sided fit +0.0048 ± 0.0025 (bias +0.0017 ± 0.0024); two-sided fit -0.0265 ± 0.0114 (bias -0.0295 ± 0.0114)

Fit bias on the plane (replica-exact value gradients, 3 seeds, x/v-part of the windowed v_y^2 estimator): true +0.3554 ± 0.1207; one-sided fit +0.0772 ± 0.1280 (bias -0.2837 ± 0.1803); two-sided fit +1.1194 ± 0.2782 (bias +0.7700 ± 0.3109)

### LS1/LS2 level-set circle vs analytic circle, seed by seed (CRN)

| quantity | max |Δ| / max |analytic| over common seeds |
|---|---|
| objective | 1.34e-01 |
| fd | 1.26e-01 |
| hits | 2.54e-02 |
| pathwise_tally | 7.21e-01 |
| timing | 4.38e-01 |
| gradient, variant `one` (x-part, v-part) | 7.79e-01 (1.79e+00, 1.00e+00) |
| gradient, variant `one_pool` (x-part, v-part) | 7.95e-01 (8.87e+00, 1.00e+00) |
| gradient, variant `along` (x-part, v-part) | 4.91e-01 (2.11e+00, 1.00e+00) |
| gradient, variant `along_pool` (x-part, v-part) | 3.35e-01 (3.26e+00, 1.00e+00) |

level set (`along_pool`): 4 seeds, ensemble +1.312e-20 ± 5.2e-22, FD +1.128e-20 ± 1.6e-21, z(FD) = +1.10; same seeds analytic mean +1.196e-20 vs level set +1.312e-20; per-seed gradient difference ≤ 2.3170 seed-sd (seeds [0, 1, 2, 3])

### Grazing regularisation of the tally term (circle radius, CRN seeds)

| eps | tally mean | seed sd (%) | paired bias vs eps = 0 (%) | largest per-hit weight | hits below cut-off / window |
|---|---|---|---|---|---|
| 0.0 | 1.257e-20 | 8.4 | +0.00 ± 0.00 | 3.45e+04 | 0.0 |
| 0.01 | 1.269e-20 | 5.9 | +0.97 ± 1.14 | 2.21e+04 | 0.5 |
| 0.03 | 1.264e-20 | 4.2 | +0.51 ± 1.75 | 1.2e+04 | 4.8 |
| 0.1 | 1.243e-20 | 2.9 | -1.16 ± 2.26 | 4.39e+03 | 32.2 |
| 0.3 | 1.165e-20 | 2.1 | -7.35 ± 2.41 | 1.68e+03 | 308.1 |

### GP: the diffuse normal term, three estimators seed by seed (CRN)

`ensemble_mesh_plate_levelset0.0625_acc1_sil0.03_gz0.05_width_f4_H700_M300_W200_virt1x2st_dev.json` (hybrid, shipped): 4 seeds, ensemble 1.191e-20 ± 6.8e-22 vs FD 1.169e-20 ± 9.6e-22, z = +0.19; per seed score 6.42e-23, -4.93e-24, 3.43e-23, -2.94e-24; tally 5.06e-21, 6.32e-21, 6.5e-21, 6.9e-21

| estimator | seeds | identical: J, FD, silhouette, timing (max rel Δ) | normal term per seed | seed sd (form → hybrid) | paired Δ (hybrid - form) | total seed sd (form → hybrid) |
|---|---|---|---|---|---|---|
| score form, hit-time cut-off only (`_prefloor`) | [0, 1, 2, 3] | 2.5e-15 | 1.07e-20, 6.29e-21, 3.3e-21, 7.83e-21 | 3.1e-21 → 7.7e-22 | -8.26e-22 ± 1.8e-21 | 3.2e-21 → 1.4e-21 |
| score form + emission-side floor (`_scoreform`) | [0, 1, 2, 3] | 2.5e-15 | 8.64e-21, 6.54e-21, 3.37e-21, 7.75e-21 | 2.3e-21 → 7.7e-22 | -3.62e-22 ± 1.4e-21 | 2.6e-21 → 1.4e-21 |

### GP: CRN finite difference vs step (seeds [0, 1, 2, 3]; hybrid ensemble 1.191e-20 ± 6.8e-22)

| h | FD per seed | mean ± SE | z(ensemble - FD) |
|---|---|---|---|
| 0.2 | 1.23e-20, 1.35e-20, 1.23e-20, 1.38e-20 | 1.298e-20 ± 3.9e-22 | -1.37 |
| 0.1 | 1.31e-20, 1.5e-20, 1.22e-20, 1.31e-20 | 1.336e-20 ± 6e-22 | -1.60 |
| 0.05 | 9.04e-21, 1.36e-20, 1.2e-20, 1.21e-20 | 1.169e-20 ± 9.6e-22 | +0.19 |
| 0.025 | 1.28e-20, 1.24e-20, 8.85e-21, 1.3e-20 | 1.177e-20 ± 9.8e-22 | +0.12 |
| 0.0125 | 3.45e-21, 1.02e-20, 1.13e-20, 1.75e-20 | 1.062e-20 ± 2.9e-21 | +0.44 |

### Fit error by particle class (plane, replica-exact truth, 2 seeds)

| class | n | scale d lambda/dx | two-sided bias | one-sided bias | rms two / one | value scale | value bias two / one | value rms two / one |
|---|---|---|---|---|---|---|---|---|
| hit | 512 | 6.16e-06 | +4.69e-06 ± 4.0e-06 | +3.41e-06 ± 2.3e-06 | 9.05e-05 / 5.14e-05 | 1.12e-06 | +1.67e-08 / +1.31e-07 | 2.57e-06 / 1.82e-06 |
| near-out | 512 | 3.59e-06 | +1.09e-06 ± 4.7e-06 | +6.85e-07 ± 1.9e-06 | 1.05e-04 / 4.31e-05 | 6.09e-07 | -1.36e-07 / -1.23e-07 | 4.29e-06 / 3.33e-06 |
| near-in | 512 | 7.36e-04 | -4.34e-06 ± 3.5e-05 | -1.13e-06 ± 3.5e-05 | 7.83e-04 / 7.96e-04 | 2.04e-05 | +2.80e-07 / +1.11e-07 | 9.98e-06 / 1.01e-05 |
| wall0-in | 512 | 4.34e-05 | -1.19e-06 ± 5.2e-06 | +9.99e-07 ± 5.6e-06 | 1.18e-04 / 1.27e-04 | 4.84e-06 | +3.81e-07 / +5.55e-07 | 6.58e-06 / 6.93e-06 |
| wall0-out | 512 | 9.50e-06 | +1.71e-06 ± 4.1e-06 | -5.90e-06 ± 3.7e-06 | 9.19e-05 / 8.34e-05 | 9.50e-06 | +1.66e-07 / +1.93e-07 | 5.31e-06 / 5.11e-06 |
| interior | 512 | 2.34e-04 | -2.57e-06 ± 1.1e-05 | +8.39e-07 ± 1.1e-05 | 2.56e-04 / 2.55e-04 | 1.17e-05 | -1.20e-07 / -2.36e-07 | 6.26e-06 / 6.22e-06 |

### Estimators of grad_x lambda vs the replica-exact gradient (windowed shear, 4 seeds)

replica-exact x/v-part: +0.0032 ± 0.0006
| variant | x/v-part ± SE | bias vs exact | per-seed SE (variance) | rms per hit |
|---|---|---|---|---|
| two-sided fit | -0.0208 ± 0.0101 | -0.0242 ± 0.0102 | 0.0209 | 3.92e-04 |
| one-sided fit | +0.0053 ± 0.0022 | +0.0019 ± 0.0022 | 0.0057 | 1.08e-04 |
| one-sided, pooled 3^D (k = 96) | -0.0001 ± 0.0008 | -0.0031 ± 0.0010 | 0.0017 | 4.03e-05 |
| one-sided + along-flight identity | -0.0003 ± 0.0005 | -0.0030 ± 0.0010 | 0.0030 | 6.26e-05 |
| pooled + along-flight identity | -0.0001 ± 0.0002 | -0.0031 ± 0.0007 | 0.0021 | 5.25e-05 |
along-flight identity per particle: (-T) - v.grad V = -1.85e-07 ± 1.1e-06 (rms diff 4.63e-05, scale of v.grad V 1.39e-05)

### Estimators of grad_x lambda vs the replica-exact gradient (windowed v_y^2, 4 seeds)

replica-exact x/v-part: +0.3742 ± 0.1110
| variant | x/v-part ± SE | bias vs exact | per-seed SE (variance) | rms per hit |
|---|---|---|---|---|
| two-sided fit | +1.0625 ± 0.2432 | +0.6832 ± 0.2720 | 0.4920 | 1.03e-02 |
| one-sided fit | +0.0818 ± 0.1204 | -0.2948 ± 0.1667 | 0.2598 | 6.51e-03 |
| one-sided, pooled 3^D (k = 96) | +0.1232 ± 0.0277 | -0.1996 ± 0.1113 | 0.0581 | 4.30e-03 |
| one-sided + along-flight identity | +0.2268 ± 0.0697 | -0.0816 ± 0.1103 | 0.1480 | 4.33e-03 |
| pooled + along-flight identity | +0.2686 ± 0.0588 | -0.0720 ± 0.1064 | 0.1314 | 4.18e-03 |
along-flight identity per particle: (-T) - v.grad V = -1.29e-05 ± 2.8e-05 (rms diff 1.83e-03, scale of v.grad V 1.63e-03)

### Estimator variants on the production runs (same seeds; reference = truncated FD for the plane, CRN FD for the circle)

| case | seeds | variant | gradient ± SE | per-seed sd | x-part ± SE | z vs reference |
|---|---|---|---|---|---|---|
| embedded plane position, windowed v_y^2 | 32 | one-sided fit | -0.3717 ± 0.067 | 0.38 | +0.0735 ± 0.06 | -2.91 |
| embedded plane position, windowed v_y^2 | 32 | one-sided, pooled 3^D (k = 96) | -0.3378 ± 0.036 | 0.21 | +0.107 ± 0.019 | -3.64 |
| embedded plane position, windowed v_y^2 | 32 | one-sided + along-flight identity | -0.1879 ± 0.04 | 0.23 | +0.257 ± 0.027 | -0.71 |
| embedded plane position, windowed v_y^2 | 32 | pooled + along-flight identity | -0.2044 ± 0.041 | 0.23 | +0.241 ± 0.027 | -1.01 |
| embedded plane position, windowed shear | 32 | one-sided fit | -0.001785 ± 0.005 | 0.028 | -0.0017 ± 0.0013 | -0.63 |
| embedded plane position, windowed shear | 32 | one-sided, pooled 3^D (k = 96) | +0.0004305 ± 0.0052 | 0.03 | +0.000512 ± 0.00049 | -0.22 |
| embedded plane position, windowed shear | 32 | one-sided + along-flight identity | +0.001387 ± 0.0052 | 0.03 | +0.00147 ± 0.00075 | -0.06 |
| embedded plane position, windowed shear | 32 | pooled + along-flight identity | +0.001431 ± 0.0053 | 0.03 | +0.00151 ± 0.0007 | -0.05 |
| analytic circle radius, windowed drag [mixed-pool ∇λ fits] | 35 | one-sided fit | +1.452e-20 ± 1.1e-21 | 6.6e-21 | +1.48e-21 ± 1.1e-21 | +0.40 |
| analytic circle radius, windowed drag [mixed-pool ∇λ fits] | 35 | one-sided, pooled 3^D (k = 96) | +1.377e-20 ± 1.6e-22 | 9.2e-22 | +1.35e-21 ± 7e-23 | -1.11 |
| analytic circle radius, windowed drag [mixed-pool ∇λ fits] | 35 | one-sided + along-flight identity | +1.434e-20 ± 5.3e-22 | 3.2e-21 | +1.3e-21 ± 4.8e-22 | +0.47 |
| analytic circle radius, windowed drag [mixed-pool ∇λ fits] | 35 | pooled + along-flight identity | +1.353e-20 ± 2.4e-22 | 1.4e-21 | +1.1e-21 ± 1.6e-22 | -1.65 |
| level-set circle radius, windowed drag [mixed-pool ∇λ fits] | 4 | one-sided fit | +1.449e-20 ± 3.5e-21 | 7.1e-21 | +2.3e-21 ± 3.3e-21 | +0.14 |
| level-set circle radius, windowed drag [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | +1.29e-20 ± 7.1e-22 | 1.4e-21 | +8.21e-22 ± 1.3e-22 | -0.86 |
| level-set circle radius, windowed drag [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | +1.259e-20 ± 1.7e-21 | 3.4e-21 | +3.92e-22 ± 1.2e-21 | -0.71 |
| level-set circle radius, windowed drag [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | +1.207e-20 ± 8.9e-22 | 1.8e-21 | -5.71e-24 ± 3.4e-22 | -1.39 |
| level-set circle cos 2θ mode, windowed drag [mixed-pool ∇λ fits] | 4 | one-sided fit | -1.529e-20 ± 7.7e-22 | 1.5e-21 | +2.47e-22 ± 1.1e-21 | +0.44 |
| level-set circle cos 2θ mode, windowed drag [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | -1.674e-20 ± 4.1e-22 | 8.3e-22 | -1.15e-21 ± 1.5e-22 | -1.43 |
| level-set circle cos 2θ mode, windowed drag [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | -1.467e-20 ± 6e-22 | 1.2e-21 | +8.66e-22 ± 6.5e-22 | +1.24 |
| level-set circle cos 2θ mode, windowed drag [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | -1.587e-20 ± 3e-22 | 6.1e-22 | -2.76e-22 ± 1.4e-22 | -0.23 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form, hit-time cut-off only] [mixed-pool ∇λ fits] | 4 | one-sided fit | +1.268e-20 ± 1.9e-21 | 3.7e-21 | -2.14e-22 ± 4.6e-22 | +0.47 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form, hit-time cut-off only] [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | +1.147e-20 ± 1.6e-21 | 3.1e-21 | -1.42e-21 ± 1.1e-22 | -0.11 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form, hit-time cut-off only] [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | +1.291e-20 ± 1.8e-21 | 3.5e-21 | +1.6e-23 ± 2.1e-22 | +0.61 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form, hit-time cut-off only] [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | +1.274e-20 ± 1.6e-21 | 3.2e-21 | -1.54e-22 ± 5.3e-24 | +0.57 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form + emission-side floor] [mixed-pool ∇λ fits] | 4 | one-sided fit | +1.222e-20 ± 1.5e-21 | 3e-21 | -2.14e-22 ± 4.6e-22 | +0.30 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form + emission-side floor] [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | +1.101e-20 ± 1.3e-21 | 2.5e-21 | -1.42e-21 ± 1.1e-22 | -0.43 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form + emission-side floor] [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | +1.245e-20 ± 1.5e-21 | 3e-21 | +1.6e-23 ± 2.1e-22 | +0.43 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [score form + emission-side floor] [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | +1.228e-20 ± 1.3e-21 | 2.6e-21 | -1.54e-22 ± 5.3e-24 | +0.36 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [mixed-pool ∇λ fits] | 4 | one-sided fit | +1.185e-20 ± 4.8e-22 | 9.6e-22 | -2.14e-22 ± 4.6e-22 | +0.16 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | +1.065e-20 ± 5.9e-22 | 1.2e-21 | -1.42e-21 ± 1.1e-22 | -0.92 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | +1.208e-20 ± 7.5e-22 | 1.5e-21 | +1.6e-23 ± 2.1e-22 | +0.33 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | +1.191e-20 ± 6.8e-22 | 1.4e-21 | -1.54e-22 ± 5.3e-24 | +0.19 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] | 4 | one-sided fit | +1.335e-20 ± 4.3e-22 | 8.7e-22 | -2.42e-24 ± 5.8e-25 | -0.36 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] | 4 | one-sided, pooled 3^D (k = 96) | +1.336e-20 ± 3.9e-22 | 7.8e-22 | -2.83e-23 ± 1.8e-22 | -0.35 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] | 4 | one-sided + along-flight identity | +1.335e-20 ± 4.3e-22 | 8.6e-22 | -3.24e-25 ± 2.3e-24 | -0.35 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] | 4 | pooled + along-flight identity | +1.337e-20 ± 3.6e-22 | 7.2e-22 | -1.93e-23 ± 7.7e-23 | -0.35 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [mixed-pool ∇λ fits] | 4 | one-sided fit | +1.289e-20 ± 1.4e-21 | 2.8e-21 | -5.79e-22 ± 3.1e-22 | -0.46 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | +8.791e-21 ± 6.5e-22 | 1.3e-21 | -1.29e-21 ± 5.6e-23 | -6.36 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | +1.301e-20 ± 1.3e-21 | 2.5e-21 | -4.65e-22 ± 2e-22 | -0.41 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | +9.452e-21 ± 6.7e-22 | 1.3e-21 | -6.27e-22 ± 4.5e-23 | -5.32 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 4 | one-sided fit | +1.109e-20 ± 5.3e-22 | 1.1e-21 | -1.58e-22 ± 4.1e-22 | -1.31 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 4 | one-sided, pooled 3^D (k = 96) | +9.831e-21 ± 1.7e-22 | 3.3e-22 | -1.41e-21 ± 1.4e-22 | -2.29 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 4 | one-sided + along-flight identity | +1.226e-20 ± 5.9e-22 | 1.2e-21 | +1.01e-21 ± 6.5e-22 | -0.52 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 4 | pooled + along-flight identity | +1.104e-20 ± 2.5e-22 | 5.1e-22 | -2.02e-22 ± 4.5e-23 | -1.41 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 2 | one-sided fit | +1.276e-20 ± 1.6e-21 | 2.3e-21 | -1.8e-22 ± 9.5e-22 | -0.41 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 2 | one-sided, pooled 3^D (k = 96) | +9.694e-21 ± 7.1e-22 | 1e-21 | -1.33e-21 ± 6.7e-23 | -3.23 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 2 | one-sided + along-flight identity | +1.299e-20 ± 1.5e-21 | 2.1e-21 | +5.36e-23 ± 7.9e-22 | -0.31 |
| diffuse capsule plate half-width, windowed drag (silhouette term) [specular wall: no silhouette] [coarse h_φ = 0.125] [mixed-pool ∇λ fits] | 2 | pooled + along-flight identity | +1.039e-20 ± 7.5e-22 | 1.1e-21 | -6.31e-22 ± 2.9e-23 | -2.59 |
| diffuse level-set circle radius, windowed drag (hybrid normal + silhouette terms) | 4 | one-sided fit | +7.21e-21 ± 2.4e-21 | 4.8e-21 | -4.5e-21 ± 2.5e-21 | -1.43 |
| diffuse level-set circle radius, windowed drag (hybrid normal + silhouette terms) | 4 | one-sided, pooled 3^D (k = 96) | +1.194e-20 ± 3.7e-21 | 7.5e-21 | +2.35e-22 ± 3.7e-21 | +0.16 |
| diffuse level-set circle radius, windowed drag (hybrid normal + silhouette terms) | 4 | one-sided + along-flight identity | +1.175e-20 ± 1.2e-21 | 2.4e-21 | +4.35e-23 ± 1.2e-21 | +0.23 |
| diffuse level-set circle radius, windowed drag (hybrid normal + silhouette terms) | 4 | pooled + along-flight identity | +1.312e-20 ± 5.2e-22 | 1e-21 | +1.41e-21 ± 3.8e-22 | +1.10 |
wrote /home/aakash/jax_dsmc/JAX-DSMC-adjoint/examples/mesh/results/forward_gates.json
