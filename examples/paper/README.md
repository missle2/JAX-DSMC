# examples/paper

Runs and reference computations behind the paper *Shape optimization with adjoint DSMC*
(the draft itself is kept outside this repository).

| path | what |
|---|---|
| `cluster/study.py` | the paper's cluster groups (`paper_fd`, `paper_fwd`, `paper_airfoil`, `paper_newton`, `paper_heat`, `paper_sph`, `paper_ld2d`); the module docstring maps each to its figure |
| `reference/newton_fm.py` | exact free-molecular optimum of Newton's problem (convex bodies of revolution in a cylinder of given height and base): panel integrals of a drifting Maxwellian, specular or diffuse, optionally with a cap on the panel heat flux |
| `figures/` | data exports for the paper's TikZ/pgfplots figures |

Submitting (from the repository root on a login node, after pulling):

```sh
STUDY_MODULE=examples.paper.cluster.study GPU_KIND=rtx      bash examples/mesh/cluster/submit_curc.sh paper_fd paper_fwd paper_airfoil paper_ld2d
STUDY_MODULE=examples.paper.cluster.study GPU_KIND=h200_71  bash examples/mesh/cluster/submit_curc.sh paper_newton paper_heat
STUDY_MODULE=examples.paper.cluster.study GPU_KIND=h200_full ONLY='^pilot$' bash examples/mesh/cluster/submit_curc.sh paper_sph
```

Optimisation results land in `examples/mesh/results/optimize/paper_*` (one directory per tag,
resumable: resubmit a group to continue it); the finite-difference gates in
`examples/paper/results/fd/*.json`.  Neither is tracked by git.

The local reference (seconds, CPU):

```sh
python3 examples/paper/reference/newton_fm.py --height 2 --stations 200            # specular, M 10
python3 examples/paper/reference/newton_fm.py --height 2 --wall diffuse --beta 0.8 0.6
```
