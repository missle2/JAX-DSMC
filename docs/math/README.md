# docs/math

Mathematical notes on the estimators, written to be cited from code comments by
equation number (`docs/math/ensemble_adjoint.tex eq. (3.7)`).  Equation numbers are
section-scoped; do not renumber sections without updating the comments that cite them.

| file | content |
|---|---|
| `ensemble_adjoint.tex` | The particle-carried ensemble adjoint as implemented in `jax_dsmc/ensemble_backward.py` / `ensemble_device.py`: forward notation, objectives, the three collision pull-backs (sequential, linearised, virtual-partner substepped), the kNN quadratic field interpolant, the score terms, births/deaths, the validation table (numbers from `docs/ADJOINT_BRANCH_REVIEW.md`), the shape adjoint, the shape optimiser and its status, and (appendix A) a primer on the DSMC forward solver for CFD readers (NTC, VSS, sources, tracking, cut/split cells).  Cites the equation labels of P3 (`papers_reference/arXiv-2603.20946v2/main.tex`). |

## Building

Only `amsmath`, `amssymb` and `hyperref` are used; any TeX Live from 2020 on suffices.
Run from the repository root (two passes so cross-references and the hyperref
bookmarks resolve):

```sh
cd docs/math
pdflatex -interaction=nonstopmode ensemble_adjoint.tex
pdflatex -interaction=nonstopmode ensemble_adjoint.tex
```

`latexmk -pdf ensemble_adjoint.tex` does the same.  On a machine without TeX,
`sudo apt install texlive-latex-recommended texlive-latex-extra` provides everything
needed (`hyperref` is in `texlive-latex-recommended`).

The build products (`*.aux`, `*.log`, `*.out`, `*.pdf`) are not committed.
