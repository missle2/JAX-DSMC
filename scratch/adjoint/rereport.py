"""Re-print the mode-A report and figures from saved rows -- no solver runs.

`Case.work` is `N*C*M`, independent of `a_max`, so the cases can be rebuilt without
calibrating.  This is the path for regenerating a figure after changing the plotting
code, which is why `__main__.py` now pickles before it plots.
"""

import pickle
import sys

sys.path.insert(0, ".")
from examples.shock_adjoint import case  # noqa: E402
from examples.shock_adjoint import verify as V  # noqa: E402

out = pickle.load(open("scratch/adjoint/modeA_study.pkl", "rb"))
res = out["pathwise"]
cases = {d: case(d) for d in sorted({d for d, _ in res})}
V.print_pathwise(res, cases)
print("  " + V.figure_pathwise(res, cases))
print("  " + V.figure_diagnostics(res, cases))
