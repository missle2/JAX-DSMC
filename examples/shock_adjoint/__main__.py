"""CLI for the adjoint verification.

python3 -m examples.shock_adjoint                  # mode A, all D, figures (~5 min)
python3 -m examples.shock_adjoint --dims 1         # one dimension, quick
python3 -m examples.shock_adjoint --mode ensemble --keys 1024

Mode A (the default) is the verification: pathwise gradients against a fixed-key
finite difference, with a decision-flip guard.  Mode B is the ensemble study of the
score terms -- see the README for why that is not a verification claim.
"""

from __future__ import annotations

import argparse
import pickle
import time

from examples.shock_adjoint import HEADLINE, SCORE_ONLY
from examples.shock_adjoint import verify as V


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("pathwise", "ensemble", "both"), default="both")
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--params", nargs="+", default=list(HEADLINE))
    ap.add_argument("--keys", type=int, default=1024, help="AD ensemble size for mode B")
    ap.add_argument("--fd-factor", type=int, default=4, help="FD keys = fd_factor * keys")
    ap.add_argument("--chunk", type=int, default=None, help="keys per vmapped batch")
    ap.add_argument("--npc", type=int, default=50)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save", default=None, help="pickle the raw sweep rows to this path")
    a = ap.parse_args()

    kw = {"npc": a.npc, "n_steps": a.steps}
    out = {}

    if a.mode in ("pathwise", "both"):
        t0 = time.time()
        res, cases = V.run_pathwise(dims=a.dims, names=tuple(a.params), seed=a.seed, **kw)
        V.print_pathwise(res, cases)
        print(f"  mode A wall {time.time() - t0:.0f}s")
        out["pathwise"] = res
        _save(a.save, out)  # persist BEFORE plotting -- a figure bug must not cost the run
        if not a.no_plot:
            print("  " + V.figure_pathwise(res, cases))
            print("  " + V.figure_diagnostics(res, cases))

    if a.mode in ("ensemble", "both"):
        # A pathwise-only parameter is still worth sweeping here: the score term is a
        # *correction* to it, and mode B is the only place that correction is checked.
        t0 = time.time()
        res, cases = V.run_ensemble(
            dims=a.dims,
            names=tuple(a.params),
            n_keys=a.keys,
            fd_factor=a.fd_factor,
            chunk=a.chunk,
            **kw,
        )
        V.print_ensemble(res, cases, a.keys)
        print(f"  mode B wall {time.time() - t0:.0f}s")
        out["ensemble"] = res
        _save(a.save, out)
        if not a.no_plot:
            print("  " + V.figure_ensemble(res, cases))

    score_swept = [p for p in a.params if p in SCORE_ONLY]
    if a.mode == "pathwise" and score_swept:
        print(
            f"\n  ** {', '.join(score_swept)} reach J only through a frozen decision, so mode A\n"
            "     reports 0.0 for them by construction.  Run --mode ensemble to test them. **"
        )

    _save(a.save, out)


def _save(path, out):
    """Write the raw sweep rows.  Called after each mode, not only at the end.

    The first run of this study lost 378 s of GPU time to an ``IndexError`` in the
    plotting code, because the pickle was written last.  Compute is expensive and
    matplotlib is not; persist first.
    """
    if not path:
        return
    with open(path, "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
