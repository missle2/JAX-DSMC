"""Cut an optimisation run back to its first ``n`` iterates so a resubmission resumes there:
``history.json`` keeps rows ``0 .. n-1``; ``coef_k`` for ``k > n``, ``grad_k`` / ``frames_k``
for ``k >= n`` and the rule's saved state (Adam moments etc., the state *after* the last row,
possibly nan) are removed -- the resumed rule restarts its state fresh at ``coef_n``.

    python3 examples/mesh/cluster/truncate_run.py <run dir> <n> [--yes]

Without ``--yes`` it only lists what it would remove.
"""

import argparse
import json
import re
from pathlib import Path

STATES = ("optax_state.npz", "opt_state.npz", "adam_state.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("n", type=int)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    hist_path = args.run / "history.json"
    hist = json.loads(hist_path.read_text())
    if not 0 < args.n <= len(hist["rows"]):
        raise SystemExit(f"n must be in 1 .. {len(hist['rows'])}")
    if not (args.run / f"coef_{args.n}.npy").exists():
        raise SystemExit(f"coef_{args.n}.npy is missing: cannot resume there")
    doomed = [args.run / s for s in STATES if (args.run / s).exists()]
    for p in sorted(args.run.iterdir()):
        m = re.fullmatch(r"(coef|grad|frames)_(\d+)\.(npy|npz)", p.name)
        if m and int(m[2]) >= args.n + (m[1] == "coef"):
            doomed.append(p)
    print(f"{args.run}: keep rows 0..{args.n - 1} of {len(hist['rows'])}, remove {len(doomed)} "
          f"files ({', '.join(p.name for p in doomed[:6])}{' ...' if len(doomed) > 6 else ''})")  # fmt: skip
    if not args.yes:
        print("dry run: add --yes to apply")
        return
    hist["rows"] = hist["rows"][: args.n]
    hist["converged"] = False
    hist_path.write_text(json.dumps(hist))
    for p in doomed:
        p.unlink()


if __name__ == "__main__":
    main()
