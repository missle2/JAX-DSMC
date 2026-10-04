"""LaTeX tables of the bias-study gates for docs/math/shape_sensitivities_slides.tex:
per s, every grad_x-lambda variant's paired (adjoint - FD) / FD in percent, without and
with the window-edge timing term.

    python3 scratch/bias/latex_tables.py IN.jsonl OUT.tex [variant ...]
"""

import json
import sys

import numpy as np


def pct(diff, fd):
    d = np.asarray(diff) / abs(np.mean(fd)) * 100
    return f"${d.mean():+.1f}\\pm{d.std(ddof=1) / np.sqrt(len(d)):.1f}$"


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    names = sys.argv[3:] or list(rows[0]["variants"])
    out = []
    ss = sorted({r["s"] for r in rows})
    head = " & ".join(f"$s={s:g}$" for s in ss)
    out.append("\\begin{tabular}{@{}l" + "cc" * len(ss) + "@{}}\n\\toprule")
    out.append("variant & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$s={s:g}$}}" for s in ss)
               + "\\\\")
    out.append(" & " + " & ".join(["no timing & + timing"] * len(ss)) + "\\\\\n\\midrule")
    del head
    for n in names:
        cells = []
        for s in ss:
            sub = [r for r in rows if r["s"] == s]
            fd = np.array([np.mean(r["fd"]) for r in sub])
            g = np.array([r["tally"] + r["score"] + r.get("volume", 0.0) + r.get("order", 0.0)
                          + r["variants"][n]["x"] + r["variants"][n]["v"]
                          + r["variants"][n].get("sil", r["silhouette"]) for r in sub])
            t = np.array([r["timing"] for r in sub])
            cells += [pct(g - fd, fd), pct(g + t - fd, fd)]
        out.append("\\code{" + n.replace("_", "\\_") + "} & " + " & ".join(cells) + "\\\\")
    nseed = {s: sum(r["s"] == s for r in rows) for s in ss}
    out.append("\\bottomrule\n\\end{tabular}")
    out.append(f"% seeds per s: {nseed}")
    open(sys.argv[2], "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
