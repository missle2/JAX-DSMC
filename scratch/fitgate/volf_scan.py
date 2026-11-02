"""Paired comparison of collision fields in the volume gate (FD shared per seed)."""

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "examples/mesh/results/fitgate"
FIELDS = ("base", "noguard", "guard8", "cubic", "quad", "pool1", "knn")


def rows(gate, field):
    pat = f"volume_{gate}_s*.jsonl" if field == "base" else f"volume_{gate}_{field}_s*.jsonl"
    out = [json.loads(x) for p in sorted(RESULTS.glob(pat)) for x in p.read_text().splitlines()]
    return {r["seed"]: r for r in out if not r.get("smoke")}


def mse(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)


for gate in ("c10_spec", "c10_diff"):
    base = rows(gate, "base")
    for cells in ("all", "cut"):
        fd = np.array([np.mean(r[f"fd_{cells}"]) for r in base.values()])
        f, fs = mse(fd)
        print(f"\n{gate} {cells}: FD (mean of eps, n={len(fd)}) {f:.4f} +- {fs:.4f}")
        for field in FIELDS:
            rr = rows(gate, field)
            seeds = sorted(set(rr) & set(base))
            if not seeds:
                continue
            p, ps = mse([rr[s][f"pred_{cells}"] for s in seeds])
            d, ds = mse([rr[s][f"pred_{cells}"] - base[s][f"pred_{cells}"] for s in seeds])
            fdp, _ = mse([np.mean(base[s][f"fd_{cells}"]) for s in seeds])
            print(f"  {field:8s} n={len(seeds):2d} pred {p:.4f}+-{ps:.4f}  "
                  f"pred-base {d:+.4f}+-{ds:.4f}  pred/FD(all seeds) {p / f:.2f}  "
                  f"pred/FD(same seeds) {p / fdp:.2f}")  # fmt: skip

# the virtual pull-back's substeps x partners (volp jobs), paired with the base rows; the
# prediction's change per collision-rate bin locates where the substeps move it
for gate, virtuals in (("c10_diff", ("4x8", "8x16", "16x8", "32x8")), ("c10_spec", ("4x8",))):
    base = rows(gate, "base")
    for v in virtuals:
        rr = rows(gate, f"v{v}")
        seeds = sorted(set(rr) & set(base))
        if not seeds:
            continue
        fdp, _ = mse([np.mean(base[s]["fd_all"]) for s in seeds])
        p, ps = mse([rr[s]["pred_all"] for s in seeds])
        d, ds = mse([rr[s]["pred_all"] - base[s]["pred_all"] for s in seeds])
        print(f"\n{gate} v{v} n={len(seeds)} pred {p:.4f}+-{ps:.4f} pred-base {d:+.4f}+-{ds:.4f} "
              f"pred/FD(same seeds) {p / fdp:.2f}")  # fmt: skip
        r0 = rr[seeds[0]]
        print("  rate quantiles (50/90/99/max):", np.round(r0["rate_quantiles"], 3))
        by = np.array([rr[s]["pred_by_rate"] for s in seeds]).mean(0)
        for lo, hi, c, b in zip(r0["rate_edges"][:-1], r0["rate_edges"][1:], r0["cells_by_rate"],
                                by, strict=True):  # fmt: skip
            print(f"  rate [{lo}, {hi}) cells {c:6d} pred {b:+.4f}")

# the captured horizon (volh jobs): the same forward and FD as c10_diff, 600 / 1000 steps
base = rows("c10_diff", "base")
for horizon in ("h600", "h1000"):
    rr = rows(f"c10_diff_{horizon}", "base")
    seeds = sorted(set(rr) & set(base))
    if not seeds:
        continue
    same = all(abs(rr[s]["objective"] - base[s]["objective"]) < 1e-9 for s in seeds)
    for cells in ("all", "cut"):
        fdp, _ = mse([np.mean(base[s][f"fd_{cells}"]) for s in seeds])
        p, ps = mse([rr[s][f"pred_{cells}"] for s in seeds])
        d, ds = mse([rr[s][f"pred_{cells}"] - base[s][f"pred_{cells}"] for s in seeds])
        print(f"c10_diff {horizon} {cells} n={len(seeds)} pred {p:.4f}+-{ps:.4f} "
              f"pred-h300 {d:+.4f}+-{ds:.4f} pred/FD {p / fdp:.2f} same forward {same}")  # fmt: skip
