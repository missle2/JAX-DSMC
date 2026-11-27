"""The specular term budget (2026-11-27, paper plan phase B gate 1): which term carries the
shared +5-10 % specular excess?

Per specular case (spec = M 10, m03_spec = M 0.32, m25_spec = M 25; all Kn 0.5, circle_long)
and per control with FDs (radius, mode2), the seed means of every adjoint term under each
production variant the dump carries (``meta["production"][c]["terms"][variant]``) and of
pathwise_x under the kNN fits (``out/knn3_<case>_C.json``: the chosen C1 and its neighbours),
against the FD mean.  The excess (adjoint - FD) is set beside each term's mean and seed spread:
a term whose variation across fits or cases tracks the excess is the suspect.

    python3 scratch/fitlab/spec_budget.py [--cases spec m03_spec m25_spec]
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

KNN = ("k3_C1_0.25_0.033_0.3_o0", "k3_C0.5_0.25_0.033_0.6_o0", "k3_C2_0.25_0.033_0.6_o0")
CONTROLS = ("radius", "mode2")


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / len(x) ** 0.5 if len(x) > 1 else 0.0)


def budget(case):
    files = lab.seed_files(lab.CASES[case])
    metas = {s: json.loads(p.with_suffix(".json").read_text()) for s, p in files}
    fds = lab.finite_differences(case)
    knn_path = lab.OUT / f"knn3_{case}_C.json"
    knn = ({r["seed"]: r for r in json.loads(knn_path.read_text())["rows"]}
           if knn_path.exists() else {})  # fmt: skip
    variants = metas[files[0][0]]["variants"]
    out = {}
    for c in CONTROLS:
        fd = fds.get(c, {})
        if not fd:
            continue
        fd_m, fd_se = mean_se(list(fd.values()))
        print(f"\n{case} {c}: FD {fd_m:+.5g} +- {fd_se:.2g} ({len(fd)} seeds)")
        rows = {}
        for v in variants:
            terms = {t: [m["production"][c]["terms"][v][t] for m in metas.values()]
                     for t in lab.TERMS}  # fmt: skip
            terms["volume"] = [m["production"][c]["volume"] for m in metas.values()]
            total = np.sum([terms[t] for t in terms], axis=0)
            rows[v] = (terms, total)
        for name in KNN:
            if not knn:
                break
            terms, _ = rows[lab.BASE_VARIANT]
            seeds = [s for s in metas if s in knn]
            px = [knn[s]["pathwise_x"][c][name] for s in seeds]
            other = [knn[s]["base_total"][c] - metas[s]["production"][c]["terms"][
                lab.BASE_VARIANT]["pathwise_x"] for s in seeds]  # fmt: skip
            t2 = dict(terms)
            t2["pathwise_x"] = px
            rows[name] = (t2, np.add(other, px))
        head = "".join(f"{t[:9]:>11s}" for t in (*lab.TERMS, "volume"))
        print(f"    {'fit':30s}{head}{'total':>11s}{'ratio':>8s}{'excess/se':>10s}")
        out[c] = {"fd": [fd_m, fd_se]}
        for name, (terms, total) in rows.items():
            tm, tse = mean_se(total)
            cells = "".join(f"{np.mean(terms[t]):+11.4g}" for t in (*lab.TERMS, "volume"))
            z = (tm - fd_m) / np.hypot(tse, fd_se)
            print(f"    {name[:30]:30s}{cells}{tm:+11.4g}{tm / fd_m:8.3f}{z:+10.1f}")
            out[c][name] = {"total": [tm, tse],
                            **{t: [*mean_se(terms[t])] for t in (*lab.TERMS, "volume")}}  # fmt: skip
        # the share each term would need to absorb: excess as a fraction of each term's mean
        _, base_total = rows[lab.BASE_VARIANT]
        excess = np.mean(base_total) - fd_m
        print(f"    excess {excess:+.4g}  = "
              + ", ".join(f"{t} x {excess / np.mean(rows[lab.BASE_VARIANT][0][t]):+.2f}"
                          for t in ("pathwise_x", "pathwise_v", "pathwise_tally", "volume")
                          if abs(np.mean(rows[lab.BASE_VARIANT][0][t])) > 1e-12))  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["spec", "m03_spec", "m25_spec"])
    args = ap.parse_args()
    res = {case: budget(case) for case in args.cases}
    path = lab.OUT / "spec_budget.json"
    path.write_text(json.dumps(res))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
