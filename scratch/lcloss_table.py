"""lcloss: each loss variant's validation on the training cases against the base operator,
with the pass rule fixed before the runs (study.py _lcloss_jobs)."""

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "examples/learned_collision/results"
BASE = {"S3": 1.065, "S4": 1.131,  # mu ratio, lcbase_shear_hi / lctime validation
        "K1": (1.047, 0.868), "K4": (1.104, 0.971), "K8": (1.170, 0.948)}  # eval_bound_*


def main():
    for k in ("ctl", "q3", "q10", "m1", "m2", "m4", "m2q3"):
        path = RES / f"train_linear_lcloss_{k}.json"
        if not path.exists():
            print(f"{k:5s} missing")
            continue
        v = json.loads(path.read_text())["validation"]
        cells, ok = [], True
        for c in ("S3", "S4"):
            mu = v[c]["mu_ratio"]
            ok &= abs(mu - 1) <= abs(BASE[c] - 1) + v[c]["mu_ratio_se"]
            cells.append(f"{c} mu {mu:.3f}")
        for c in ("K1", "K4", "K8"):
            t, q = v[c]["int_txx_ratio"], v[c]["int_qx_ratio"]
            st, sq = v[c]["int_txx_ratio_se"], v[c]["int_qx_ratio_se"]
            bt, bq = BASE[c]
            if c == "K1":  # no worse than the base
                ok &= abs(t - 1) <= abs(bt - 1) + 2 * st and abs(q - 1) <= abs(bq - 1) + 2 * sq
            else:  # both closer to 1
                ok &= abs(t - 1) < abs(bt - 1) and abs(q - 1) < abs(bq - 1)
            cells.append(f"{c} txx {t:.3f} qx {q:.3f} L2 {v[c]['l2_txx']:.3f}/{v[c]['l2_qx']:.3f}")
        print(f"{k:5s} {'PASS' if ok else 'fail'} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
