"""lcloss: each loss variant's validation on the training cases against the base operator,
with the pass rule fixed before the runs (study.py _lcloss_jobs)."""

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "examples/learned_collision/results"
BASE = {"S3": 1.065, "S4": 1.131,  # mu ratio, lcbase_shear_hi / lctime validation
        "K1": (1.047, 0.868), "K4": (1.104, 0.971), "K8": (1.170, 0.948)}  # eval_bound_*


def main(prefix="lcloss"):
    """``python scratch/lcloss_table.py [lcloss|lcchain]``."""
    keys = ("ctl", "q1", "q2", "q3", "q10", "m1", "m2", "m4", "m2q3", "ctl_long", "mlp_q2",
            "nok1", "nok1_mlp", "k4k8", "norm", "norm_mlp", "only_K1", "only_K4", "only_K8",
            "ref", "decay", "lr1", "pair", "b20", "b20decay", "pair_b20", "mlp_q1",
            "only_K8", "cubic_q1", "mlp_cubic_q1")
    for k in keys:
        model = "mlp" if "mlp" in k else "linear"
        path = RES / f"train_{model}_{prefix}_{k}.json"
        if not path.exists():
            continue
        v = json.loads(path.read_text())["validation"]
        cells, ok = [], True
        for c in ("S3", "S4"):
            if c not in v:
                continue
            mu = v[c]["mu_ratio"]
            ok &= abs(mu - 1) <= abs(BASE[c] - 1) + v[c]["mu_ratio_se"]
            cells.append(f"{c} mu {mu:.3f}")
        for c in ("K1", "K4", "K8"):
            if c not in v:
                continue
            t, q = v[c]["int_txx_ratio"], v[c]["int_qx_ratio"]
            st, sq = v[c]["int_txx_ratio_se"], v[c]["int_qx_ratio_se"]
            bt, bq = BASE[c]
            if c == "K1":  # no worse than the base
                ok &= abs(t - 1) <= abs(bt - 1) + 2 * st and abs(q - 1) <= abs(bq - 1) + 2 * sq
            else:  # both closer to 1
                ok &= abs(t - 1) < abs(bt - 1) and abs(q - 1) < abs(bq - 1)
            cells.append(f"{c} txx {t:.3f} qx {q:.3f} L2 {v[c]['l2_txx']:.3f}/{v[c]['l2_qx']:.3f}")
        log = json.loads(path.read_text())["log"]
        tail = log[-max(1, len(log) // 5):]
        pairs = " ".join(f"{c}:{sum(r['pair'][c] for r in tail) / len(tail):+.3f}"
                         for c in tail[0]["pair"] if c.startswith("K"))
        cells.append(f"pair(last 1/5) {pairs}")
        a = json.loads(path.read_text())["theta"].get("A")
        rows = "" if a is None else " | A rows " + " ".join(f"{sum(r) / len(r):+.4f}" for r in a)
        print(f"{k:8s} {'PASS' if ok else 'fail'} | " + " | ".join(cells) + rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
