"""Does the warm start keep the common-random-number coupling between iterates?

Per seed, on adjacent iterates A -> B of an optm run: cold J(A), cold J(B) and warm J(B)
(started from A's final gas, ``restart_steps`` re-equilibration).  The spread over seeds of
J(B) - J(A) (cold-cold: what the trust region's rho sees now) against warm J(B) - cold J(A),
and the mean warm - cold J(B) (the warm start's bias).

    PYTHONPATH=. python3 scratch/warmstart/crn.py OUT.json RUN_DIR A B SEEDS R
"""

import json
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "scratch/warmstart")
from relax import body  # noqa: E402

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402


def windowed(cfg, tallies, setup):
    p = setup["params"]
    scale = float(p.fnum[0]) / (cfg["window"] * float(p.dt))
    return EM.objective_value(cfg, tallies, setup["static"].D, scale)


def main():
    out, run, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seeds, r = int(sys.argv[5]), int(sys.argv[6])
    with open(f"{run}/history.json") as fh:
        cfg = dict(OPT.default_config(), **json.load(fh)["config"])
    ls_a, ls_b = body(cfg, run, a), body(cfg, run, b)
    warm_cfg = dict(cfg, restart_steps=r)
    rows = []
    for i in range(seeds):
        key = jax.random.key(cfg["seed_start"] + i)
        t0 = time.time()
        res = {}
        for name, c, ls, restart in (("cold_a", cfg, ls_a, None), ("cold_b", cfg, ls_b, None)):
            _s0, _f, diag, _r, t, s = EM.forward(c, "radius", 0.0, key, False, ls, restart=restart)
            EM._check(diag)
            res[name] = windowed(c, t, s)
            if name == "cold_a":
                final_a = s["final"]
        t1 = time.time()
        _s0, _f, diag, _r, t, s = EM.forward(warm_cfg, "radius", 0.0, key, False, ls_b,
                                             restart=final_a)  # fmt: skip
        EM._check(diag)
        res["warm_b"] = windowed(warm_cfg, t, s)
        res["seconds_warm"] = time.time() - t1
        res["seconds"] = time.time() - t0
        rows.append(res)
        print(i, json.dumps(res), flush=True)
        with open(out, "w") as fh:
            json.dump({"a": a, "b": b, "restart_steps": r, "rows": rows}, fh, indent=1)
    j = {k: np.array([x[k] for x in rows]) for k in ("cold_a", "cold_b", "warm_b")}
    ref = abs(j["cold_a"].mean())
    for name, d in (("cold B - cold A", j["cold_b"] - j["cold_a"]),
                    ("warm B - cold A", j["warm_b"] - j["cold_a"]),
                    ("warm B - cold B", j["warm_b"] - j["cold_b"])):  # fmt: skip
        print(f"{name}: mean {d.mean() / ref * 100:+.2f} %  sd {d.std(ddof=1) / ref * 100:.2f} %")
    print(f"per-seed sd of J: {j['cold_a'].std(ddof=1) / ref * 100:.2f} %")


if __name__ == "__main__":
    main()
