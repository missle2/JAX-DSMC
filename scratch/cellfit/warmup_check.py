"""How long must the empty `circle_long` box (examples/mesh/forward.py) warm up before the
windowed drag is steady?  Forward only (no capture): the windowed drag after warm-ups of
700 .. 2800 steps, free-molecular and Kn 0.5, fnum x2, 3 seeds; the 10 m `circle` box at
its production 700 as the reference.  Prints mean +- se per setting.

    PYTHONPATH=. python3 scratch/cellfit/warmup_check.py [kn,..] [warmup,..] [seeds]
(defaults 0,0.5  700,1400,2100,2800  3; the 10 m reference only with the defaults)
"""

import sys
import time

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402


def drag(cfg, seed):
    _, _, diag, _, tallies, s = EM.forward(
        cfg, "radius", EM.nominal(cfg, "radius"), jax.random.key(32 + seed), False
    )
    EM._check(diag)
    scale = float(s["params"].fnum[0]) / (cfg["window"] * float(s["params"].dt))
    return EM.objective_value(cfg, tallies, s["static"].D, scale)


def main():
    kns = [float(k) for k in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.0, 0.5]
    warms = [int(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2 else [700, 1400, 2100, 2800]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for kn in kns:
        rows = [("circle", 700)] if len(sys.argv) == 1 else []
        rows += [("circle_long", w) for w in warms]
        for case, warm in rows:
            cfg = OPT.default_config()
            cfg.update(case=case, fnum_scale=2, grid_scale=1, levelset_h=0.0625,
                       levelset_samples=16, grazing=0.05, knudsen=kn, a_max=64, warmup=warm)
            t0 = time.time()
            j = np.array([drag(cfg, i) for i in range(seeds)])
            print(f"Kn {kn:g} {case:12s} warmup {warm:5d}: drag {j.mean():.5e} +- "
                  f"{j.std(ddof=1) / np.sqrt(len(j)):.1e}  ({(time.time() - t0) / seeds:.0f} s/seed)",
                  flush=True)


if __name__ == "__main__":
    main()
