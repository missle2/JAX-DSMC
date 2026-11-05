"""Time pooling against ensemble pooling (2026-11-05): does the near-wall window fit need the
quasi-steady assumption, or only data?

The window fit (scratch/fitlab/window.py) pools a seed's near-wall nodes over the whole
captured window -- unbiased for the pathwise sum when the flow's statistics are stationary
over it, not for an unsteady flow.  Pooling across independent realizations at the SAME
steps needs no such assumption (lambda is an ensemble expectation).  Per setting:

* ``self`` with ``--block B``: the seed's own nodes in B-step blocks (time-local, 300 / B
  times less data than the full window);
* ``ensemble`` with ``--block B``: every seed's nodes in the same B-step blocks (8 seeds x
  38 steps ~ one seed's 300), the hit's own node left out.

Closure rows to ``out/ens_<case>_<mode>_b<B>.json`` (fit ``win_...``), scored by varsum.py.

    PYTHONPATH=. python3 scratch/fitlab/ensemble.py --case drag --mode ensemble --block 38
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import knn  # noqa: E402
import lab  # noqa: E402
import window  # noqa: E402

D_ENS = 0.15  # nodes kept (the 2 cm windows and the 64-count floor never reach further)
SETTING = (64, 0.02, 0.4, 0.5)  # k, width floor, sector, velocity box: the M 10 best


def load_nodes(seed, path):
    z = lab.load(path, ("node_step", "node_slot", "node_x", "node_v", "node_lam"))
    nc, _th = knn.coordinates(z["node_x"], z["node_v"], z["node_step"])
    keep = (nc[:, 2] > 0) & (nc[:, 0] < D_ENS)
    lam = lab.demeaned({"node_lam": z["node_lam"][keep], "node_step": z["node_step"][keep]})
    key = (seed * 1_000_000_000_000 + z["node_step"][keep].astype(np.int64) * 1_000_000
           + z["node_slot"][keep])  # fmt: skip
    return nc[keep], lam, key


def closure_row(seed, path, nc, lam, nkey):
    k, w, sector, dv = SETTING
    h = lab.load(path, ("hit_step", "hit_slot", "hit_x", "hit_v", "hit_dx", "hit_gx"))
    meta = json.loads(path.with_suffix(".json").read_text())
    hc, hth = knn.coordinates(h["hit_x"], h["hit_v"], h["hit_step"])
    hkey = (seed * 1_000_000_000_000 + h["hit_step"].astype(np.int64) * 1_000_000
            + h["hit_slot"])  # fmt: skip
    g, count = window.window_gradients(nc, lam, nkey, hc, hkey, [k], [w], sector, dv,
                                       meta["dt"])[(k, w)]  # fmt: skip
    n_hat = np.stack((np.cos(hth), np.sin(hth)), axis=1)
    t_hat = np.stack((-np.sin(hth), np.cos(hth)), axis=1)
    fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
    grad = g[:, :1] * n_hat + g[:, 1:2] * t_hat
    fits["win"] = np.where((count >= window.MIN_NODES)[:, None], grad, fits[window.FALLBACK])
    px = {c: {name: float(np.sum(gg * h["hit_dx"][i])) for name, gg in fits.items()}
          for i, c in enumerate(meta["controls"])}  # fmt: skip
    totals = {}
    for c in meta["controls"]:
        t = meta["production"][c]["terms"][lab.BASE_VARIANT]
        totals[c] = sum(t[kk] for kk in lab.TERMS) + meta["production"][c]["volume"]
    return {"seed": seed, "pathwise_x": px, "base_total": totals,
            "fallback": float(np.mean(count < window.MIN_NODES))}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--mode", choices=("self", "ensemble"), default="ensemble")
    ap.add_argument("--block", type=int, default=0, help="time block, steps (0: whole window)")
    args = ap.parse_args()
    window.BLOCK = args.block
    files = lab.seed_files(lab.CASES[args.case])
    rows = []
    t0 = time.time()
    if args.mode == "self":
        for seed, path in files:
            rows.append(closure_row(seed, path, *load_nodes(seed, path)))
            print(f"  {args.case} self b{args.block} seed {seed}: fallback "
                  f"{100 * rows[-1]['fallback']:.1f} %  {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    else:
        parts = [load_nodes(seed, path) for seed, path in files]
        nc = np.concatenate([p[0] for p in parts])
        lam = np.concatenate([p[1] for p in parts])
        nkey = np.concatenate([p[2] for p in parts])
        del parts
        print(f"  {args.case} ensemble: {nc.shape[0]} nodes from {len(files)} seeds", flush=True)
        for seed, path in files:
            rows.append(closure_row(seed, path, nc, lam, nkey))
            print(f"  {args.case} ensemble b{args.block} seed {seed}: fallback "
                  f"{100 * rows[-1]['fallback']:.1f} %  {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    out = lab.OUT / f"ens_{args.case}_{args.mode}_b{args.block}.json"
    out.write_text(json.dumps({"case": args.case, "settings": vars(args), "rows": rows}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
