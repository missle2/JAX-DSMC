"""Local wall slope with minimal assumptions (2026-11-05): within ``--angle`` deg of the
stagnation point, outgoing nodes pooled over the seeds are binned in narrow 3-D velocity cells
(wall frame, ``--dv`` V_REF wide) and in step blocks; per (velocity cell, block) lambda is
regressed linearly on the wall distance over [0, W] for several windows W.  The slopes are
averaged with the hits' weights (the hits in that velocity cell and block) and compared
with the production fits' normal derivative at the same hits.

    PYTHONPATH=. python3 scratch/fitlab/narrow.py --case drag
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

WINDOWS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5)
QUAD = False


def vcell(vn, vt, vz, dv):
    i = np.floor(vn / dv).astype(np.int64)
    j = np.floor(vt / dv).astype(np.int64) + 200
    k = np.floor(np.abs(vz) / dv).astype(np.int64)
    return (i * 400 + j) * 100 + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--angle", type=float, default=20.0)
    ap.add_argument("--dv", type=float, default=0.25)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--quad", action="store_true", help="slope at d = 0 of a quadratic in d")
    args = ap.parse_args()
    global QUAD  # noqa: PLW0603 -- a scratch switch
    QUAD = args.quad
    gate = lab.CASES[args.case]
    # sums per (window, key): n, sum d, sum d^2, sum lam, sum d lam
    acc = [dict() for _ in WINDOWS]
    hit_keys, prod_gn = [], {}
    for seed, path in lab.seed_files(gate):
        z = lab.load(path, ("node_step", "node_x", "node_v", "node_lam"))
        d, th, vn, vt, vz = lab.frame(z["node_x"], z["node_v"])
        out = vn > 0
        lam = lab.demeaned({"node_lam": z["node_lam"][out], "node_step": z["node_step"][out]})
        d, th, vn, vt, vz = d[out], th[out], vn[out], vt[out], vz[out]
        step = z["node_step"][out]
        del z
        ang = np.degrees(np.abs(np.angle(np.exp(1j * (th - np.pi)))))
        sel = ang < args.angle
        key = vcell(vn, vt, vz, args.dv) * 10 + step // args.block
        for w, a in zip(WINDOWS, acc, strict=True):
            m = sel & (d < w)
            k, inv = np.unique(key[m], return_inverse=True)
            dm, lm = d[m], lam[m]
            sums = [np.bincount(inv, weights=x, minlength=k.size)
                    for x in (np.ones(m.sum()), dm, dm**2, lm, dm * lm, dm**3, dm**4, dm**2 * lm)]  # fmt: skip
            for i, kk in enumerate(k.tolist()):
                cur = a.get(kk)
                if cur is None:
                    a[kk] = np.array([s[i] for s in sums])
                else:
                    cur += [s[i] for s in sums]
        h = lab.load(path, ("hit_step", "hit_x", "hit_v", "hit_gx"))
        meta = json.loads(path.with_suffix(".json").read_text())
        hd, hth, hvn, hvt, hvz = lab.frame(h["hit_x"], h["hit_v"])
        hang = np.degrees(np.abs(np.angle(np.exp(1j * (hth - np.pi)))))
        hs = hang < args.angle
        rel = h["hit_x"] - lab.CENTRE
        n = rel / np.hypot(rel[:, 0], rel[:, 1])[:, None]
        hit_keys.append(vcell(hvn, hvt, hvz, args.dv)[hs] * 10 + h["hit_step"][hs] // args.block)
        for j, name in enumerate(meta["variants"]):
            prod_gn.setdefault(name, []).append(np.sum(h["hit_gx"][j][hs] * n[hs], axis=1))
        print(f"  seed {seed}", flush=True)
    keys = np.concatenate(hit_keys)
    print(f"\n{args.case}: {keys.size} hits within {args.angle} deg; production mean normal "
          "derivative at them:")  # fmt: skip
    for name, g in prod_gn.items():
        print(f"  {name:24s} {np.mean(np.concatenate(g)):+.4e}")
    print("narrow-cell local slope d(lambda)/dd, hit-weighted (cells with >= 30 nodes):")
    for w, a in zip(WINDOWS, acc, strict=True):
        slopes, weights = [], []
        uk, cnt = np.unique(keys, return_counts=True)
        covered = 0
        for kk, c in zip(uk.tolist(), cnt.tolist(), strict=True):
            s = a.get(kk)
            if s is None or s[0] < 30:
                continue
            nn, sd, sdd, sl, sdl, s3, s4, sddl = s
            if QUAD:
                # slope at d = 0 of the least-squares quadratic
                mat = np.array([[nn, sd, sdd], [sd, sdd, s3], [sdd, s3, s4]])
                if s[0] < 60 or np.linalg.cond(mat) > 1e12:
                    continue
                slopes.append(np.linalg.solve(mat, [sl, sdl, sddl])[1])
            else:
                var = sdd / nn - (sd / nn) ** 2
                if var <= 0:
                    continue
                slopes.append((sdl / nn - sd * sl / nn**2) / var)
            weights.append(c)
            covered += c
        slopes, weights = np.array(slopes), np.array(weights, float)
        mean = np.sum(slopes * weights) / weights.sum()
        # spread of the cell slopes -> a rough se of the weighted mean
        se = np.sqrt(np.sum(weights**2 * (slopes - mean) ** 2)) / weights.sum()
        print(f"  window {100 * w:5.1f} cm: {mean:+.4e} +- {2 * se:.1e}   ({covered} of "
              f"{keys.size} hits covered)")  # fmt: skip


if __name__ == "__main__":
    main()
