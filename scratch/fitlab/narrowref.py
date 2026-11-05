"""The narrow-cell wall-slope reference in closure (2026-11-05): the outgoing nodes pooled
over the seeds, grouped by angular sector x step block x a narrow 3-D wall-frame velocity cell
(``--dv`` V_REF wide), lambda regressed linearly on the wall distance over [0, W] per group --
the least-assumption estimate of d(lambda)/dd at the wall (no velocity model: the velocity
mix cannot fake a slope).  At each hit the normal component of the production fit is
replaced by that slope (groups with fewer than ``--min-nodes`` keep production's), the
tangential part kept; pathwise_x per control into the lab's closure format
(``out/nar_<case>_dv<dv>.json``: fits ``nar<W>_<variant>``).

    PYTHONPATH=. python3 scratch/fitlab/narrowref.py --case drag [--dv 0.25]
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402

WINDOWS = (0.01, 0.02, 0.03, 0.05, 0.1)
VARIANTS = ("cell_along_pool_plane", "cell_pool_plane")


def keys_of(th, step, vn, vt, vz, dv, sectors, block):
    sector = np.floor((th + np.pi) / (2 * np.pi) * sectors).astype(np.int64) % sectors
    i = np.clip(np.floor(vn / dv).astype(np.int64), 0, 999)
    j = np.clip(np.floor(vt / dv).astype(np.int64) + 1000, 0, 1999)
    k = np.clip(np.floor(np.abs(vz) / dv).astype(np.int64), 0, 999)
    return (((step // block) * sectors + sector) * 1000 + i) * 2000 * 1000 + j * 1000 + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--dv", type=float, default=0.25)
    ap.add_argument("--sectors", type=int, default=36)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--min-nodes", type=int, default=20)
    args = ap.parse_args()
    gate = lab.CASES[args.case]
    files = lab.seed_files(gate)
    # per window: concatenated (key, n, sd, sdd, sl, sdl) over the seeds, reduced at the end
    parts = {w: [] for w in WINDOWS}
    for seed, path in files:
        z = lab.load(path, ("node_step", "node_x", "node_v", "node_lam"))
        d, th, vn, vt, vz = lab.frame(z["node_x"], z["node_v"])
        out = (vn > 0) & (d < max(WINDOWS))
        lam = lab.demeaned({"node_lam": z["node_lam"][vn > 0], "node_step": z["node_step"][vn > 0]})
        lam_full = np.zeros(vn.size)
        lam_full[vn > 0] = lam
        key = keys_of(th, z["node_step"], vn, vt, vz, args.dv, args.sectors, args.block)
        del z
        for w in WINDOWS:
            m = out & (d < w)
            k, inv = np.unique(key[m], return_inverse=True)
            dm, lm = d[m], lam_full[m]
            sums = [np.bincount(inv, weights=x, minlength=k.size)
                    for x in (np.ones(dm.size), dm, dm * dm, lm, dm * lm)]  # fmt: skip
            parts[w].append((k, np.stack(sums, axis=1)))
        print(f"  nodes seed {seed}", flush=True)
    slope_of = {}
    for w in WINDOWS:
        k = np.concatenate([p[0] for p in parts[w]])
        s = np.concatenate([p[1] for p in parts[w]])
        uk, inv = np.unique(k, return_inverse=True)
        tot = np.zeros((uk.size, 5))
        np.add.at(tot, inv, s)
        n, sd, sdd, sl, sdl = tot.T
        var = sdd / np.maximum(n, 1) - (sd / np.maximum(n, 1)) ** 2
        ok = (n >= args.min_nodes) & (var > 0)
        slope = np.where(ok, (sdl / np.maximum(n, 1) - sd * sl / np.maximum(n, 1) ** 2)
                         / np.where(var > 0, var, 1.0), np.nan)  # fmt: skip
        slope_of[w] = (uk, slope)
    del parts
    rows = []
    for seed, path in files:
        h = lab.load(path, ("hit_step", "hit_x", "hit_v", "hit_dx", "hit_gx"))
        meta = json.loads(path.with_suffix(".json").read_text())
        _d, th, vn, vt, vz = lab.frame(h["hit_x"], h["hit_v"])
        key = keys_of(th, h["hit_step"], vn, vt, vz, args.dv, args.sectors, args.block)
        rel = h["hit_x"] - lab.CENTRE
        nrm = rel / np.hypot(rel[:, 0], rel[:, 1])[:, None]
        fits = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
        cover = {}
        for w, (uk, slope) in slope_of.items():
            pos = np.clip(np.searchsorted(uk, key), 0, uk.size - 1)
            sl = np.where(uk[pos] == key, slope[pos], np.nan)
            have = ~np.isnan(sl)
            cover[w] = float(have.mean())
            for name in VARIANTS:
                g = fits[name]
                gn = np.sum(g * nrm, axis=1)
                new = np.where(have, sl, gn)
                fits[f"nar{w}_{name[:9]}"] = g + (new - gn)[:, None] * nrm
        px = {c: {name: float(np.sum(g * h["hit_dx"][i])) for name, g in fits.items()}
              for i, c in enumerate(meta["controls"])}  # fmt: skip
        totals = {}
        for c in meta["controls"]:
            t = meta["production"][c]["terms"][lab.BASE_VARIANT]
            totals[c] = sum(t[k] for k in lab.TERMS) + meta["production"][c]["volume"]
        rows.append({"seed": seed, "pathwise_x": px, "base_total": totals, "cover": cover})
        print(f"  hits seed {seed}: cover " + " ".join(f"{w}:{c:.2f}" for w, c in cover.items()),
              flush=True)  # fmt: skip
    lab.OUT.mkdir(parents=True, exist_ok=True)
    out = lab.OUT / f"nar_{args.case}_dv{args.dv}_s{args.sectors}_b{args.block}.json"
    out.write_text(json.dumps({"case": args.case, "settings": vars(args), "rows": rows}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
