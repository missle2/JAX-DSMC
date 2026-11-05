"""Where the fits differ (2026-11-04): at one seed's hits, the wall-normal derivative of the
pooled reference (scratch/fitlab/lab.py's saved coefficients) against the production fits,
binned by angle from the stagnation point and by speed -- weighted by the hits' normal
displacement under the nose bump (what the nose term sums), and the identity's
-(T + L* V) / |v| against the reference's along-flight derivative.

    PYTHONPATH=. python3 scratch/fitlab/profile.py --case drag --seed 0 [--tag ""]
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch/fitlab")
import lab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="drag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    gate = lab.CASES[args.case]
    z = np.load(lab.OUT / f"coef_{args.case}{args.tag}.npz")
    coef = dict(zip(z["ids"].tolist(), z["coef"], strict=True))
    settings = dict(lab.SETTINGS)
    ref_json = json.loads((lab.OUT / f"ref_{args.case}{args.tag}.json").read_text())
    for k, v in ref_json["settings"].items():
        settings[k] = tuple(v) if isinstance(v, list) else v
    path = dict(lab.seed_files(gate))[args.seed]
    meta = json.loads(path.with_suffix(".json").read_text())
    h = lab.load(path, ("hit_step", "hit_x", "hit_v", "hit_lt", "hit_coll", "hit_dx", "hit_gx"))
    g_ref, _value, _gid, _gv = lab.model_gradient(h["hit_x"], h["hit_v"], h["hit_step"], coef,
                                              settings)  # fmt: skip
    d, th, vn, vt, vz = lab.frame(h["hit_x"], h["hit_v"])
    rel = h["hit_x"] - lab.CENTRE
    n = rel / np.hypot(rel[:, 0], rel[:, 1])[:, None]
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (th - np.pi)))))  # 0 at the stagnation point
    speed = np.sqrt(vn**2 + vt**2 + vz**2)
    vd = h["hit_v"][:, :2]
    vmag = np.sqrt(np.sum(vd * vd, axis=1))
    ident = -(h["hit_lt"] + h["hit_coll"] / meta["dt"]) / vmag  # along-flight derivative
    along_ref = np.sum(g_ref * vd, axis=1) / vmag
    names = {name: h["hit_gx"][j] for j, name in enumerate(meta["variants"])}
    c_nose = meta["controls"].index("bump_nose")
    w = np.sum(h["hit_dx"][c_nose] * n, axis=1)  # the normal displacement per unit control
    ok = ~np.isnan(g_ref[:, 0])
    print(f"{args.case} seed {args.seed}: {ok.sum()} hits; d at hits: median "
          f"{np.median(d):.4f} m, p99 {np.percentile(d, 99):.4f}")  # fmt: skip
    print("\nnormal derivative, nose-bump weighted sum by angle from stagnation (deg):")
    head = "  ".join(f"{k[:14]:>14s}" for k in ("ref", *names))
    print(f"  {'angle':>9s} {'hits':>7s}  {head}")
    for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 180)):
        sel = ok & (ang >= lo) & (ang < hi)
        vals = [np.sum(np.sum(g_ref[sel] * n[sel], axis=1) * w[sel])]
        vals += [np.sum(np.sum(g[sel] * n[sel], axis=1) * w[sel]) for g in names.values()]
        print(f"  {lo:3d}-{hi:<5d} {sel.sum():7d}  " + "  ".join(f"{x:+14.4g}" for x in vals))
    print("\nwithin 30 deg of stagnation, by speed / V_REF: mean d(lambda)/dn and the along-flight")
    print("derivative (reference v.grad / |v| vs the identity -(T + L*V) / |v|):")
    print(f"  {'speed':>8s} {'hits':>7s} {'ref dn':>11s} " + " ".join(
        f"{k[:13]:>13s}" for k in names) + f" {'ref along':>11s} {'identity':>11s}")  # fmt: skip
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 20)):
        sel = ok & (ang < 30) & (speed >= lo) & (speed < hi)
        if sel.sum() < 20:
            continue
        dn = [np.mean(np.sum(g[sel] * n[sel], axis=1)) for g in (g_ref, *names.values())]
        print(f"  {lo:3.1f}-{hi:<4.1f} {sel.sum():7d} " + " ".join(f"{x:+11.4g}" for x in dn[:1])
              + " " + " ".join(f"{x:+13.4g}" for x in dn[1:])
              + f" {np.mean(along_ref[sel]):+11.4g} {np.mean(ident[sel]):+11.4g}")  # fmt: skip


if __name__ == "__main__":
    main()
