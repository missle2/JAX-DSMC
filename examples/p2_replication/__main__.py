"""Run the [P2] section 6 replication.

python3 -m examples.p2_replication              # everything
python3 -m examples.p2_replication --part euler # one part
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import numpy as np
from jax import random

from examples import p2_replication as p2

PART = ("forward", "euler", "replicate", "alpha")


def _hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def part_forward(args):
    _hdr("1. forward physics: does the 0-D relaxation match [P2]'s setup?")
    print("   [P2]: T^0 = (1, 1, 0.5) relaxes towards T_M = 0.8333 but only PARTIALLY")
    print("   at t = 2 (main.tex:1454).  N_c/N = dt Sigma_v/eps sets how far it gets.\n")
    print(
        f"   {'beta':>4} {'omega':>6} {'N_c/N':>7} {'Sigma':>7} {'Sigma used':>11} "
        f"{'T_x':>8} {'T_y':>8} {'T_z':>8} {'mean':>8}"
    )
    for b in (0, 1, 2):
        cs = p2.case(b, args.n_part, npc_target=args.npc)
        m = np.asarray(p2.moments_of(p2.T0_REF, cs, random.PRNGKey(0)))
        print(
            f"   {b:>4} {p2.omega_of(b):>6.1f} {p2.expected_collision_fraction(b):>7.2f} "
            f"{p2.majorant(b):>7.2f} {p2.majorant_used(cs):>11.2f} "
            f"{m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>8.4f} {m.mean():>8.4f}",
            flush=True,
        )
    print("\n   `mean` is a collision invariant and must be IDENTICAL across beta; it sits")
    print("   slightly off 0.8333 only because the IC is sampled at finite N.")
    print("   At beta = 0 `Sigma used` is 2x [P2]'s -- see `majorant_used` for why, and why")
    print("   it changes the variance but not the gradient.")


def part_euler(args):
    _hdr("2. an ANALYTIC check on the gradient, with no finite difference")
    print("   At beta = 0 the rate q = prefactor is velocity-independent, so the dynamics")
    print("   are scale-invariant and T_l is homogeneous of degree 1 in m.  Euler:")
    print("       sum_p T_p^0 dT_l/dT_p^0 = T_l   exactly, for beta = 0 ONLY.")
    print("   The RHS is a FORWARD quantity, so an adjoint error cannot cancel from both")
    print("   sides.  beta > 0 must FAIL, by a margin growing with beta.\n")
    print(f"   {'beta':>4} {'sum T_p dT_y/dT_p':>19} {'T_y':>12} {'rel residual':>14}  verdict")
    for b in (0, 1, 2):
        cs = p2.case(b, args.n_part, npc_target=args.npc)
        lhs, rhs, res = p2.euler_identity_residual(p2.T0_REF, cs, random.PRNGKey(7))
        if b == 0:
            v = "PASS (exact)" if res < 1e-12 else f"FAIL -- expected 0, got {res:.1e}"
        else:
            v = "fails as predicted" if res > 1e-6 else "UNEXPECTED: identity should not hold"
        print(f"   {b:>4} {lhs:>19.9f} {rhs:>12.6f} {res:>14.2e}  {v}", flush=True)


def part_replicate(args):
    _hdr("3. [P2] section 6.1 replication: e = |mean AD - mean FD| vs N")
    print(f"   Their protocol verbatim: M_s = {args.keys} i.i.d. runs with random ICs on BOTH")
    print(f"   sides, Delta m = {p2.DELTA_M}, M = {p2.N_STEPS} steps, dt = {p2.DT}, phi = v_y^2.")
    print("   [P2] expects e to fall with N and then PLATEAU at the O(Delta m^2) finite")
    print("   difference error (main.tex:1423).\n")
    keys = random.split(random.PRNGKey(2024), args.keys)
    for b in args.betas:
        print(f"   --- beta = {b}  (omega = {p2.omega_of(b)})")
        print(f"   {'N':>8} {'A_max':>6} {'dTy/dTx':>22} {'dTy/dTy':>22} {'dTy/dTz':>22} {'s':>5}")
        print(f"   {'':>8} {'':>6} " + " ".join(f"{'AD / FD / e':>22}" for _ in range(3)))
        for n in args.n_list:
            cs = p2.case(b, n, npc_target=args.npc)
            t0 = time.time()
            ad, fd = p2.ensemble(cs, keys, chunk=args.chunk)
            e, adm, fdm, ads, fds = (np.asarray(z) for z in p2.error_metric(ad, fd))
            cells = [f"{adm[p]:.4f}/{fdm[p]:.4f}/{e[p]:.1e}".rjust(22) for p in range(3)]
            print(
                f"   {n:>8} {cs.a_max:>6} " + " ".join(cells) + f" {time.time() - t0:>5.0f}",
                flush=True,
            )
            print(
                f"   {'':>8} {'+-SE':>6} "
                + " ".join(
                    f"{'+-' + format(ads[p], '.4f') + '/' + format(fds[p], '.4f'):>22}"
                    for p in range(3)
                )
            )


def part_alpha(args):
    _hdr("4. POSITIVE CONTROL: is dJ/d(alpha) detectable when it should be?")
    print("   The shock study found E[dT_y/d(alpha)] consistent with ZERO at every N, with")
    print("   a large per-run spread.  FD agreed independently, but 'both say zero' is weak.")
    print("   This is 0-D homogeneous relaxation: alpha sets the scattering anisotropy and")
    print("   therefore the rate T_y relaxes towards T_M, with no transport or boundary in")
    print("   the way.  If dT_y/d(alpha) is resolvable ANYWHERE, it is here.\n")
    keys = random.split(random.PRNGKey(99), args.keys)
    print(
        f"   {'beta':>4} {'alpha':>6} {'N':>8} {'AD mean':>12} {'+-SE':>10} "
        f"{'FD mean':>12} {'+-SE':>10} {'|z|':>6} {'|AD|/SE':>8}"
    )
    for b in args.betas:
        for a in args.alphas:
            cs = p2.case(b, args.n_part, npc_target=args.npc)
            ad, fd = p2.ensemble_alpha(cs, keys, alpha=a, chunk=args.chunk)
            ad, fd = np.asarray(ad), np.asarray(fd)
            k = len(ad)
            adm, fdm = ad.mean(), fd.mean()
            ase = ad.std(ddof=1) / math.sqrt(k)
            fse = fd.std(ddof=1) / math.sqrt(k)
            z = abs(adm - fdm) / math.hypot(ase, fse)
            print(
                f"   {b:>4} {a:>6.2f} {cs.n_part:>8} {adm:>12.5f} {ase:>10.5f} "
                f"{fdm:>12.5f} {fse:>10.5f} {z:>6.2f} {abs(adm) / ase:>8.1f}",
                flush=True,
            )
    print("\n   `|z|` compares AD against FD -- small means they agree.  `|AD|/SE` asks the")
    print("   different question the shock case failed: is the gradient resolvable from")
    print("   ZERO at all?  For scale, dT_y/dT_y^0 is ~0.5-0.96 in the same runs.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m examples.p2_replication",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--part", nargs="+", default=list(PART), choices=PART)
    ap.add_argument("--n-part", type=int, default=8000, help="N for the single-case parts")
    ap.add_argument("--n-list", type=int, nargs="+", default=[2000, 8000, 32000])
    ap.add_argument("--betas", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 1.4])
    ap.add_argument("--keys", type=int, default=p2.M_S, help="M_s")
    ap.add_argument("--npc", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=None)
    args = ap.parse_args(argv)

    print(f"devices: {jax.devices()}")
    for name in PART:
        if name in args.part:
            globals()[f"part_{name}"](args)


if __name__ == "__main__":
    main()
