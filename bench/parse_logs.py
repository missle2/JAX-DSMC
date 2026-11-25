#!/usr/bin/env python3
"""Rebuild the result table from the logs alone (both codes write SPARTA-format logs).

    python3 bench/parse_logs.py <results dir> [...]  -> <first dir>/results.csv

Each log ``log.<date>.<deck>.<machine>_<variant>.<size>.<P>[.rN]`` gives one row: the last run's
``Loop time``, ``Particle moves`` (sparta_jax writes the same line) and, for SPARTA, the event
totals.  Rows from ``rows.jsonl`` (written by run_bench.py, with the node count, compile times
and imbalance) are merged in by log name; a log without a jsonl row still yields a row.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bench import LOOP_RE, MOVES_RE, parse_size, sparta_totals  # noqa: E402

NAME_RE = re.compile(r"^log\.(\w+)\.(free|collide|sphere)\.([\w.-]+?)_([\w.-]+)\.(\d+[KMB]?)\.(\d+)(?:\.r(\d+))?$")
FIELDS = [
    "code", "machine", "variant", "deck", "particles", "nodes", "procs", "steps", "loop_time", "particle_moves",
    "mpsps", "mpsps_per_proc", "rep", "compile_s", "imbalance", "date", "log",
]


def parse_log(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    date, deck, machine, variant, size, procs, rep = m.groups()
    with open(path, errors="replace") as fh:
        text = fh.read()
    loops = LOOP_RE.findall(text)
    if not loops:
        return None
    t, p, steps, _np = loops[-1]
    moves = MOVES_RE.findall(text)
    pm = int(moves[-1]) if moves else None
    code = "jax" if text.startswith("sparta_jax") else "sparta"
    n = parse_size(size)
    row = {
        "code": code, "machine": machine, "variant": variant, "deck": deck, "particles": n, "procs": int(p),
        "steps": int(steps), "loop_time": float(t), "particle_moves": pm, "rep": int(rep or 0), "date": date,
        "log": os.path.basename(path),
    }
    row["mpsps"] = (pm or n * int(steps)) / float(t) / 1e6
    row["mpsps_per_proc"] = row["mpsps"] / int(p)
    if code == "sparta":
        row.update({f"total_{k}": v for k, v in sparta_totals(text).items()})
    return row


def collect(dirs):
    rows = {}
    for d in dirs:
        for root, _sub, files in os.walk(d):
            for f in files:
                if f.startswith("log."):
                    r = parse_log(os.path.join(root, f))
                    if r:
                        rows[r["log"]] = r
        for root, _sub, files in os.walk(d):
            if "rows.jsonl" in files:
                with open(os.path.join(root, "rows.jsonl")) as fh:
                    for ln in fh:
                        if ln.strip():
                            j = json.loads(ln)
                            base = rows.get(j.get("log"), {})
                            base.update({k: v for k, v in j.items() if not isinstance(v, (dict, list))})
                            base["mpsps_per_proc"] = base["mpsps"] / max(1, base["procs"])
                            rows[j.get("log", str(len(rows)))] = base
    return list(rows.values())


def main():
    dirs = sys.argv[1:] or ["bench/results"]
    rows = collect(dirs)
    out = os.path.join(dirs[0], "results.csv")
    keys = FIELDS + sorted({k for r in rows for k in r} - set(FIELDS))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["machine"], r["code"], r["deck"], r["particles"], r["procs"], r.get("rep", 0))):
            w.writerow(r)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
