"""Summarise a job's GPU monitor log (``job.sbatch``'s ``logs/<job>_<id>_gpu.log``): per GPU
the mean / max utilisation and memory, and the distinct compute processes seen on it -- a
working ``optimize --workers N`` job shows N GPUs busy, each with its own python process.

    python3 -m examples.mesh.cluster.gpu_usage examples/mesh/cluster/logs/<job>_<id>_gpu.log
"""

import argparse
import collections
import re
from pathlib import Path

import numpy as np


def number(text):
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    return float(m.group()) if m else np.nan


def summarise(path):
    util, mem, total, pids = (collections.defaultdict(list) for _ in range(4))
    index = {}
    samples = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            samples += 1
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == "gpu" and len(parts) >= 6:
            i, uuid = parts[1], parts[2]
            index[uuid] = i
            util[i].append(number(parts[3]))
            mem[i].append(number(parts[4]))
            total[i].append(number(parts[5]))
        elif parts[0] == "app" and len(parts) >= 4:
            pids[parts[2]].append(parts[1])
    print(f"{path}: {samples} samples")
    for uuid, i in sorted(index.items(), key=lambda t: t[1]):
        u, m = np.array(util[i]), np.array(mem[i])
        busy = np.mean(u > 10) if u.size else np.nan
        procs = sorted(set(pids.get(uuid, [])))
        print(f"  GPU {i}: util mean {np.nanmean(u):5.1f} % max {np.nanmax(u):5.1f} % "
              f"(>10 % in {100 * busy:4.0f} % of samples), memory max {np.nanmax(m):8.0f} / "
              f"{np.nanmax(total[i]):.0f} MiB, {len(procs)} distinct processes")  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    for path in ap.parse_args(argv).logs:
        summarise(path)


if __name__ == "__main__":
    main()
