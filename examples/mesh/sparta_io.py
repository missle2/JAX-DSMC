"""Readers for the SPARTA reference runs in ``examples/mesh/results/sparta/``.

    python3 examples/mesh/sparta_io.py            # summary table over every log found
    python3 examples/mesh/sparta_io.py --dir DIR  # another results directory

Pure numpy, no JAX.  Parses three kinds of SPARTA output:

* the ``-log`` file: the ``stats`` table (one row every ``stats N`` steps), the
  ``read_surf`` cell/volume summary, and the final particle count;
* ``dump grid`` files (``ITEM: CELLS ...``);
* ``dump surf`` files (``ITEM: SURFS ...``).

The dumps contain one snapshot per multiple of the dump interval, which includes a step-0
snapshot in which the time-averaged fix columns are all zero; ``read_grid_dump`` and
``read_surf_dump`` return the *last* snapshot unless ``snapshot`` is given.

The inputs that produced these files live in ``examples/mesh/sparta/``; the summary table
printed by ``main`` is the provenance of the reference numbers quoted in the mesh study.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sparta")

# Column names used by ``stats_style``; SPARTA prints abbreviated headers.
_STATS_HEADER_ALIASES = {
    "Step": "step",
    "CPU": "cpu",
    "Np": "np",
    "Natt": "nattempt",
    "Ncoll": "ncoll",
    "Nscoll": "nscoll",
    "Nscheck": "nscheck",
    "Nexit": "nexit",
    "Nbound": "nboundary",
    "Ntouch": "ntouch",
    "Ncomm": "ncomm",
    "Nreact": "nreact",
    "Nsreact": "nsreact",
    "Time": "time",
    "Wall": "wall",
}

_RE_CELLS = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+=\s+cells outside/inside/overlapping surfs")
_RE_VOLUME = re.compile(r"^\s*(\S+)\s+(\S+)\s+=\s+cell-wise and global flow volume")
_RE_LOOP = re.compile(r"^Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) particles")
_RE_ELEMENTS = re.compile(r"^\s*(\d+)\s+(lines|triangles)\s*$")
_RE_TOTAL = re.compile(
    r"^(SurfColl occurs|Boundary exits|Boundary collides|Particle moves)\s*=\s*(\d+)"
)


def _stats_key(header: str) -> str:
    return _STATS_HEADER_ALIASES.get(header, header.lower())


def read_log_stats(path: str) -> dict[str, np.ndarray]:
    """Parse the ``stats`` table of a SPARTA log into column arrays.

    Keys follow the ``stats_style`` keywords (``step``, ``cpu``, ``np``, ``nattempt``,
    ``ncoll``, ``nscoll``, ``nscheck``, ``nexit``, ``nboundary``, ...).  Integer columns are
    returned as ``int64``, ``cpu``/``time``/``wall`` as ``float64``.  If the log holds several
    ``run`` blocks their rows are concatenated in order.
    """
    columns: list[str] = []
    rows: list[list[float]] = []
    in_table = False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("Step "):
                header = [_stats_key(tok) for tok in line.split()]
                if columns and header != columns:
                    raise ValueError(f"{path}: stats_style changed mid-log: {columns} -> {header}")
                columns = header
                in_table = True
                continue
            if in_table:
                if line.startswith("Loop time"):
                    in_table = False
                    continue
                toks = line.split()
                if len(toks) != len(columns):
                    raise ValueError(f"{path}: malformed stats row {line!r}")
                rows.append([float(t) for t in toks])
    if not columns:
        raise ValueError(f"{path}: no stats table found")
    data = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(columns))
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(columns):
        col = data[:, j]
        out[name] = col if name in ("cpu", "time", "wall") else col.astype(np.int64)
    return out


def read_surf_summary(path: str) -> dict:
    """Parse the ``read_surf`` summary and the end-of-run totals from a SPARTA log.

    Returns ``n_elements`` (lines in 2-D, triangles in 3-D), ``cells_outside``,
    ``cells_inside``, ``cells_overlapping``, ``flow_volume_cellwise``, ``flow_volume_global``,
    ``np_final``, ``n_steps``, ``loop_time``, ``n_procs`` and the run totals
    ``surfcoll_total``, ``exit_total``, ``boundary_total``, ``moves_total``.
    """
    out: dict = {}
    with open(path) as fh:
        for raw in fh:
            m = _RE_ELEMENTS.match(raw)
            if m:
                out["n_elements"] = int(m.group(1))
                out["element_kind"] = m.group(2)
                continue
            m = _RE_CELLS.match(raw)
            if m:
                out["cells_outside"] = int(m.group(1))
                out["cells_inside"] = int(m.group(2))
                out["cells_overlapping"] = int(m.group(3))
                continue
            m = _RE_VOLUME.match(raw)
            if m:
                out["flow_volume_cellwise"] = float(m.group(1))
                out["flow_volume_global"] = float(m.group(2))
                continue
            m = _RE_LOOP.match(raw)
            if m:
                out["loop_time"] = float(m.group(1))
                out["n_procs"] = int(m.group(2))
                out["n_steps"] = int(m.group(3))
                out["np_final"] = int(m.group(4))
                continue
            m = _RE_TOTAL.match(raw)
            if m:
                key = {
                    "SurfColl occurs": "surfcoll_total",
                    "Boundary exits": "exit_total",
                    "Boundary collides": "boundary_total",
                    "Particle moves": "moves_total",
                }[m.group(1)]
                out[key] = int(m.group(2))
    required = ("cells_outside", "flow_volume_global", "np_final")
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"{path}: missing {missing} (truncated log?)")
    return out


def _parse_snapshot(path: str, item: str, block: str) -> tuple[int, list[str], np.ndarray]:
    """Parse one ``ITEM: TIMESTEP`` block of a dump into (step, column names, data)."""
    lines = block.strip().splitlines()
    step = int(lines[0].strip())
    count = None
    columns: list[str] = []
    rows: list[list[float]] = []
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: NUMBER OF"):
            count = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 4
        elif line.startswith(f"ITEM: {item}"):
            columns = line.split()[2:]
            rows = [[float(t) for t in ln.split()] for ln in lines[i + 1 :] if ln.strip()]
            break
        else:
            raise ValueError(f"{path}: unexpected header {line!r} (wanted ITEM: {item})")
    if not columns:
        raise ValueError(f"{path}: snapshot at step {step} has no ITEM: {item} section")
    if count is not None and len(rows) != count:
        raise ValueError(f"{path}: step {step} has {len(rows)} rows, header says {count}")
    data = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(columns))
    return step, columns, data


def _read_dump(path: str, item: str, snapshot: int | None) -> dict[str, np.ndarray]:
    """Parse one snapshot of a SPARTA ``ITEM:``-style dump (``item`` = CELLS or SURFS)."""
    with open(path) as fh:
        blocks = fh.read().split("ITEM: TIMESTEP")[1:]
    if not blocks:
        raise ValueError(f"{path}: no ITEM: TIMESTEP found")
    idx = -1 if snapshot is None else snapshot
    step, columns, data = _parse_snapshot(path, item, blocks[idx])
    out: dict[str, np.ndarray] = {"step": np.int64(step)}
    for j, name in enumerate(columns):
        out[name] = data[:, j].astype(np.int64) if name == "id" else data[:, j]
    return out


def read_grid_dump(path: str, snapshot: int | None = None) -> dict[str, np.ndarray]:
    """Read a ``dump grid`` file; keys are the dump's column names plus ``step``.

    Split cells are already resolved into their sub-cells by SPARTA, so the ``id`` column can
    repeat for cut cells; ``vol`` is the flow volume (area in 2-D) of each row.
    """
    return _read_dump(path, "CELLS", snapshot)


def read_surf_dump(path: str, snapshot: int | None = None) -> dict[str, np.ndarray]:
    """Read a ``dump surf`` file; keys are the dump's column names plus ``step``.

    For the reference inputs the fix columns are ``f_as[1..12]`` in the order of the
    ``compute surf`` values: n press px py pz shx shy shz fx fy fz ke.
    """
    return _read_dump(path, "SURFS", snapshot)


# Order of values in `compute s surf all all ...` of in.circle_ar / in.sphere_ar.
SURF_VALUES = ("n", "press", "px", "py", "pz", "shx", "shy", "shz", "fx", "fy", "fz", "ke")


def surf_columns(dump: dict[str, np.ndarray], prefix: str = "f_as") -> dict[str, np.ndarray]:
    """Relabel ``f_as[k]`` columns of a surface dump by their ``compute surf`` value names."""
    out = {k: v for k, v in dump.items() if not k.startswith(prefix)}
    for k, name in enumerate(SURF_VALUES, start=1):
        key = f"{prefix}[{k}]"
        if key in dump:
            out[name] = dump[key]
    return out


def _case_seed(log_path: str) -> tuple[str, str]:
    base = os.path.basename(log_path)
    _, case, seed = base.split(".", 2)
    return case, seed


def summarize(results_dir: str, tail_steps: int = 300) -> list[dict]:
    """One row per log: final Np, mean Nscoll over the last ``tail_steps``, drag, volume."""
    rows = []
    for log in sorted(glob.glob(os.path.join(results_dir, "log.*"))):
        case, seed = _case_seed(log)
        stats = read_log_stats(log)
        summary = read_surf_summary(log)
        step = stats["step"]
        tail = step > step[-1] - tail_steps
        row = {
            "case": case,
            "seed": seed,
            "np_final": summary["np_final"],
            "nscoll_mean": float(stats["nscoll"][tail].mean()),
            "nscheck_mean": float(stats["nscheck"][tail].mean()),
            "flow_volume": summary["flow_volume_global"],
            "cells": (
                summary["cells_outside"],
                summary["cells_inside"],
                summary["cells_overlapping"],
            ),
            "n_elements": summary.get("n_elements"),
        }
        surf_path = os.path.join(results_dir, f"surf_ave.{case}.{seed}.txt")
        if os.path.exists(surf_path):
            surf = surf_columns(read_surf_dump(surf_path))
            row["fx_total"] = float(surf["fx"].sum())
            row["fy_total"] = float(surf["fy"].sum())
            row["fz_total"] = float(surf["fz"].sum())
            row["hits_total"] = float(surf["n"].sum())
            row["surf_step"] = int(surf["step"])
        vol_path = os.path.join(results_dir, f"grid_vol.{case}.{seed}.txt")
        if os.path.exists(vol_path):
            grid = read_grid_dump(vol_path, snapshot=0)
            row["vol_dump_sum"] = float(grid["vol"].sum())
            row["n_dump_cells"] = int(grid["vol"].shape[0])
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'case':<11} {'seed':<6} {'Np_final':>9} {'<Nscoll>_300':>12} {'<Nscheck>_300':>13} "
        f"{'sum fx':>12} {'sum fy':>12} {'sum n':>9} {'vol(log)':>9} {'vol(dump)':>10} "
        f"{'cells out/in/ovl':>18}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        fx = f"{r['fx_total']:12.5g}" if "fx_total" in r else f"{'-':>12}"
        fy = f"{r['fy_total']:12.5g}" if "fy_total" in r else f"{'-':>12}"
        n = f"{r['hits_total']:9.0f}" if "hits_total" in r else f"{'-':>9}"
        vd = f"{r['vol_dump_sum']:10.5g}" if "vol_dump_sum" in r else f"{'-':>10}"
        cells = "/".join(str(c) for c in r["cells"])
        lines.append(
            f"{r['case']:<11} {r['seed']:<6} {r['np_final']:>9d} {r['nscoll_mean']:>12.2f} "
            f"{r['nscheck_mean']:>13.1f} {fx} {fy} {n} {r['flow_volume']:>9.5g} {vd} "
            f"{cells:>18}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default=RESULTS, help="directory with log.* and dump files")
    parser.add_argument("--tail", type=int, default=300, help="steps averaged for <Nscoll>")
    args = parser.parse_args()
    rows = summarize(args.dir, tail_steps=args.tail)
    if not rows:
        raise SystemExit(f"no log.* files under {args.dir}")
    print(f"SPARTA reference runs in {args.dir}")
    print(
        "Nscoll/Nscheck: per-step surface collisions/checks from the stats table, averaged over "
        f"the last {args.tail} steps.  sum fx/fy: drag/lift summed over surface elements from the "
        "fix ave/surf dump (mean of the same window).  sum n: hits per step, same window.\n"
    )
    print(format_table(rows))


if __name__ == "__main__":
    main()
