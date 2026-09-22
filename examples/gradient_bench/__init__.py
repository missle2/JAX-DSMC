"""Focused benchmark of the branch's gradient estimators (see docs/ADJOINT_BRANCH_REVIEW.md).

Every job is one ``examples.gradient_convergence`` invocation; this package only names
the jobs, orders them (pilot -> baseline -> confirmation) and reports on the artifacts.
Numbers in the review document come from ``report.py`` reading those artifacts, never
from a transcript.
"""

import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

COUETTE = {"flow": "couette", "particles": 256, "cells": 4, "steps": 10, "attempts": 256}
INLET = {"flow": "inlet-shock", "particles": 128, "cells": 8, "steps": 4, "attempts": 512}
INLET2 = {"flow": "inlet-shock", "particles": 256, "cells": 8, "steps": 8, "attempts": 512}
SHOCK = {"flow": "shock", "particles": 256, "cells": 16, "steps": 8, "attempts": 512}

PILOT_SEEDS = 32
"""Seeds 0..31 estimate the objective mean that becomes the fixed baseline."""
SEED_START = PILOT_SEEDS
"""Baselined runs use a disjoint seed range, so the baseline is independent of them."""


@dataclass(frozen=True)
class Job:
    name: str
    case: dict
    parameter: str
    score_mode: str = "trajectory"
    majorant: str = "thermal"
    baseline: str | float = 0.0
    """``0.0``, a float, or ``"pilot"`` (resolved from the study's pilot artifact)."""
    seeds: int = 512
    seed_start: int = SEED_START
    time: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def path(self):
        return os.path.join(RESULTS, f"{self.name}.json")

    def argv(self, baseline):
        cfg = {**self.case, **self.extra}
        argv = ["--flow", cfg.pop("flow"), "--parameter", self.parameter]
        for key, value in cfg.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        argv += [
            "--score-mode",
            self.score_mode,
            "--majorant",
            self.majorant,
            "--baseline",
            repr(float(baseline)),
            "--seeds",
            str(self.seeds),
            "--seed-start",
            str(self.seed_start),
            "--output",
            self.path,
        ]
        if self.time:
            argv.append("--time")
        return argv


def pilot(name, case, parameter, **kw):
    """A pilot is the paper estimator with no baseline on seeds ``0..PILOT_SEEDS-1``."""
    return Job(
        name, case, parameter, score_mode="paper", seeds=PILOT_SEEDS, seed_start=0, time=True, **kw
    )


def objective_mean(job):
    """Mean objective over a completed pilot -- the only sanctioned baseline source."""
    with open(job.path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    if len(rows) < job.seeds:
        raise RuntimeError(f"pilot {job.name} incomplete: {len(rows)}/{job.seeds}")
    return sum(r["objective"] for r in rows) / len(rows)


def _arms(prefix, case, parameter, pilot_job, seeds=512, **kw):
    """The estimator arms every study compares, all on the same case and seed range."""
    return [
        Job(f"{prefix}_paper_b0", case, parameter, "paper", baseline=0.0, seeds=seeds, **kw),
        Job(
            f"{prefix}_paper_bpilot", case, parameter, "paper", baseline="pilot", seeds=seeds, **kw
        ),
        Job(f"{prefix}_traj_b0", case, parameter, "trajectory", baseline=0.0, seeds=seeds, **kw),
        Job(
            f"{prefix}_traj_bpilot",
            case,
            parameter,
            "trajectory",
            baseline="pilot",
            seeds=seeds,
            **kw,
        ),
    ]


def studies():
    out = {}
    # B1: the estimator triad on the cheapest wall-driven case
    for parameter in ("speed", "temperature_left"):
        p = pilot(f"triad_{parameter}_pilot", COUETTE, parameter)
        out[f"triad_{parameter}"] = [
            p,
            *_arms(f"triad_{parameter}", COUETTE, parameter, p, time=True),
        ]
    # B2: variance of the trajectory estimator vs horizon and vs population
    jobs = []
    for steps in (5, 10, 20, 40):
        case = {**COUETTE, "steps": steps}
        p = pilot(f"scaling_M{steps}_pilot", case, "speed")
        jobs += [
            p,
            Job(
                f"scaling_M{steps}_traj_bpilot",
                case,
                "speed",
                baseline="pilot",
                seeds=256,
                time=True,
            ),
        ]
    for particles in (128, 512, 1024):
        case = {**COUETTE, "particles": particles}
        p = pilot(f"scaling_N{particles}_pilot", case, "speed")
        jobs += [
            p,
            Job(
                f"scaling_N{particles}_traj_bpilot",
                case,
                "speed",
                baseline="pilot",
                seeds=256,
                time=True,
            ),
        ]
    out["scaling"] = jobs
    # B3: inlet shock, reservoir temperature and density
    for parameter in ("temp", "nrho"):
        p = pilot(f"inlet_{parameter}_pilot", INLET, parameter)
        out[f"inlet_{parameter}"] = [
            p,
            *_arms(f"inlet_{parameter}", INLET, parameter, p, time=True),
        ]
    p = pilot("inlet2_temp_pilot", INLET2, "temp")
    out["inlet2_temp"] = [p, *_arms("inlet2_temp", INLET2, "temp", p, seeds=256, time=True)]
    # B4: reflected shock diam/omega -- the case VARIANCE.md ran with the mean baseline.
    # The b=1 arm lets report.py reconstruct the mean-baseline estimator per seed:
    # g(b) = g(0) - b * sum(w dL)  =>  g_mean = g(0) - J * (g(0) - g(1)).
    for parameter in ("diam", "omega"):
        p = pilot(f"shockbl_{parameter}_pilot", SHOCK, parameter)
        out[f"shockbl_{parameter}"] = [
            p,
            Job(
                f"shockbl_{parameter}_paper_b0",
                SHOCK,
                parameter,
                "paper",
                baseline=0.0,
                seeds=256,
                time=True,
            ),
            Job(
                f"shockbl_{parameter}_paper_b1", SHOCK, parameter, "paper", baseline=1.0, seeds=256
            ),
            Job(
                f"shockbl_{parameter}_paper_bpilot",
                SHOCK,
                parameter,
                "paper",
                baseline="pilot",
                seeds=256,
            ),
            Job(
                f"shockbl_{parameter}_traj_bpilot",
                SHOCK,
                parameter,
                "trajectory",
                baseline="pilot",
                seeds=256,
            ),
        ]
    # B5: majorant choice, cost and score variance
    jobs = []
    for parameter in ("speed", "diameter"):
        p = pilot(f"majorant_{parameter}_pilot", COUETTE, parameter)
        jobs.append(p)
        for majorant in ("thermal", "certified"):
            jobs.append(
                Job(
                    f"majorant_{parameter}_{majorant}",
                    COUETTE,
                    parameter,
                    "trajectory",
                    majorant=majorant,
                    baseline="pilot",
                    seeds=256,
                    time=True,
                )
            )
    out["majorant"] = jobs
    return out


def pilot_of(job, jobs):
    """The pilot that shares a job's case and parameter."""
    for candidate in jobs:
        if (
            candidate.seed_start == 0
            and candidate.case == job.case
            and candidate.parameter == job.parameter
            and candidate.extra == job.extra
        ):
            return candidate
    raise KeyError(f"no pilot for {job.name}")
