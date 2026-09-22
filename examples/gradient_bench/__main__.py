"""Run one benchmark study (or all), pilot first, then the baselined arms.

    python3 -m examples.gradient_bench --study triad_speed
    python3 -m examples.gradient_bench --study all --seeds-scale 0.25   # quick pass

Each job resumes from its own artifact, so an interrupted study restarts where it stopped.
"""

import argparse
import dataclasses
import time

from examples import gradient_convergence
from examples.gradient_bench import objective_mean, pilot_of, studies


def run_job(job, jobs):
    if job.baseline == "pilot":
        baseline = objective_mean(pilot_of(job, jobs))
    else:
        baseline = float(job.baseline)
    t0 = time.time()
    print(f"\n=== {job.name}  (baseline={baseline:.6g}, seeds={job.seeds}) ===", flush=True)
    gradient_convergence.main(job.argv(baseline))
    print(f"=== {job.name} done in {time.time() - t0:.0f}s ===", flush=True)


def main(argv=None):
    all_studies = studies()
    ap = argparse.ArgumentParser(prog="python3 -m examples.gradient_bench", description=__doc__)
    ap.add_argument("--study", default="all", choices=[*all_studies, "all"])
    ap.add_argument(
        "--seeds-scale", type=float, default=1.0, help="multiply every non-pilot seed count"
    )
    ap.add_argument("--only", default=None, help="run a single job by name")
    args = ap.parse_args(argv)
    names = list(all_studies) if args.study == "all" else [args.study]
    for name in names:
        jobs = all_studies[name]
        for job in jobs:
            if args.only and job.name != args.only:
                continue
            scaled = job
            if job.seed_start != 0 and args.seeds_scale != 1.0:
                scaled = dataclasses.replace(job, seeds=max(2, int(job.seeds * args.seeds_scale)))
            run_job(scaled, jobs)


if __name__ == "__main__":
    main()
