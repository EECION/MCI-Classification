"""Two-GPU job scheduler (paper environment: 2 x RTX 4090).

Job-level parallelism: tuning is one job per variant (8 jobs), hold-out is one
job per (baseline, enhanced) pair (4 jobs) so paired runs share a process.
Jobs are dispatched round-robin to the given GPUs, one at a time per GPU.

    python scripts/schedule_all.py --stage all
    python scripts/schedule_all.py --stage holdout --pairs transformer
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.registry import PAIRS  # noqa: E402


def build_jobs(stage, variants, pairs):
    jobs = []
    if stage in ("tune", "all"):
        for variant in variants:
            jobs.append(("tune", ["--config", f"configs/{variant}.yaml"]))
    if stage in ("holdout", "all"):
        for base, enh in pairs:
            jobs.append(("holdout", ["--configs", f"configs/{base}.yaml", f"configs/{enh}.yaml"]))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tune", "holdout", "all"], default="all")
    parser.add_argument("--gpus", default="auto", help='e.g. "0,1"; auto = all visible')
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="backbone names to include, e.g. transformer resnet")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if args.gpus == "auto":
        import torch
        n = torch.cuda.device_count()
        gpus = [str(i) for i in range(n)] if n else ["0"]
    else:
        gpus = [g.strip() for g in args.gpus.split(",")]

    pairs = PAIRS
    if args.pairs:
        wanted = set(args.pairs)
        pairs = [(b, e) for b, e in PAIRS if b in wanted or e in wanted]
        if not pairs:
            raise SystemExit(f"no pairs match --pairs {args.pairs}")
    variants = [v for pair in pairs for v in pair]

    jobs = build_jobs(args.stage, variants, pairs)
    print(f"scheduling {len(jobs)} jobs on GPUs {gpus}:")
    for name, extra in jobs:
        print(f"  - {name}: {' '.join(extra)}")

    log_dir = ROOT / "outputs" / "schedule_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running = {}   # gpu -> (process, name, log_path)
    failures = []
    queue = list(jobs)
    python_exe = sys.executable

    while queue or running:
        for gpu in list(running):
            proc, name, log_path = running[gpu]
            if proc.poll() is not None:
                status = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[{status}] {name} ({log_path.name})")
                if proc.returncode != 0:
                    failures.append((name, str(log_path)))
                del running[gpu]
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            name, extra = queue.pop(0)
            script = "run_tuning.py" if name == "tune" else "run_holdout.py"
            cmd = [python_exe, str(ROOT / "scripts" / script), *extra]
            if args.fast:
                cmd.append("--fast")
            tag = "_".join(Path(a).stem for a in extra if str(a).endswith(".yaml"))
            log_path = log_dir / f"{name}_{tag}.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                        env=env, cwd=str(ROOT))
            running[gpu] = (proc, name, log_path)
            print(f"[start gpu{gpu}] {name} {' '.join(extra)} -> {log_path}")
        time.sleep(5)

    print("all jobs finished")
    if failures:
        print(f"{len(failures)} job(s) FAILED:")
        for name, log in failures:
            print(f"  - {name}: {log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
