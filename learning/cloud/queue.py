"""Runs a file of cloud jobs, keeping the GPU slots busy, until all have finished.

    uv run python cloud/queue.py cloud/queues/scaling.txt --spot 2 --on-demand 1

Each line of the file is one job, as launch.py's arguments: KIND NAME
[CONFIG] [-- TRAINER ARGS] (comments start with #). Every --poll seconds
the queue looks at each job: finished if its job_status on S3 is 0,
failed if it is anything else, running if an instance tagged with it is
up; otherwise it is started, on spot while fewer than --spot spot
instances run, else on demand while fewer than --on-demand do. A job
whose instance vanished without a status, as a spot interruption leaves
it, is simply started again: jobs carry on from their checkpoints on S3.
A failed job is never retried.
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch import run_uri  # noqa: E402


def aws(profile, *args):
    return subprocess.run(["aws", "--profile", profile, *args], capture_output=True, text=True)


def status(job, profile) -> str | None:
    out = aws(profile, "s3", "cp", f"{run_uri(job['kind'], job['name'])}/job_status", "-")
    return out.stdout.strip() if out.returncode == 0 else None


def running(profile) -> dict[str, str]:
    """The instances up now: their Name tag (KIND/NAME) and lifecycle (spot or on-demand)."""
    out = aws(profile, "ec2", "describe-instances", "--filters",
              "Name=instance-state-name,Values=pending,running,stopping,shutting-down",
              "--query", "Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceLifecycle]",
              "--output", "text")
    up = {}
    for line in out.stdout.splitlines():
        tag, lifecycle = line.split("\t")
        up[tag] = "spot" if lifecycle == "spot" else "on-demand"
    return up


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--spot", type=int, default=2)
    parser.add_argument("--on-demand", type=int, default=1)
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument("--profile", default="andrea-personal")
    args = parser.parse_args()

    jobs = []
    for line in args.file.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            words = shlex.split(line)
            jobs.append({"kind": words[0], "name": words[1], "args": words})
    failed = set()
    while True:
        up = running(args.profile)
        pending = []
        for job in jobs:
            tag = f"{job['kind']}/{job['name']}"
            if tag in up or job["name"] in failed:
                continue
            s = status(job, args.profile)
            if s == "0":
                continue
            if s is not None:
                print(f"{time.strftime('%H:%M')} {job['name']} failed with status {s}", flush=True)
                failed.add(job["name"])
                continue
            pending.append(job)
        active = sum(1 for j in jobs if f"{j['kind']}/{j['name']}" in up)
        if not pending and not active:
            print(f"{time.strftime('%H:%M')} all done; failed: {sorted(failed) or 'none'}", flush=True)
            return
        spot = sum(1 for tag, kind in up.items() if kind == "spot")
        on_demand = sum(1 for tag, kind in up.items() if kind == "on-demand")
        for job in pending:
            for market, free in (("spot", spot < args.spot), ("on-demand", on_demand < args.on_demand)):
                if not free:
                    continue
                cmd = ["uv", "run", "-q", "python", str(HERE / "launch.py"), *job["args"]]
                cmd[5:5] = ["--on-demand"] if market == "on-demand" else []
                out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE.parent)
                if out.returncode == 0:
                    print(f"{time.strftime('%H:%M')} started {job['name']} on {market}: {out.stdout.strip()[:90]}", flush=True)
                    spot += market == "spot"
                    on_demand += market == "on-demand"
                    break
                print(f"{time.strftime('%H:%M')} could not start {job['name']} on {market}: "
                      f"{out.stderr.strip().splitlines()[-1][:120] if out.stderr.strip() else '?'}", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
