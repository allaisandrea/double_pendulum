"""Starts a training job on an EC2 instance of its own, from the Mac.

    uv run python cloud/launch.py bench bench-g6 --instance g6.xlarge --follow
    uv run python cloud/launch.py world_model wm4 world_model/configs/base.toml --follow
    uv run python cloud/launch.py policy ppo-wm4 imagination/configs/base.toml -- --set seed=1

Packages the current commit (the working tree must be clean) to
S3_ROOT/code/, and starts a spot instance (or --on-demand) from AWS's Deep
Learning Base GPU AMI, which runs cloud/job.sh on the job and terminates
itself when it ends, or after --max-hours (12) whatever happens. Arguments after -- go to the trainer. The run ends up
in S3_ROOT/runs/KIND/NAME, with job.log and job_status; --follow prints the
log as it arrives. Launching a policy job again with the same name resumes
it from its latest checkpoint there.

Needs the IAM role and secret cloud/setup.sh creates.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S3_ROOT = "s3://allais-andrea-store/double_pendulum"
ROLE = "double-pendulum-trainer"
AMI_PARAMETER = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
LEARNING = Path(__file__).resolve().parents[1]


def aws(*args: str, profile: str) -> str:
    out = subprocess.run(["aws", "--profile", profile, *args], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"aws {args[0]} {args[1]}: {out.stderr.strip()}")
    return out.stdout.strip()


def package(profile: str) -> str:
    """The S3 URI of the current commit's code, uploaded if new."""
    status = subprocess.run(["git", "status", "--porcelain"], cwd=LEARNING, capture_output=True, text=True)
    if status.stdout.strip():
        raise SystemExit("commit first: the job runs the current commit, not the working tree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=LEARNING, capture_output=True, text=True, check=True
    ).stdout.strip()
    uri = f"{S3_ROOT}/code/{commit}.tar.gz"
    if subprocess.run(["aws", "--profile", profile, "s3", "ls", uri], capture_output=True).returncode:
        archive = subprocess.run(
            ["git", "archive", "--format=tar.gz", "HEAD"], cwd=LEARNING.parent, capture_output=True, check=True
        ).stdout
        subprocess.run(["aws", "--profile", profile, "s3", "cp", "-", uri, "--only-show-errors"], input=archive, check=True)
    return uri


def user_data(code: str, job: list[str], max_hours: float) -> str:
    quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in job)
    return f"""#!/bin/bash
# However the job goes, the instance terminates after --max-hours.
shutdown -h +{round(max_hours * 60)}
mkdir -p /opt/job && cd /opt/job
aws s3 cp {code} code.tar.gz --only-show-errors && tar xzf code.tar.gz || shutdown -h now
exec bash learning/cloud/job.sh {S3_ROOT} {quoted}
"""


# Errors that another availability zone may not have.
ZONE_ERRORS = ("InsufficientInstanceCapacity", "Unsupported", "SpotMaxPriceTooLow", "InsufficientCapacity")


def launch_somewhere(run: list[str], types: list[str], profile: str) -> tuple[str, str]:
    """Runs `run` with each instance type in turn, in each public subnet in
    turn, until one starts; returns the instance and its type. The account
    has no default VPC; its VPC's subnets give public IPs, and its default
    security group lets the instance reach the internet."""
    subnets = aws("ec2", "describe-subnets", "--filters", "Name=map-public-ip-on-launch,Values=true",
                  "--query", "Subnets[].[SubnetId,AvailabilityZone]", "--output", "text", profile=profile)
    tried = []
    for instance_type in types:
        for line in subnets.splitlines():
            subnet, zone = line.split()
            out = subprocess.run(
                ["aws", "--profile", profile, *run, "--instance-type", instance_type, "--subnet-id", subnet],
                capture_output=True, text=True,
            )
            if out.returncode == 0:
                return out.stdout.strip(), instance_type
            tried.append(f"{instance_type} in {zone}: {out.stderr.strip()[:160]}")
            if not any(e in out.stderr for e in ZONE_ERRORS):
                raise SystemExit("could not start the instance:\n  " + "\n  ".join(tried))
    raise SystemExit("no capacity for any of the instance types:\n  " + "\n  ".join(tried))


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("kind", choices=["world_model", "policy", "bench", "run", "scale", "sweep"])
    parser.add_argument("name")
    parser.add_argument("config", nargs="?", default="-", help="relative to learning/, or for run a module; not for bench")
    parser.add_argument("--instance", default="g6.xlarge,g5.xlarge",
                        help="instance types to try in turn, comma-separated")
    parser.add_argument("--on-demand", action="store_true")
    parser.add_argument("--follow", action="store_true", help="print the job's log until it ends")
    parser.add_argument("--max-hours", type=float, default=12, help="terminate the instance after this, however the job goes")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "andrea-personal"))
    args = parser.parse_args(argv)
    if args.kind != "bench" and args.config == "-":
        parser.error(f"a {args.kind} job needs a config")

    code = package(args.profile)
    ami = aws("ssm", "get-parameter", "--name", AMI_PARAMETER, "--query", "Parameter.Value", "--output", "text", profile=args.profile)
    run = [
        "ec2", "run-instances",
        "--image-id", ami,
        "--iam-instance-profile", f"Name={ROLE}",
        "--instance-initiated-shutdown-behavior", "terminate",
        # The CLI base64-encodes the user data itself.
        "--user-data", user_data(code, [args.kind, args.name, args.config, *extra], args.max_hours),
        "--tag-specifications", json.dumps(
            [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"{args.kind}/{args.name}"}]}]
        ),
        "--query", "Instances[0].InstanceId", "--output", "text",
    ]
    if not args.on_demand:
        run += ["--instance-market-options", json.dumps(
            {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
        )]
    instance, instance_type = launch_somewhere(run, args.instance.split(","), args.profile)
    # Scale and sweep jobs keep their runs, and their log, with the world models.
    kind_dir = "world_model" if args.kind in ("scale", "sweep") else args.kind
    run_uri = f"{S3_ROOT}/runs/{kind_dir}/{args.name}"
    print(f"{instance}: {args.kind} {args.name} on {'on-demand' if args.on_demand else 'spot'} {instance_type}; "
          f"its run and log go to {run_uri}")
    if args.follow:
        follow(instance, run_uri, args.profile)


def follow(instance: str, run_uri: str, profile: str):
    """Prints the job's log as it reaches S3, until the instance is gone."""
    shown = 0
    while True:
        time.sleep(30)
        log = subprocess.run(["aws", "--profile", profile, "s3", "cp", f"{run_uri}/job.log", "-"],
                             capture_output=True, text=True).stdout
        lines = [l for l in log.splitlines() if "more than one new minimum" not in l]
        for line in lines[shown:]:
            print(line)
        shown = len(lines)
        state = aws("ec2", "describe-instances", "--instance-ids", instance, "--query",
                    "Reservations[0].Instances[0].State.Name", "--output", "text", profile=profile)
        if state in ("terminated", "stopped"):
            status = subprocess.run(["aws", "--profile", profile, "s3", "cp", f"{run_uri}/job_status", "-"],
                                    capture_output=True, text=True).stdout.strip()
            print(f"{instance} {state}; job status {status or 'unknown (interrupted?)'}")
            return


if __name__ == "__main__":
    main()
