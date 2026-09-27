"""Runs a list of world model trainings one after another, carrying on after interruptions.

    uv run python -m world_model.sweep world_model/sweeps/batch.txt --wandb online

Each line of the file holds the arguments of one world_model.train run
(comments start with #). Run again, the sweep skips finished runs,
resumes unfinished ones from their latest checkpoint, and starts the rest.
In the cloud, a sweep job's runs must all be named with the job's name as
prefix, so that it syncs them.
"""
import argparse
import shlex
from pathlib import Path

import torch

from world_model.scale import latest, step_of
from world_model.train import main as train


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    common = ["--wandb", args.wandb, "--device", args.device]
    for line in args.file.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        run = shlex.split(line)
        name = run[run.index("--name") + 1]
        last = latest(name)
        if last is None:
            train([*run, *common])
        elif step_of(last) < torch.load(last, map_location="cpu", weights_only=False)["config"]["steps"]:
            print(f"resuming {name} from {last}", flush=True)
            train(["--resume", str(last), *common])
        else:
            print(f"{name} is finished", flush=True)


if __name__ == "__main__":
    main()
