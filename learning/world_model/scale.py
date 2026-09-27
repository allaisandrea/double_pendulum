"""Trains one world model at several lengths: a long run, and cooldowns branched from it.

    uv run python -m world_model.scale world_model/configs/base.toml --name wm-w512-d3 \\
        --steps 200000 --branches 25000 50000 100000 200000 -- --set hidden=512 --set eval_every=5000

The long run, NAME, holds the learning rate constant after its warmup, for
--steps, checkpointing at each of --branches. From each branch point b, a
branch NAME-cd<b/1000>k (NAME-cd<b> for b not in thousands) cools the
rate down over another --cooldown-fraction * b steps: a model trained on a
trapezoid for b steps plus its cooldown, at the cost of the cooldown alone. Arguments after -- go to world_model.train for
the long run; branches take its config.

Run again, it carries on from whatever an interruption left: the long run
from its latest checkpoint, each unfinished branch from its own, and it
skips branches already finished.
"""
import argparse
import shutil
import sys
from pathlib import Path

from world_model.train import main as train

RUNS = Path("runs/world_model")


def latest(run: str) -> Path | None:
    """The run's latest checkpoint. A run directory with none holds nothing
    to carry on from, so it is removed, for the run to start over."""
    checkpoints = sorted((RUNS / run / "checkpoints").glob("step_*.pt"))
    if not checkpoints and (RUNS / run).exists():
        shutil.rmtree(RUNS / run)
    return checkpoints[-1] if checkpoints else None


def step_of(checkpoint: Path) -> int:
    return int(checkpoint.stem.split("_")[1])


def branch_name(name: str, step: int) -> str:
    return f"{name}-cd{step // 1000}k" if step % 1000 == 0 else f"{name}-cd{step}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--branches", type=int, nargs="+", required=True)
    parser.add_argument("--cooldown-fraction", type=float, default=0.2)
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if max(args.branches) > args.steps:
        parser.error("a branch point lies beyond --steps")
    common = ["--wandb", args.wandb, "--device", args.device]

    last = latest(args.name)
    if last is None:
        branches = "[" + ", ".join(map(str, sorted(args.branches))) + "]"
        train([str(args.config), "--name", args.name, "--steps", str(args.steps),
               "--set", "cooldown_fraction=0", "--set", f"checkpoint_at={branches}", *extra, *common])
    elif step_of(last) < args.steps:
        print(f"resuming {args.name} from {last}", flush=True)
        train(["--resume", str(last), *common])

    for b in sorted(args.branches):
        name = branch_name(args.name, b)
        cooldown = round(args.cooldown_fraction * b)
        last = latest(name)
        if last is not None and step_of(last) >= b + cooldown:
            print(f"{name} is finished", flush=True)
        elif last is not None:
            print(f"resuming {name} from {last}", flush=True)
            train(["--resume", str(last), *common])
        else:
            parent = RUNS / args.name / "checkpoints" / f"step_{b:07d}.pt"
            train(["--branch", str(parent), "--cooldown-steps", str(cooldown), "--name", name, *common])


if __name__ == "__main__":
    main()
