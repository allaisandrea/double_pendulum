"""Writes a video of a policy acting in the world model, from the pendulum hanging still.

    uv run python -m imagination.video runs/ppo-base/policy.pt
    uv run python -m imagination.video runs/ppo-base/policy.pt --seconds 20 --sample --out spin.mp4

The start is the pendulum hanging still, at the yaws it rests at in the
recordings, and the arms are drawn corrected to hang straight down. The
world model is the one the policy trained in, unless
--world-model names another. The video goes next to the policy unless
--out says otherwise.
"""
import argparse
from pathlib import Path

import torch

from common.run_lib import pick_device
from imagination.rollout_lib import from_hanging, load_policy, write_rollout_video
from world_model.model_lib import load_world_model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("policy", type=Path, help="a policy.pt from imagination.train")
    parser.add_argument("--world-model", type=Path, help="a model.pt to use instead")
    parser.add_argument("--out", type=Path, help="default: rollout.mp4 next to the policy")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample", action="store_true", help="sample actions, not the likeliest")
    parser.add_argument("--slowdown", type=float, default=4, help="times slower than real time")
    parser.add_argument("--seed", type=int, default=0, help="for tag misses and sampled actions")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    agent, env, cfg = load_policy(args.policy, device)
    if args.world_model:
        env.model = load_world_model(args.world_model, device)
        if env.model.window > env.history:
            raise SystemExit(f"{args.world_model} needs a longer history than the policy's")
    out = args.out or args.policy.with_name("rollout.mp4")
    if args.sample:
        torch.manual_seed(args.seed)
    steps = round(args.seconds / 0.008)
    run = from_hanging(agent, env, steps, args.seed, greedy=not args.sample)
    write_rollout_video(run, env, out, args.slowdown)
    print(f"{out}: {args.seconds} s, mean reward {run.reward.mean().item():.3f}")


if __name__ == "__main__":
    main()
