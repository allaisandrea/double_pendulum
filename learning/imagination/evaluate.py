"""Evaluates policies over many rollouts from the pendulum hanging still.

    uv run python -m imagination.evaluate runs/policy/ppo-wm3/policy.pt runs/policy/ppo-wm2/policy.pt
    uv run python -m imagination.evaluate runs/policy/*/policy.pt --world-model runs/world_model/wm3/model.pt

Training evaluates one rollout, which a chaotic system makes noisy. This
runs --rollouts of them, greedy, --seconds long, from the pendulum hanging
still, each with its own tag misses drawn from one generator seeded with
--seed, and reports the mean sum of cosines per frame with its standard
error across rollouts, the share of frames with every arm within 30° of
upright, the speed penalty, and the first arm's speeds. Every policy runs
in the same world model: --world-model, or else the first policy's.
"""
import argparse
from pathlib import Path

import torch

from common.data_lib import correct_yaws
from common.run_lib import pick_device
from imagination.env_lib import upright
from imagination.rollout_lib import closed_loop, load_policy
from world_model.model_lib import load_world_model


@torch.no_grad()
def evaluate_policy(agent, env, rollouts: int, steps: int, seed: int) -> dict:
    env.reset_hanging(rollouts)
    env.generator = torch.Generator().manual_seed(seed)
    try:
        run = closed_loop(agent, env, steps)
    finally:
        env.generator = None
    corrected = correct_yaws(run.prediction, env.hanging)
    per_rollout = corrected[..., 1].sum(-1).mean(0)  # [rollouts]
    arm0 = run.speed[1:, :, 0].flatten()
    return {
        "cos_sum": per_rollout.mean().item(),
        "cos_sum_sem": (per_rollout.std() / rollouts**0.5).item(),
        "upright": upright(corrected).float().mean().item(),
        "penalty": run.penalty.mean().item(),
        "arm0_median": arm0.median().item(),
        "arm0_p99": arm0.quantile(0.99).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("policies", type=Path, nargs="+", help="policy.pt checkpoints")
    parser.add_argument("--world-model", type=Path, help="default: the first policy's")
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)

    model = load_world_model(args.world_model, device) if args.world_model else None
    steps = round(args.seconds / 0.008)
    print(f"{args.rollouts} rollouts of {args.seconds:g} s from hanging, greedy, seed {args.seed}")
    print(f"{'policy':28s} {'cos sum':>16s} {'upright':>8s} {'penalty':>8s} {'arm 0 rev/s (median, p99)':>28s}")
    for path in args.policies:
        agent, env, cfg = load_policy(path, device)
        if model is None:
            model = env.model
        if model.window > env.history:
            print(f"{path}: the world model needs more history than this policy keeps")
            continue
        env.model = model
        m = evaluate_policy(agent, env, args.rollouts, steps, args.seed)
        print(
            f"{path.parent.name:28s} {m['cos_sum']:+8.3f} ± {m['cos_sum_sem']:.3f} {m['upright']:8.1%}"
            f" {m['penalty']:8.3f} {m['arm0_median']:14.2f} {m['arm0_p99']:13.2f}"
        )


if __name__ == "__main__":
    main()
