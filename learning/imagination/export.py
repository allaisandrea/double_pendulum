"""Exports a policy checkpoint for the harness, as JSON with test cases.

    uv run python -m imagination.export runs/policy/ppo-persistent/policy.pt
    uv run python -m imagination.export --fixture ../harness/src/testdata/mlp_policy.json

The export goes next to the checkpoint as policy.json unless --out says
otherwise. --fixture instead writes a small random policy, for the
harness's own tests.
"""
import argparse
from pathlib import Path

import torch

from imagination.agent_lib import Agent
from imagination.env_lib import action_levels
from imagination.export_lib import export_policy
from world_model.model_lib import STEP_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, nargs="?", help="a policy.pt or iter_NNNNNN.pt")
    parser.add_argument("--out", type=Path, help="default: policy.json next to the checkpoint")
    parser.add_argument("--fixture", type=Path, help="write a small random policy here instead")
    parser.add_argument("--cases", type=int, default=16, help="test cases to include")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.fixture):
        parser.error("give a checkpoint or --fixture")

    if args.fixture:
        torch.manual_seed(0)
        window, levels = 4, action_levels(16, 9)
        agent = Agent(window * STEP_FEATURES, 8, 2, len(levels))
        # Larger final weights than a fresh policy's, so the cases' logits,
        # and the actions they pick, differ.
        agent.actor[-1].weight.data.normal_(0, 1)
        export_policy(agent, levels, window, args.fixture, {"fixture": "random, seed 0"}, args.cases)
        print(f"{args.fixture}: random policy, window {window}")
        return

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("observation", "yaw") != "yaw":
        raise SystemExit(f"{args.checkpoint} sees calibrated angles, which the harness does not compute yet")
    agent = Agent(ck["features"], cfg["hidden"], cfg["layers"], len(ck["levels"]))
    agent.load_state_dict(ck["agent"])
    out = args.out or args.checkpoint.with_name("policy.json")
    source = {"checkpoint": str(args.checkpoint), "iteration": ck.get("iteration")}
    export_policy(agent, ck["levels"], cfg["policy_window"], out, source, args.cases)
    print(f"{out}: window {cfg['policy_window']}, actions {ck['levels'].tolist()}, {args.cases} cases")


if __name__ == "__main__":
    main()
