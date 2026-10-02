"""Closed-loop rollouts of a policy in the world model, and videos of them."""
from dataclasses import dataclass
from pathlib import Path

import torch

from common.data_lib import ACTION_SCALE, hanging_yaws, load_recordings
from common.render_lib import render, write_video
from imagination.agent_lib import Agent
from imagination.env_lib import ImaginedEnv
from world_model.model_lib import load_world_model


@dataclass
class Rollout:
    prediction: torch.Tensor  # [T, N, NUM_TAGS, 2]: each predicted frame
    seen: torch.Tensor  # [T, N, NUM_TAGS]: whether each tag was kept as seen
    action: torch.Tensor  # [T, N]: the int8 action taken on the frame before
    reward: torch.Tensor  # [T, N]: net of the speed penalty
    penalty: torch.Tensor  # [T, N]: the speed penalty
    speed: torch.Tensor  # [T, N, NUM_TAGS]: each arm's speed, revolutions per second
    value: torch.Tensor  # [T, N]: the critic's value of the state each action was chosen in


@torch.no_grad()
def closed_loop(agent: Agent, env: ImaginedEnv, steps: int, greedy: bool = True) -> Rollout:
    """Runs the policy for `steps` frames from the environments' current
    state, taking its likeliest action or sampling one."""
    out = {k: [] for k in ("prediction", "seen", "action", "reward", "penalty", "speed", "value")}
    for _ in range(steps):
        x = env.observe()
        dist = agent.policy(x)
        out["value"].append(agent.value(x))
        bins = dist.probs.argmax(-1) if greedy else dist.sample()
        out["reward"].append(env.step(bins))
        out["penalty"].append(env.penalty)
        out["speed"].append(env.speed)
        out["prediction"].append(env.prediction)
        out["seen"].append(env.present[:, -1])
        out["action"].append(env.levels[bins])
    return Rollout(**{k: torch.stack(v) for k, v in out.items()})


def from_hanging(
    agent: Agent, env: ImaginedEnv, steps: int, seed: int = 0, greedy: bool = True
) -> Rollout:
    """One rollout of `steps` frames from the pendulum hanging still. The tag
    misses come from a generator seeded with `seed`, so a greedy policy
    always gives the same rollout."""
    env.reset_hanging(1)
    env.generator = torch.Generator().manual_seed(seed)
    try:
        return closed_loop(agent, env, steps, greedy)
    finally:
        env.generator = None


def write_rollout_video(run: Rollout, env: ImaginedEnv, path: Path, slowdown: float = 4):
    """An mp4 of the first rollout in `run`, from the pendulum hanging still,
    drawn with `env`'s hanging yaws corrected to straight down."""
    # Each frame is shown with the action chosen on it, as the recordings
    # store it, so the last predicted frame, with none, is left out.
    obs = torch.cat([env.hanging[None], run.prediction[:-1, 0]])
    seen = torch.cat([torch.ones_like(run.seen[:1, 0]), run.seen[:-1, 0]])
    action = run.action[:, 0] / ACTION_SCALE
    frames = render(
        obs.cpu().numpy(),
        seen.cpu().numpy(),
        action.cpu().numpy(),
        slowdown=slowdown,
        hanging=env.hanging.cpu().numpy(),
    )
    write_video(path, frames, slowdown=slowdown)


def load_policy(path, device) -> tuple[Agent, ImaginedEnv, dict]:
    """The agent a checkpoint train.py saved, in eval mode, with an
    environment on the world model its config names, and the config.
    Checkpoints from before the yaw correction have no hanging yaws; they
    are measured from the config's recordings."""
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    levels = ck["levels"]
    hanging = ck.get("hanging")
    if hanging is None:
        hanging = hanging_yaws(load_recordings(Path(cfg["data_dir"]), cfg["recordings"], cfg.get("observation", "yaw")))
    model = load_world_model(cfg["world_model"], device)
    env = ImaginedEnv(
        model,
        levels,
        cfg["policy_window"],
        cfg["sample_missing"],
        hanging,
        cfg.get("speed_limits_rev_s"),
        cfg.get("speed_penalty", 0.0),
        cfg.get("tau", 1.0),
    )
    agent = Agent(ck["features"], cfg["hidden"], cfg["layers"], len(levels)).to(device)
    agent.load_state_dict(ck["agent"])
    return agent.eval(), env, cfg
