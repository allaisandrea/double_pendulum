"""Closed-loop rollouts of a policy in the world model, and videos of them."""
from dataclasses import dataclass
from pathlib import Path

import torch

from common.data_lib import ACTION_SCALE, Batch
from common.render_lib import render, write_video
from imagination.agent_lib import Agent
from imagination.env_lib import ImaginedEnv
from world_model.model_lib import load_world_model


@dataclass
class Rollout:
    prediction: torch.Tensor  # [T, N, NUM_TAGS, 2]: each predicted frame
    seen: torch.Tensor  # [T, N, NUM_TAGS]: whether each tag was kept as seen
    action: torch.Tensor  # [T, N]: the int8 action taken on the frame before
    reward: torch.Tensor  # [T, N]


@torch.no_grad()
def closed_loop(agent: Agent, env: ImaginedEnv, steps: int, greedy: bool = True) -> Rollout:
    """Runs the policy for `steps` frames from the environments' current
    state: greedily, or sampling its actions."""
    out = {k: [] for k in ("prediction", "seen", "action", "reward")}
    for _ in range(steps):
        dist = agent.policy(env.observe())
        bins = dist.probs.argmax(-1) if greedy else dist.sample()
        out["reward"].append(env.step(bins))
        out["prediction"].append(env.prediction)
        out["seen"].append(env.present[:, -1])
        out["action"].append(env.levels[bins])
    return Rollout(**{k: torch.stack(v) for k, v in out.items()})


def load_policy(path, device) -> tuple[Agent, ImaginedEnv, dict]:
    """The agent a checkpoint train.py saved, in eval mode, with an
    environment on the world model its config names, and the config."""
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    levels = ck["levels"]
    model = load_world_model(cfg["world_model"], device)
    env = ImaginedEnv(model, levels, cfg["policy_window"], cfg["sample_missing"])
    agent = Agent(ck["features"], cfg["hidden"], cfg["layers"], len(levels)).to(device)
    agent.load_state_dict(ck["agent"])
    return agent.eval(), env, cfg


def video_from_rest(
    agent: Agent,
    env: ImaginedEnv,
    rest: Batch,
    path: Path,
    seconds: float,
    greedy: bool = True,
    slowdown: float = 4,
) -> Rollout:
    """Writes an mp4 of the policy acting from the pendulum at rest: the
    last recorded rest frame, then `seconds` of rollout, played `slowdown`
    times slower than real time."""
    env.reset_to(Batch(*(t[:, -env.history :] for t in (rest.obs, rest.present, rest.action))))
    run = closed_loop(agent, env, round(seconds / 0.008), greedy)
    # Each frame is shown with the action chosen on it, as the recordings
    # store it, so the last predicted frame, with none, is left out.
    obs = torch.cat([rest.obs[0, -1:], run.prediction[:-1, 0]])
    seen = torch.cat([rest.present[0, -1:], run.seen[:-1, 0]])
    action = run.action[:, 0] / ACTION_SCALE
    frames = render(obs.cpu(), seen.cpu(), action.cpu(), slowdown=slowdown)
    write_video(path, frames, slowdown=slowdown)
    return run
