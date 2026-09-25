"""Trains a policy with PPO in the world model, logging to Weights & Biases.

    uv run python -m imagination.train imagination/configs/base.toml --name first
    uv run python -m imagination.train imagination/configs/base.toml --name smoke --wandb disabled --iterations 3

Each iteration starts `num_envs` environments from real windows of the
recordings, runs them for `episode_steps` frames with actions sampled from
the policy, and updates the policy on that batch. Rewards are scaled by
1 - gamma for the critic, which puts returns in about [-3, 3].

Every `eval_every` iterations, the policy acts greedily from a fixed set
of starts for `eval_steps` frames, and the run's checkpoint is saved to
`runs/<name>/policy.pt`. At the end, a video of the policy acting from the
pendulum at rest for `video_seconds` goes to `runs/<name>/rollout.mp4` and
to W&B.
"""
import argparse
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import wandb

from common.data_lib import Windows, load_recordings, rest_window
from common.run_lib import git_commit, pick_device
from common.schedule_lib import trapezoid_scheduler
from imagination.agent_lib import Agent
from imagination.env_lib import ImaginedEnv, action_levels
from imagination.ppo_lib import gae, ppo_loss
from imagination.rollout_lib import closed_loop, video_from_rest
from world_model.model_lib import load_world_model

# A tag counts as upright within 30° of vertical.
UPRIGHT_COS = float(np.cos(np.radians(30)))


def upright(obs: torch.Tensor) -> torch.Tensor:
    """Whether every tag is upright, for obs [N, NUM_TAGS, 2]."""
    return (obs[..., 1] > UPRIGHT_COS).all(-1)


def evaluate(agent, env, starts, index, steps) -> dict:
    """Greedy rollouts from the starts at `index`."""
    env.reset(starts, index)
    run = closed_loop(agent, env, steps)
    ups = upright(run.prediction).float()
    return {
        "eval/reward": run.reward.mean().item(),
        "eval/reward_second_half": run.reward[steps // 2 :].mean().item(),
        "eval/upright": ups.mean().item(),
        "eval/upright_second_half": ups[steps // 2 :].mean().item(),
        "eval/mean_abs_action": run.action.abs().float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", required=True, help="the run's name, here and in W&B")
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, help="override the config's iterations")
    args = parser.parse_args()
    out = Path("runs") / args.name
    if out.exists():
        raise SystemExit(f"{out} exists: pick another --name")

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])

    model = load_world_model(cfg["world_model"], device)
    levels = action_levels(cfg["action_step"], cfg["action_bins"])
    env = ImaginedEnv(model, levels, cfg["policy_window"], cfg["sample_missing"])
    recordings = load_recordings(Path(cfg["data_dir"]), cfg["recordings"])
    starts = Windows(recordings, env.history, device)
    rest = rest_window(recordings, env.history, device=device)
    eval_index = starts.sample(cfg["eval_envs"], generator)

    agent = Agent(env.features, cfg["hidden"], cfg["layers"], len(levels)).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)
    sched = trapezoid_scheduler(
        opt, cfg["iterations"], cfg["warmup_iterations"], cfg["cooldown_fraction"]
    )

    run = wandb.init(
        project=cfg["wandb_project"],
        name=args.name,
        config={
            **cfg,
            "commit": git_commit(),
            "device": str(device),
            "levels": levels.tolist(),
            "parameters": sum(p.numel() for p in agent.parameters()),
        },
        mode=args.wandb,
    )
    out.mkdir(parents=True)
    print(f"{args.name}: actions {levels.tolist()}, {len(starts)} start windows on {device}")

    T, N, gamma = cfg["episode_steps"], cfg["num_envs"], cfg["gamma"]
    batch = T * N
    minibatch = batch // cfg["minibatches"]
    x_buf = torch.zeros(T, N, env.features, device=device)
    bin_buf = torch.zeros(T, N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(T, N, device=device)
    value_buf = torch.zeros(T, N, device=device)
    reward_buf = torch.zeros(T, N, device=device)
    up_buf = torch.zeros(T, N, device=device)

    def save():
        torch.save(
            {"config": cfg, "levels": levels, "features": env.features, "agent": agent.state_dict()},
            out / "policy.pt",
        )

    for it in range(1, cfg["iterations"] + 1):
        started = time.perf_counter()
        env.reset(starts, starts.sample(N, generator))
        with torch.no_grad():
            for t in range(T):
                x = env.observe()
                dist = agent.policy(x)
                bins = dist.sample()
                x_buf[t], bin_buf[t], logp_buf[t] = x, bins, dist.log_prob(bins)
                value_buf[t] = agent.value(x)
                reward_buf[t] = env.step(bins)
                up_buf[t] = upright(env.prediction).float()
            last_value = agent.value(env.observe())
            advantages, returns = gae(
                reward_buf * (1 - gamma), value_buf, last_value, gamma, cfg["gae_lambda"]
            )
        rollout_s = time.perf_counter() - started

        flat = [b.flatten(0, 1) for b in (x_buf, bin_buf, logp_buf, advantages, returns)]
        stats = []
        for _ in range(cfg["update_epochs"]):
            order = torch.randperm(batch, device=device)
            for i in range(0, batch, minibatch):
                mb = order[i : i + minibatch]
                loss, s = ppo_loss(agent, *(b[mb] for b in flat), cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                opt.step()
                stats.append(s)
        sched.step()

        values, rets = value_buf.flatten(), returns.flatten()
        metrics = {k: torch.stack([s[k] for s in stats]).mean().item() for k in stats[0]}
        metrics = {f"train/{k}": v for k, v in metrics.items()}
        metrics |= {
            "train/reward": reward_buf.mean().item(),
            "train/reward_last_step": reward_buf[-1].mean().item(),
            "train/upright": up_buf.mean().item(),
            "train/explained_variance": (1 - (rets - values).var() / rets.var()).item(),
            "lr": sched.get_last_lr()[0],
            "rollout_s": rollout_s,
            "iteration_s": time.perf_counter() - started,
        }
        if it % cfg["eval_every"] == 0 or it == cfg["iterations"]:
            metrics |= evaluate(agent, env, starts, eval_index, cfg["eval_steps"])
            save()
            print(
                f"iteration {it}: reward {metrics['train/reward']:.3f},"
                f" greedy {metrics['eval/reward']:.3f}"
                f" (second half {metrics['eval/reward_second_half']:.3f},"
                f" upright {metrics['eval/upright_second_half']:.1%}),"
                f" entropy {metrics['train/entropy']:.3f}, {metrics['iteration_s']:.1f} s",
                flush=True,
            )
        wandb.log(metrics, step=it)

    video = out / "rollout.mp4"
    video_from_rest(agent, env, rest, video, cfg["video_seconds"])
    wandb.log({"rollout_from_rest": wandb.Video(str(video), format="mp4")}, step=cfg["iterations"])
    run.finish()


if __name__ == "__main__":
    main()
