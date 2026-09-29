"""Trains a policy with PPO in the world model, logging to Weights & Biases.

    uv run python -m imagination.train imagination/configs/base.toml --name first
    uv run python -m imagination.train imagination/configs/base.toml --name smoke --wandb disabled --iterations 15
    uv run python -m imagination.train --resume runs/policy/first/checkpoints/iter_000100.pt
    uv run python -m imagination.train imagination/configs/base.toml --name greedier --set ent_coef=0.003

`num_envs` environments start from real windows of the recordings and
carry on from one iteration to the next. Each iteration runs them for
`rollout_steps` frames with actions sampled from the policy, and updates
the policy on that batch. After each frame, each environment starts again
from a new window with probability (1 - gamma) / `reset_horizons`, so
episodes last `reset_horizons` / (1 - gamma) frames on average: with
`reset_horizons` 1, the states are weighted as the discounted objective
weights them. A reset is a truncation, so the return is bootstrapped from
the state the episode would have gone on to. Rewards are scaled by
1 - gamma for the critic, which puts returns in about [-3, 3].

Every `eval_every` iterations the policy is evaluated: one rollout of
`eval_steps` frames from the pendulum hanging still, acting greedily. Its
tag misses are drawn from a generator seeded with `eval_seed`, so
evaluations are repeatable. With `eval_rollouts` N above 1, N more
rollouts from hanging, from the same seed, give `eval/mean/cos_sum` with
its standard error and `eval/mean/upright`: in a stochastic world model,
one rollout is one draw. At the end, a video of the last evaluation's
rollout goes to `runs/policy/<name>/rollout.mp4` and to W&B.

--init starts a new run from another run's policy (a policy.pt or
iter_NNNNNN.pt checkpoint), its actor and critic, with the new run's
config, which must give it the same inputs and actions; the optimiser,
schedule and environments start afresh. With `tau_start` and `tau_end`,
a stochastic world model's tau moves linearly from one to the other over
the run's iterations; evaluations use `tau_end`.

Every `checkpoint_every` iterations, and at the end, the whole training
state goes to `runs/policy/<name>/checkpoints/iter_NNNNNN.pt`, and a copy
to `runs/policy/<name>/policy.pt`. --resume continues from one, with the
config it saved, in the same directory and W&B run; iterations W&B already
has past the checkpoint are not logged again.

Each phase of an iteration is timed, with the device synchronised so that
the times are real: `time/<phase>_s` per iteration, and `time/share/<phase>`
of all the time timed so far.
"""
import argparse
import shutil
from pathlib import Path

import torch
import wandb

from common.data_lib import FRAME_S, Windows, correct_yaws, hanging_yaws, load_recordings
from common.run_lib import Timer, git_commit, load_config, pick_device, rng_state, set_rng_state
from common.schedule_lib import trapezoid_scheduler
from imagination.agent_lib import Agent
from imagination.env_lib import ImaginedEnv, Starts, action_levels, upright
from imagination.evaluate import evaluate_policy
from imagination.ppo_lib import discounted_returns, gae, ppo_loss
from imagination.rollout_lib import Rollout, from_hanging, write_rollout_video
from world_model.model_lib import load_world_model


def evaluate(agent, env, cfg) -> tuple[dict, Rollout]:
    """The evaluation's metrics, and its rollout.

    `eval/reward` is the reward the policy maximises: `eval/cos_sum`, the
    sum of the arms' cosines, less `eval/speed_penalty`. Without speed
    limits the two are the same. The critic is checked against the
    rollout's actual discounted return, in the critic's units (rewards scaled by 1 - gamma), over the frames early
    enough for the rest of the rollout to measure it. The rollout is greedy
    while the critic values the sampling policy, so some bias is expected.
    """
    gamma = cfg["gamma"]
    metrics = {}
    if cfg.get("eval_rollouts", 1) > 1:
        m = evaluate_policy(agent, env, cfg["eval_rollouts"], cfg["eval_steps"], cfg["eval_seed"])
        metrics = {"eval/mean/cos_sum": m["cos_sum"], "eval/mean/cos_sum_sem": m["cos_sum_sem"],
                   "eval/mean/upright": m["upright"]}
    run = from_hanging(agent, env, cfg["eval_steps"], cfg["eval_seed"])
    actual = discounted_returns(run.reward[:, 0] * (1 - gamma), gamma)
    value = run.value[: len(actual), 0]
    metrics |= {
        "eval/reward": run.reward.mean().item(),
        "eval/cos_sum": (run.reward + run.penalty).mean().item(),
        "eval/speed_penalty": run.penalty.mean().item(),
        # The first frame's speed is measured from the start, not a step.
        **{f"eval/speed/arm{i}": v.item() for i, v in enumerate(run.speed[1:, 0].median(0).values)},
        "eval/upright": upright(correct_yaws(run.prediction, env.hanging)).float().mean().item(),
        "eval/mean_abs_action": run.action.abs().float().mean().item(),
        "eval/value_hanging": run.value[0, 0].item(),
    }
    if len(actual):
        seconds = (torch.arange(len(actual)) * FRAME_S).tolist()
        metrics |= {
            "eval/value_error": (value - actual).abs().mean().item(),
            "eval/value_bias": (value - actual).mean().item(),
            "eval/value_vs_return": wandb.plot.line_series(
                xs=seconds,
                ys=[value.tolist(), actual.tolist()],
                keys=["critic's value", "actual discounted return"],
                title="Critic against the evaluation rollout",
                xname="seconds from hanging",
            ),
        }
    return metrics, run


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, nargs="?", help="for a new run")
    parser.add_argument("--name", help="the new run's name, here and in W&B")
    parser.add_argument("--resume", type=Path, help="a checkpoint to continue from")
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, help="override a new run's iterations")
    parser.add_argument("--init", type=Path, help="a new run's starting policy: another run's checkpoint")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a new run's config value, VALUE as TOML (e.g. gamma=0.998); repeatable",
    )
    args = parser.parse_args(argv)
    if args.resume:
        if args.config or args.name or args.iterations or args.set or args.init:
            parser.error("--resume takes the config, name and iterations from the checkpoint")
    elif not (args.config and args.name):
        parser.error("a new run needs a config and --name")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg, out = ck["config"], args.resume.parent.parent
        name = out.name
    else:
        ck, name = None, args.name
        out = Path("runs/policy") / name
        if out.exists():
            raise SystemExit(f"{out} exists: pick another --name")
        # tau: the spread of a stochastic world model's frames (default 1).
        cfg = load_config(args.config, args.set, {"tau", "eval_rollouts", "tau_start", "tau_end"})
        if args.iterations is not None:
            cfg["iterations"] = args.iterations
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])

    model = load_world_model(cfg["world_model"], device)
    levels = action_levels(cfg["action_step"], cfg["action_bins"])
    recordings = load_recordings(Path(cfg["data_dir"]), cfg["recordings"])
    hanging = hanging_yaws(recordings)
    env_args = (
        model,
        levels,
        cfg["policy_window"],
        cfg["sample_missing"],
        hanging,
        cfg.get("speed_limits_rev_s"),
        cfg.get("speed_penalty", 0.0),
        cfg.get("tau", 1.0),
    )
    env = ImaginedEnv(*env_args)
    # Evaluating resets its environment, so it gets one of its own.
    eval_env = ImaginedEnv(*env_args)
    tau_start, tau_end = cfg.get("tau_start"), cfg.get("tau_end")
    if (tau_start is None) != (tau_end is None):
        raise SystemExit("tau_start and tau_end go together")
    if tau_end is not None:
        eval_env.tau = tau_end
    starts = Starts(Windows(recordings, env.history, device), hanging, cfg.get("upright_start_fraction", 0.0))
    T, N, gamma = cfg["rollout_steps"], cfg["num_envs"], cfg["gamma"]
    reset_p = (1 - gamma) / cfg["reset_horizons"]

    agent = Agent(env.features, cfg["hidden"], cfg["layers"], len(levels)).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)
    sched = trapezoid_scheduler(
        opt, cfg["iterations"], cfg["warmup_iterations"], cfg["cooldown_fraction"]
    )
    first = 1
    timer = Timer(device)
    if ck:
        agent.load_state_dict(ck["agent"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        generator.set_state(ck["rng"]["generator"])
        set_rng_state(ck["rng"]["torch"], device)
        timer = Timer(device, ck["time"])
        env.load_state_dict(ck["envs"])
        first = ck["iteration"] + 1
        if first > cfg["iterations"]:
            raise SystemExit(f"{args.resume} is from the last iteration: nothing to resume")
    else:
        env.reset(starts, starts.sample(N, generator))
        if args.init:
            init = torch.load(args.init, map_location=device, weights_only=False)
            if init["features"] != env.features or not torch.equal(torch.as_tensor(init["levels"]).cpu(), levels.cpu()):
                raise SystemExit(f"{args.init} has other inputs or actions than this config")
            agent.load_state_dict(init["agent"])
            cfg["init"] = str(args.init)

    run = wandb.init(
        project=cfg["wandb_project"],
        name=name,
        config={
            **cfg,
            "commit": git_commit(),
            "device": str(device),
            "levels": levels.tolist(),
            "parameters": sum(p.numel() for p in agent.parameters()),
        },
        mode=args.wandb,
        **({"id": ck["wandb_id"], "resume": "allow"} if ck else {}),
    )
    (out / "checkpoints").mkdir(parents=True, exist_ok=bool(ck))
    hanging_deg = torch.rad2deg(torch.atan2(hanging[:, 0], hanging[:, 1])).tolist()
    print(
        f"{name}: {'resuming at iteration ' + str(first) + ', ' if ck else ''}"
        f"actions {levels.tolist()}, {len(starts)} start windows on {device},"
        f" hanging yaws {', '.join(f'{d:.1f}°' for d in hanging_deg)},"
        f" {len(starts.upright)} upright, {starts.upright_fraction:.0%} of starts from them"
    )

    batch = T * N
    minibatch = batch // cfg["minibatches"]
    x_buf = torch.zeros(T, N, env.features, device=device)
    bin_buf = torch.zeros(T, N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(T, N, device=device)
    value_buf = torch.zeros(T, N, device=device)
    reward_buf = torch.zeros(T, N, device=device)
    penalty_buf = torch.zeros(T, N, device=device)
    next_value_buf = torch.zeros(T, N, device=device)
    reset_buf = torch.zeros(T, N, dtype=torch.bool, device=device)
    up_buf = torch.zeros(T, N, device=device)

    def save(it):
        path = out / "checkpoints" / f"iter_{it:06d}.pt"
        torch.save(
            {
                "config": cfg,
                "levels": levels,
                "hanging": env.hanging.cpu(),
                "features": env.features,
                "agent": agent.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "iteration": it,
                "envs": env.state_dict(),
                "rng": {"generator": generator.get_state(), "torch": rng_state(device)},
                "time": timer.totals,
                "wandb_id": run.id,
            },
            path,
        )
        shutil.copyfile(path, out / "policy.pt")

    for it in range(first, cfg["iterations"] + 1):
        if tau_start is not None:
            env.tau = tau_start + (tau_end - tau_start) * (it - 1) / max(1, cfg["iterations"] - 1)
        with timer("rollout"), torch.no_grad():
            # The iteration's resets, drawn at once on the CPU, so that no
            # frame waits on the device to learn which environments reset.
            resets = torch.rand(T, N, generator=generator) < reset_p
            reset_buf.copy_(resets)
            x = env.observe()
            for t in range(T):
                dist = agent.policy(x)
                bins = dist.sample()
                x_buf[t], bin_buf[t], logp_buf[t] = x, bins, dist.log_prob(bins)
                value_buf[t] = agent.value(x)
                reward_buf[t] = env.step(bins)
                penalty_buf[t] = env.penalty
                up_buf[t] = upright(correct_yaws(env.prediction, env.hanging)).float()
                # The value of where each episode was going, even if it now resets.
                x = env.observe()
                next_value_buf[t] = agent.value(x)
                if resets[t].any():
                    env.restart(resets[t], starts, generator)
                    x = env.observe()
        with timer("advantages"), torch.no_grad():
            advantages, returns = gae(
                reward_buf * (1 - gamma),
                value_buf,
                next_value_buf,
                reset_buf,
                gamma,
                cfg["gae_lambda"],
            )

        with timer("update"):
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
            "train/cos_sum": (reward_buf + penalty_buf).mean().item(),
            "train/speed_penalty": penalty_buf.mean().item(),
            "train/resets": reset_buf.sum().item(),
            "train/upright": up_buf.mean().item(),
            "train/explained_variance": (1 - (rets - values).var() / rets.var()).item(),
            "train/value_mean": values.mean().item(),
            "train/return_mean": rets.mean().item(),
            "lr": sched.get_last_lr()[0],
            "train/tau": env.tau,
        }
        evaluated = it % cfg["eval_every"] == 0 or it == cfg["iterations"]
        if evaluated:
            with timer("eval"):
                eval_metrics, eval_run = evaluate(agent, eval_env, cfg)
            metrics |= eval_metrics
        if it % cfg["checkpoint_every"] == 0 or it == cfg["iterations"]:
            with timer("checkpoint"):
                save(it)
        metrics |= timer.metrics()
        if evaluated:
            print(
                f"iteration {it}: reward {metrics['train/reward']:.3f},"
                f" eval {metrics['eval/reward']:.3f}"
                f" (cos sum {metrics['eval/cos_sum']:.3f}, upright {metrics['eval/upright']:.1%},"
                f" value error {metrics.get('eval/value_error', float('nan')):.3f}),"
                + (f" mean of {cfg['eval_rollouts']} cos sum {metrics['eval/mean/cos_sum']:.3f},"
                   if "eval/mean/cos_sum" in metrics else "") +
                f" entropy {metrics['train/entropy']:.3f},"
                f" eval {metrics['time/share/eval']:.0%} of the time so far",
                flush=True,
            )
        wandb.log(metrics, step=it)

    video = out / "rollout.mp4"
    write_rollout_video(eval_run, eval_env, video)
    wandb.log({"rollout_from_hanging": wandb.Video(str(video), format="mp4")}, step=cfg["iterations"])
    run.finish()


if __name__ == "__main__":
    main()
