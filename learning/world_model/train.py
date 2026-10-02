"""Trains the world model and logs its metrics to Weights & Biases.

    uv run python -m world_model.train world_model/configs/base.toml --name base
    uv run python -m world_model.train world_model/configs/base.toml --name smoke --wandb disabled --steps 2000
    uv run python -m world_model.train world_model/configs/base.toml --name long --set cooldown_fraction=0
    uv run python -m world_model.train --resume runs/world_model/long/checkpoints/step_0050000.pt
    uv run python -m world_model.train --branch runs/world_model/long/checkpoints/step_0050000.pt \\
        --cooldown-steps 10000 --name long-cd50k

Every `eval_every` steps, training pauses to evaluate on a fixed sample of
windows from the training recordings and from each validation set: the
same sample each time, drawn from `seed`. `val` in the config names the
validation sets, each a list of recordings whose metrics are logged under
the set's name; a plain list is one set, `val`. Each set's windows that
start with arms 0 and 1 upright are also scored on their own, under
`<set>/upright/`: the regime the policies now live in.

The learning rate follows the trapezoid: a warmup, a constant rate, and a
cooldown over the last `cooldown_fraction` of `steps`, or `cooldown_steps`
if the config has it. With no cooldown, the rate stays constant, and
--branch starts a cooldown of its own from any checkpoint: one long run
then gives finished models at many lengths, each for the cost of its
cooldown. A branch is a run of its own, whose steps continue from the
checkpoint's.

Every `checkpoint_every` steps, at the steps in `checkpoint_at`, and at the
end, the whole training state goes to
`runs/world_model/<name>/checkpoints/step_NNNNNNN.pt`, and the weights to
`model.pt`. --resume continues from a checkpoint in the same directory and
W&B run. Each phase is timed with the device synchronised, as in policy
training, and logged at each evaluation.

With `flow`, the model is a flow_lib.FlowWorldModel, trained on the
flow-matching loss one step ahead; with `yaw_gaussian`, a
flow_lib.YawGaussianWorldModel, its Gaussian twin, on the beta-NLL of the
yaw changes one step ahead.
"""
import argparse
import math
from pathlib import Path

import torch
import wandb

from common.data_lib import Windows, correct_yaws, hanging_yaws, load_recordings
from common.run_lib import Timer, git_commit, load_config, pick_device, rng_state, set_rng_state
from common.schedule_lib import trapezoid
from world_model import flow_lib
from world_model.evaluation_lib import evaluate
from world_model.model_lib import (LOGVAR_MIN, TwoStageWorldModel, WorldModel, last_seen, load_world_model,
                                   rollout_losses, variance_losses)

UPRIGHT_COS = math.cos(math.radians(30))
# Config keys a run understands without the config file having them.
OPTIONAL = {"checkpoint_at", "checkpoint_every", "cooldown_steps", "compile", "bf16", "rollout_train", "max_grad_norm",
            "stochastic", "nll_beta", "eval_ensemble", "eval_stride", "logvar_min", "fold_block", "train_fold",
            "mean_model", "mean_folds", "flow", "flow_hidden", "flow_layers", "flow_steps", "nll_steps",
            "eval_ensemble_stride", "yaw_gaussian"}
# Upright subsets smaller than this are not scored.
MIN_UPRIGHT = 64


@torch.no_grad()
def change_scale(windows: Windows, samples: int, generator: torch.Generator) -> float:
    """RMS of one frame's change in sin and cos, over training windows of
    `window + 1` steps."""
    w = windows.length - 1
    batch = windows.gather(windows.sample(samples, generator))
    ref, has_ref = last_seen(batch.obs[:, :w], batch.present[:, :w])
    mask = batch.present[:, w] & has_ref
    return (batch.obs[:, w] - ref)[mask].pow(2).mean().sqrt().item()


def cooldown_steps(cfg: dict) -> int:
    return cfg.get("cooldown_steps", round(cfg["steps"] * cfg["cooldown_fraction"]))


def upright_subset(windows: Windows, index: torch.Tensor, w: int, hanging: torch.Tensor) -> torch.Tensor:
    """The windows at `index` whose last input frame has arms 0 and 1 seen
    and within 30° of upright."""
    last = windows.starts[index] + w - 1
    cos = correct_yaws(windows.obs[last], hanging.to(windows.obs.device))[..., 1]
    up = (cos[:, :2] > UPRIGHT_COS).all(-1) & windows.present[last][:, :2].all(-1)
    return index[up]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, nargs="?", help="for a new run")
    parser.add_argument("--name", help="the new run's name, here and in W&B")
    parser.add_argument("--resume", type=Path, help="a checkpoint to continue from")
    parser.add_argument("--branch", type=Path, help="a checkpoint to cool down from, as a new run")
    parser.add_argument("--cooldown-steps", type=int, help="the branch's cooldown")
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, help="override a new run's steps")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a new run's config value, VALUE as TOML; repeatable")
    args = parser.parse_args(argv)
    if args.resume:
        if args.config or args.name or args.branch or args.steps or args.set:
            parser.error("--resume takes everything from the checkpoint")
    elif args.branch:
        if args.config or args.steps or args.set or not (args.name and args.cooldown_steps):
            parser.error("--branch needs --name and --cooldown-steps, and takes the rest from the checkpoint")
    elif not (args.config and args.name):
        parser.error("a new run needs a config and --name")
    return args


def main(argv=None):
    args = parse_args(argv)
    ck = None
    if args.resume or args.branch:
        ck = torch.load(args.resume or args.branch, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
    if args.resume:
        out = args.resume.parent.parent
        name = out.name
    else:
        name = args.name
        out = Path("runs/world_model") / name
        if out.exists():
            raise SystemExit(f"{out} exists: pick another --name")
    if args.branch:
        cfg["steps"] = ck["step"] + args.cooldown_steps
        cfg["cooldown_steps"] = args.cooldown_steps
        cfg["branch_of"] = {"run": args.branch.parent.parent.name, "step": ck["step"]}
    if not ck:
        cfg = load_config(args.config, args.set, OPTIONAL)
        if args.steps is not None:
            cfg["steps"] = args.steps
    start = ck["step"] if ck else 0
    if start >= cfg["steps"]:
        raise SystemExit(f"{args.resume} is from the last step: nothing to resume")

    val_sets = cfg["val"] if isinstance(cfg["val"], dict) else {"val": cfg["val"]}
    for set_name, names in val_sets.items():
        if set(cfg["train"]) & set(names):
            raise SystemExit(f"a recording is in both train and {set_name}")
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    train_recs = load_recordings(data_dir, cfg["train"])
    w, horizon = cfg["window"], max(cfg["horizons"])
    # Training windows hold the window and the frames the loss rolls out over.
    rollout_k = cfg.get("rollout_train", 1)
    train = Windows(train_recs, w + rollout_k, device)
    if cfg.get("train_fold") is not None:
        # One half of the data, in alternating blocks of fold_block steps.
        train.keep_fold(cfg["fold_block"], cfg["train_fold"])
    eval_sets = {
        "train": Windows(train_recs, w + horizon, device),
        **{
            set_name: Windows(load_recordings(data_dir, names), w + horizon, device)
            for set_name, names in val_sets.items()
        },
    }
    # The training set's evaluation windows are a fixed random sample; the
    # validation sets' too, or with eval_stride every eval_stride-th start
    # position, uniform in time over their recordings.
    stride = cfg.get("eval_stride")
    eval_index = {
        k: torch.arange(0, len(v), stride, device=v.starts.device) if stride and k != "train"
        else v.sample(min(cfg["eval_samples"], len(v)), generator)
        for k, v in eval_sets.items()
    }
    hanging = hanging_yaws(train_recs)
    upright_index = {k: upright_subset(v, eval_index[k], w, hanging) for k, v in eval_sets.items()}
    upright_index = {k: v for k, v in upright_index.items() if len(v) >= MIN_UPRIGHT}

    flow = bool(cfg.get("flow"))
    yaw_gaussian = bool(cfg.get("yaw_gaussian"))
    if flow and yaw_gaussian:
        raise SystemExit("flow and yaw_gaussian are two kinds of model")
    if flow or yaw_gaussian:
        # Each tag's typical yaw change, in radians.
        one = Windows(train_recs, w + 1, device)
        batch = one.gather(one.sample(65536, generator))
        scale = flow_lib.change_scale(batch.obs, batch.present, w)
    else:
        scale = change_scale(Windows(train_recs, w + 1, device), 65536, generator)
    if ck:
        scale = ck["delta_scale"]
    folds = None
    if flow or yaw_gaussian:
        if rollout_k != 1:
            raise SystemExit("a yaw-space model trains one step ahead")
        kind = flow_lib.FlowWorldModel if flow else flow_lib.YawGaussianWorldModel
        model = kind.from_config(cfg, scale).to(device)
    elif cfg.get("mean_model"):
        # The second stage of a two-stage model: a variance for the mean
        # model, trained on each window's mean from the fold model that did
        # not see it.
        if rollout_k != 1:
            raise SystemExit("a two-stage model trains one step ahead")
        folds = [load_world_model(p, device) for p in cfg["mean_folds"]]
        model = TwoStageWorldModel(load_world_model(cfg["mean_model"], device), w, cfg["hidden"], cfg["layers"],
                                   scale, cfg.get("logvar_min", LOGVAR_MIN)).to(device)
    else:
        model = WorldModel(w, cfg["hidden"], cfg["layers"], scale, cfg.get("stochastic", False),
                           cfg.get("logvar_min", LOGVAR_MIN)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Training can run the model compiled, and its matrix products in
    # bfloat16 on CUDA; evaluation always runs it as it is, in float32.
    forward = torch.compile(model) if cfg.get("compile") else model
    bf16 = bool(cfg.get("bf16")) and device.type == "cuda"
    timer = Timer(device)
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        generator.set_state(ck["rng"]["generator"])
        set_rng_state(ck["rng"]["torch"], device)
        if args.resume:
            timer = Timer(device, ck["time"])
    cooldown = cooldown_steps(cfg)

    run = wandb.init(
        project=cfg["wandb_project"],
        name=name,
        config={
            **cfg,
            "commit": git_commit(),
            "device": str(device),
            "delta_scale": scale,
            "train_windows": len(train),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        mode=args.wandb,
        **({"id": ck["wandb_id"], "resume": "allow"} if args.resume else {}),
    )
    (out / "checkpoints").mkdir(parents=True, exist_ok=bool(args.resume))
    torch.save({k: v.cpu() for k, v in eval_index.items()}, out / "eval_index.pt")
    print(
        f"{name}: {len(train)} training windows on {device}, delta_scale {scale}, "
        f"steps {start + 1}-{cfg['steps']}, cooldown {cooldown}, "
        f"upright eval windows {({k: len(v) for k, v in upright_index.items()})}",
        flush=True,
    )

    # A stochastic model is also scored as an ensemble of sampled rollouts.
    members = cfg.get("eval_ensemble", 8) if model.stochastic else 0
    nll_steps = cfg.get("nll_steps", 32)
    # The ensemble, the costly part of a flow model's evaluation, scores
    # every eval_ensemble_stride-th window.
    ens_stride = cfg.get("eval_ensemble_stride", 1)

    def evaluate_all(step) -> dict:
        metrics = {}
        for split, windows in eval_sets.items():
            for k, v in evaluate(model, windows, eval_index[split], cfg["horizons"], members, nll_steps,
                                 ens_stride).items():
                metrics[f"{split}/{k}"] = v
            if split in upright_index:
                for k, v in evaluate(model, windows, upright_index[split], cfg["horizons"], members,
                                     nll_steps, ens_stride).items():
                    metrics[f"{split}/upright/{k}"] = v
        shown = [h for h in (16, 64) if h in cfg["horizons"]] or cfg["horizons"][-2:]
        scores = [
            f"{s} " + " ".join(f"h{h:03d} {metrics[f'{s}/one_minus_r2/h{h:03d}']:.4f}" for h in shown)
            for s in val_sets
        ]
        print(f"step {step}: " + ", ".join(scores), flush=True)
        return metrics

    def save(step):
        state = {
            "config": cfg,
            "delta_scale": scale,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "rng": {"generator": generator.get_state(), "torch": rng_state(device)},
            "time": timer.totals,
            "wandb_id": run.id,
        }
        torch.save(state, out / "checkpoints" / f"step_{step:07d}.pt")
        torch.save({"config": cfg, "delta_scale": scale, "model": model.state_dict()}, out / "model.pt")

    checkpoint_at = set(cfg.get("checkpoint_at", [])) | {cfg["steps"]}
    every = cfg.get("checkpoint_every", 0)
    is_eval = lambda s: s % cfg["eval_every"] == 0 or s == cfg["steps"]
    is_checkpoint = lambda s: (every and s % every == 0) or s in checkpoint_at

    if start == 0:
        with timer("eval"):
            metrics = evaluate_all(0)
        wandb.log(metrics | timer.metrics(), step=0)
    step = start
    while step < cfg["steps"]:
        # Train up to the next evaluation or checkpoint as one timed chunk,
        # so that timing does not synchronise the device every step.
        end = step + 1
        while not (is_eval(end) or is_checkpoint(end) or end == cfg["steps"]):
            end += 1
        with timer("train"):
            for step in range(step + 1, end + 1):
                lr = cfg["lr"] * trapezoid(step - 1, cfg["steps"], cfg["warmup_steps"], cooldown)
                for g in opt.param_groups:
                    g["lr"] = lr
                index = train.sample(cfg["batch_size"], generator)
                batch = train.gather(index)
                with torch.autocast("cuda", torch.bfloat16, enabled=bf16):
                    if flow:
                        fit, bce = flow_lib.flow_losses(forward, batch.obs[:, :w], batch.present[:, :w],
                                                        batch.action[:, :w], batch.obs[:, w], batch.present[:, w])
                    elif yaw_gaussian:
                        fit, bce = flow_lib.yaw_gaussian_losses(forward, batch.obs[:, :w], batch.present[:, :w],
                                                                batch.action[:, :w], batch.obs[:, w],
                                                                batch.present[:, w], cfg.get("nll_beta", 0.0))
                    elif folds:
                        fit, bce, mse = variance_losses(model, folds, batch.obs[:, :w], batch.present[:, :w],
                                                        batch.action[:, :w], batch.obs[:, w], batch.present[:, w],
                                                        train.fold_of(index, cfg["fold_block"]))
                    else:
                        fit, bce, mse = rollout_losses(forward, batch.obs, batch.present, batch.action, rollout_k,
                                                       cfg.get("nll_beta", 0.0))
                    loss = fit + cfg["bce_weight"] * bce
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.get("max_grad_norm"):
                    # Long rollouts can explode the gradient through the model's own predictions.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                opt.step()
                if step % cfg["log_every"] == 0:
                    logged = {"train/batch_bce": bce.item(), "lr": lr}
                    if flow:
                        logged["train/batch_fm"] = fit.item()
                    elif not yaw_gaussian:
                        logged["train/batch_mse"] = mse.item()
                    if model.stochastic and not flow:
                        logged["train/batch_beta_nll"] = fit.item()
                    wandb.log(logged, step=step)
        metrics = {}
        if is_eval(step):
            with timer("eval"):
                metrics = evaluate_all(step)
        if is_checkpoint(step):
            with timer("checkpoint"):
                save(step)
        if metrics:
            wandb.log(metrics | timer.metrics(), step=step)
    run.finish()


if __name__ == "__main__":
    main()
