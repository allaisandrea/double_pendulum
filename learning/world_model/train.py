"""Trains the world model and logs its metrics to Weights & Biases.

    uv run python -m world_model.train world_model/configs/base.toml --name base
    uv run python -m world_model.train world_model/configs/base.toml --name smoke --wandb disabled --steps 200

Every `eval_every` steps, training pauses to evaluate on a fixed sample of
windows from the training and the validation recordings: the same sample
each time, drawn from `seed`. The run's config, its evaluation sample and
the latest checkpoint go to `runs/<name>/`, which must not exist yet; the
name is also the run's name in Weights & Biases.
"""
import argparse
import subprocess
import tomllib
from pathlib import Path

import torch
import wandb

from common.data_lib import Windows, load_recordings
from common.schedule_lib import trapezoid_scheduler
from world_model.evaluation_lib import evaluate
from world_model.model_lib import WorldModel, last_seen, losses


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.no_grad()
def change_scale(windows: Windows, samples: int, generator: torch.Generator) -> float:
    """RMS of one frame's change in sin and cos, over training windows of
    `window + 1` steps."""
    w = windows.length - 1
    batch = windows.gather(windows.sample(samples, generator))
    ref, has_ref = last_seen(batch.obs[:, :w], batch.present[:, :w])
    mask = batch.present[:, w] & has_ref
    return (batch.obs[:, w] - ref)[mask].pow(2).mean().sqrt().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", required=True, help="the run's name, here and in W&B")
    parser.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, help="override the config's steps")
    args = parser.parse_args()
    out = Path("runs") / args.name
    if out.exists():
        raise SystemExit(f"{out} exists: pick another --name")

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if set(cfg["train"]) & set(cfg["val"]):
        raise SystemExit("a recording is in both train and val")
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    train_recs = load_recordings(data_dir, cfg["train"])
    val_recs = load_recordings(data_dir, cfg["val"])
    w, horizon = cfg["window"], max(cfg["horizons"])
    train = Windows(train_recs, w + 1, device)
    eval_sets = {
        "train": Windows(train_recs, w + horizon, device),
        "val": Windows(val_recs, w + horizon, device),
    }
    eval_index = {
        k: v.sample(min(cfg["eval_samples"], len(v)), generator) for k, v in eval_sets.items()
    }

    scale = change_scale(train, 65536, generator)
    model = WorldModel(w, cfg["hidden"], cfg["layers"], scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = trapezoid_scheduler(opt, cfg["steps"], cfg["warmup_steps"], cfg["cooldown_fraction"])

    run = wandb.init(
        project=cfg["wandb_project"],
        name=args.name,
        config={
            **cfg,
            "commit": git_commit(),
            "device": str(device),
            "delta_scale": scale,
            "train_windows": len(train),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        mode=args.wandb,
    )
    out.mkdir(parents=True)
    torch.save({k: v.cpu() for k, v in eval_index.items()}, out / "eval_index.pt")
    print(f"{args.name}: {len(train)} training windows on {device}, delta_scale {scale:.4g}")

    def log_eval(step):
        metrics = {}
        for split, windows in eval_sets.items():
            for k, v in evaluate(model, windows, eval_index[split], cfg["horizons"]).items():
                metrics[f"{split}/{k}"] = v
        wandb.log(metrics, step=step)
        h = f"h{horizon:03d}"
        print(
            f"step {step}: val mse {metrics['val/mse']:.4f}"
            f" 1-R² h001 {metrics['val/one_minus_r2/h001']:.4f}"
            f" {h} {metrics[f'val/one_minus_r2/{h}']:.4f}"
            f" (copy-last {metrics[f'val/copy_last/{h}']:.4f})",
            flush=True,
        )
        torch.save({"config": cfg, "delta_scale": scale, "model": model.state_dict()}, out / "model.pt")

    log_eval(0)
    for step in range(1, cfg["steps"] + 1):
        batch = train.gather(train.sample(cfg["batch_size"], generator))
        mse, bce = losses(
            model,
            batch.obs[:, :w],
            batch.present[:, :w],
            batch.action[:, :w],
            batch.obs[:, w],
            batch.present[:, w],
        )
        loss = mse + cfg["bce_weight"] * bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0:
            wandb.log(
                {"train/batch_mse": mse.item(), "train/batch_bce": bce.item(), "lr": sched.get_last_lr()[0]},
                step=step,
            )
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            log_eval(step)
    run.finish()


if __name__ == "__main__":
    main()
