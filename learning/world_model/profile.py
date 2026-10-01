"""Times world model training steps and evaluation on the real data.

    uv run python -m world_model.profile
    uv run python -m world_model.profile --widths 256 1024 --depths 3 5 --batches 1024 8192 --variants plain compile+bf16

For each width, depth and batch size: milliseconds per training step,
samples per second, and how much of a step gathering the batch takes,
in each of --variants: plain, compile (torch.compile), bf16 (autocasting
to bfloat16, on CUDA only), or compile+bf16. Then the time of one
evaluation of one validation set as world_model.train runs it. Times are
measured with the device synchronised. --csv also writes the training
step times, one row per model, batch size and variant.

With --flow, the models are flow_lib.FlowWorldModel with the config's head,
and the evaluation is timed in full, as world_model.train runs it with the
config (every set and its upright subset, the ensemble, the likelihood),
broken down into its parts, with its share of the GPU time at the config's
eval_every:

    uv run python -m world_model.profile --config world_model/configs/flow.toml --flow --variants compile+bf16
"""
import argparse
import csv
import time
import tomllib
from pathlib import Path

import torch

from common.data_lib import Windows, hanging_yaws, load_recordings
from common.run_lib import pick_device, synchronize
from world_model.evaluation_lib import ensemble_metrics, evaluate, flow_metrics, rollout
from world_model.flow_lib import FlowWorldModel, flow_losses
from world_model.model_lib import WorldModel, losses


def timed(device, fn, n):
    fn()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(n):
        fn()
    synchronize(device)
    return (time.perf_counter() - started) / n


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("world_model/configs/base.toml"))
    parser.add_argument("--widths", type=int, nargs="+", default=[256])
    parser.add_argument("--depths", type=int, nargs="+", default=[3])
    parser.add_argument("--batches", type=int, nargs="+", default=[1024])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=["plain"],
                        choices=["plain", "compile", "bf16", "compile+bf16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", type=Path, help="also write the step times here")
    parser.add_argument("--flow", action="store_true", help="profile the config's flow model")
    args = parser.parse_args()
    device = pick_device(args.device)
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    rows = []
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    w = cfg["window"]
    recs = load_recordings(Path(cfg["data_dir"]), cfg["train"])
    train = Windows(recs, w + 1, device)
    g = torch.Generator().manual_seed(0)
    print(f"{device}: {len(train)} training windows")
    print(f"{'width':>6} {'depth':>6} {'batch':>7} {'params':>9} {'variant':>13} {'ms/step':>9} {'k samples/s':>12} {'gather':>7}")
    for variant, width in ((v, wd) for v in args.variants for wd in args.widths):
        compile_, bf16 = "compile" in variant, "bf16" in variant
        for depth in args.depths:
            for batch_size in args.batches:
                model = make_model(cfg, width, depth, args.flow).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                forward = torch.compile(model) if compile_ else model

                def gather():
                    return train.gather(train.sample(batch_size, g))

                def step():
                    b = gather()
                    with torch.autocast("cuda", torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                        fit = flow_losses if args.flow else losses
                        mse, bce, *_ = fit(forward, b.obs[:, :w], b.present[:, :w], b.action[:, :w],
                                           b.obs[:, w], b.present[:, w])
                        loss = mse + 0.1 * bce
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                for _ in range(10):
                    step()
                t_step = timed(device, step, args.steps)
                t_gather = timed(device, gather, args.steps)
                params = sum(p.numel() for p in model.parameters())
                print(f"{width:6d} {depth:6d} {batch_size:7d} {params:9d} {variant:>13} "
                      f"{t_step * 1e3:9.2f} {batch_size / t_step / 1e3:12.1f} {t_gather / t_step:7.0%}", flush=True)
                rows.append([device_name, width, depth, batch_size, params, variant,
                             round(t_step * 1e3, 3), round(t_gather / t_step, 3)])
    if args.csv:
        write_csv(args.csv, rows)
    if args.flow:
        model = make_model(cfg, args.widths[0], args.depths[0], True).to(device)
        profile_evaluation(cfg, model, recs, device, rows[-1][6] / 1e3, g)
        return

    val_name, val_names = next(iter(cfg["val"].items()))
    horizon = max(cfg["horizons"])
    windows = Windows(load_recordings(Path(cfg["data_dir"]), val_names), w + horizon, device)
    index = windows.sample(min(cfg["eval_samples"], len(windows)), g)
    model = WorldModel(w, args.widths[0], args.depths[0], 0.05).to(device)
    t_eval = timed(device, lambda: evaluate(model, windows, index, cfg["horizons"]), 3)
    print(f"one evaluation of {val_name} ({len(index)} windows, {horizon} frames, width {args.widths[0]}): {t_eval:.2f} s")


def make_model(cfg: dict, width: int, depth: int, flow: bool):
    if flow:
        return FlowWorldModel.from_config(cfg | {"hidden": width, "layers": depth}, [0.05] * 3)
    return WorldModel(cfg["window"], width, depth, 0.05)


def profile_evaluation(cfg: dict, model: FlowWorldModel, recs, device, t_step: float, g: torch.Generator):
    """Times one evaluation of a flow model as world_model.train runs it,
    each set and its upright subset, and the parts of the validation sets'
    (the likelihood, the rollout on the mean, the ensemble); then its share
    of the time at eval_every, against training steps of t_step seconds."""
    from world_model.train import MIN_UPRIGHT, upright_subset

    model.eval()
    w, horizons = cfg["window"], cfg["horizons"]
    members, nll_steps = cfg.get("eval_ensemble", 8), cfg.get("nll_steps", 32)
    ens_stride = cfg.get("eval_ensemble_stride", 1)
    stride = cfg.get("eval_stride")
    sets = {"train": Windows(recs, w + max(horizons), device)}
    sets |= {k: Windows(load_recordings(Path(cfg["data_dir"]), v), w + max(horizons), device)
             for k, v in cfg["val"].items()}
    hanging = hanging_yaws(recs)
    total = 0.0
    for name, windows in sets.items():
        if stride and name != "train":
            index = torch.arange(0, len(windows), stride, device=device)
        else:
            index = windows.sample(min(cfg["eval_samples"], len(windows)), g)
        up = upright_subset(windows, index, w, hanging)
        for label, idx in ((name, index), (f"{name}/upright", up)):
            if label.endswith("upright") and len(idx) < MIN_UPRIGHT:
                continue
            t = timed(device, lambda: evaluate(model, windows, idx, horizons, members, nll_steps, ens_stride), 1)
            total += t
            print(f"evaluation of {label}: {len(idx)} windows, {t:.2f} s", flush=True)
            if name != "train" and label == name:
                batch = windows.gather(idx)
                with torch.no_grad():
                    parts = {
                        f"likelihood ({nll_steps} RK4 steps)": lambda: flow_metrics(model, batch, nll_steps),
                        f"rollout on the mean ({max(horizons)} frames)": lambda: rollout(model, batch, max(horizons)),
                        f"ensemble ({members} members, every {ens_stride}th window)":
                            lambda: ensemble_metrics(model, windows.gather(idx[::ens_stride]), horizons, members),
                    }
                    for part, fn in parts.items():
                        print(f"  {part}: {timed(device, fn, 1):.2f} s", flush=True)
    every = cfg["eval_every"]
    share = total / (total + every * t_step)
    print(f"one evaluation: {total:.1f} s; {every} training steps: {every * t_step:.1f} s; "
          f"evaluation's share at eval_every {every}: {share:.0%}")


CSV_HEADER = ["device", "width", "depth", "batch", "parameters", "variant", "ms_per_step", "gather_share"]


def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(CSV_HEADER)
        out.writerows(rows)


def parse_log(text: str) -> list[list]:
    """The step times in a log of this script's output (a cloud job's, say),
    as --csv rows; the device is the GPU named on a line of the form
    "NVIDIA L4, 23034 MiB" that nvidia-smi prints, or the one it ran on."""
    device, rows = None, []
    for line in text.splitlines():
        f = line.split()
        if line.endswith(" MiB") and "," in line and device is None:
            device = line.split(",")[0]
        elif f and f[0].endswith(":") and "training windows" in line and device is None:
            device = f[0].rstrip(":")
        elif len(f) == 8 and f[0].isdigit() and f[-1].endswith("%"):
            width, depth, batch, params, variant, ms, _, gather = f
            rows.append([device, int(width), int(depth), int(batch), int(params), variant, float(ms),
                         int(gather.rstrip("%")) / 100])
    return rows


if __name__ == "__main__":
    main()
