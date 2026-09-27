"""Times world model training steps and evaluation on the real data.

    uv run python -m world_model.profile
    uv run python -m world_model.profile --widths 256 1024 --depths 3 5 --batches 1024 8192 --compile

For each width, depth and batch size: milliseconds per training step,
samples per second, and how much of a step gathering the batch takes;
then the time of one evaluation of one validation set as world_model.train
runs it. --compile adds torch.compile, --bf16 autocasting to bfloat16 on
CUDA. Times are measured with the device synchronised.
"""
import argparse
import time
import tomllib
from pathlib import Path

import torch

from common.data_lib import Windows, load_recordings
from common.run_lib import pick_device, synchronize
from world_model.evaluation_lib import evaluate
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
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    w = cfg["window"]
    recs = load_recordings(Path(cfg["data_dir"]), cfg["train"])
    train = Windows(recs, w + 1, device)
    g = torch.Generator().manual_seed(0)
    print(f"{device}: {len(train)} training windows")
    print(f"{'width':>6} {'depth':>6} {'batch':>7} {'params':>9} {'compile':>8} {'ms/step':>9} {'k samples/s':>12} {'gather':>7}")
    for width in args.widths:
        for depth in args.depths:
            for batch_size in args.batches:
                model = WorldModel(w, width, depth, 0.05).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                forward = torch.compile(model) if args.compile else model

                def gather():
                    return train.gather(train.sample(batch_size, g))

                def step():
                    b = gather()
                    with torch.autocast("cuda", torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                        mse, bce = losses(forward, b.obs[:, :w], b.present[:, :w], b.action[:, :w],
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
                print(f"{width:6d} {depth:6d} {batch_size:7d} {params:9d} {str(args.compile):>8} "
                      f"{t_step * 1e3:9.2f} {batch_size / t_step / 1e3:12.1f} {t_gather / t_step:7.0%}", flush=True)

    val_name, val_names = next(iter(cfg["val"].items()))
    horizon = max(cfg["horizons"])
    windows = Windows(load_recordings(Path(cfg["data_dir"]), val_names), w + horizon, device)
    index = windows.sample(min(cfg["eval_samples"], len(windows)), g)
    model = WorldModel(w, args.widths[0], args.depths[0], 0.05).to(device)
    t_eval = timed(device, lambda: evaluate(model, windows, index, cfg["horizons"]), 3)
    print(f"one evaluation of {val_name} ({len(index)} windows, {horizon} frames, width {args.widths[0]}): {t_eval:.2f} s")


if __name__ == "__main__":
    main()
