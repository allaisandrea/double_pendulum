"""Evaluates world model checkpoints side by side on the same validation windows.

    uv run python -m world_model.compare runs/world_model/wm3/model.pt runs/world_model/wm-with-policy-50k/model.pt \\
        --val 1790281442 1790374602 1790378604

Each recording given to --val is its own set, with one fixed sample of
windows drawn from --seed and shared by every checkpoint, so the numbers
compare like for like, whatever each model was trained or validated on.
Prints 1 - R² at each horizon and the one-step missing-tag cross-entropy,
as world_model.train logs them.
"""
import argparse
from pathlib import Path

import torch

from common.data_lib import Windows, load_recordings
from common.run_lib import pick_device
from world_model.evaluation_lib import evaluate
from world_model.model_lib import load_world_model

HORIZONS = [1, 4, 16, 64, 125]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--val", nargs="+", required=True, help="recordings, each its own set")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)

    models = {p: load_world_model(p, device) for p in args.checkpoints}
    window = max(m.window for m in models.values())
    length = window + max(HORIZONS)
    names = {p: p.parent.name for p in args.checkpoints}
    width = max(len(n) for n in names.values())
    for rec in args.val:
        windows = Windows(load_recordings(args.data_dir, [rec]), length, device)
        index = windows.sample(min(args.samples, len(windows)), torch.Generator().manual_seed(args.seed))
        print(f"\n{rec}: 1 - R² by horizon (frames), and one-step bce")
        print(" " * width + "".join(f"{f'h{h:03d}':>8s}" for h in HORIZONS) + f"{'bce':>8s}")
        for path, model in models.items():
            # A model with a shorter window sees the end of each longer one.
            m = evaluate(model, _Trimmed(windows, window - model.window), index, HORIZONS)
            row = [m[f"one_minus_r2/h{h:03d}"] for h in HORIZONS] + [m["bce"]]
            print(f"{names[path]:{width}s}" + "".join(f"{v:8.4f}" for v in row))


class _Trimmed:
    """`windows`, less the first `skip` steps of each."""

    def __init__(self, windows: Windows, skip: int):
        self.windows, self.skip = windows, skip

    def gather(self, index):
        b = self.windows.gather(index)
        s = self.skip
        return type(b)(b.obs[:, s:], b.present[:, s:], b.action[:, s:])


if __name__ == "__main__":
    main()
