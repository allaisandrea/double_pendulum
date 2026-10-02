"""Evaluates world model checkpoints side by side on the same validation windows.

    uv run python -m world_model.compare runs/world_model/wm3/model.pt runs/world_model/wm-with-policy-50k/model.pt \\
        --val 1790281442 1790374602 1790378604

Each recording given to --val is its own set, with one fixed sample of
windows drawn from --seed and shared by every checkpoint, so the numbers
compare like for like, whatever each model was trained or validated on.
Prints 1 - R² at each horizon and the one-step missing-tag cross-entropy,
as world_model.train logs them; for a stochastic model, also its
ensemble metrics over --ensemble sampled rollouts per window
(evaluation_lib.ensemble_metrics). --stride takes every STRIDE-th start
position rather than --samples random ones; --upright keeps the windows
starting with arms 0 and 1 upright, and skips a recording with fewer than
64 of them. --csv writes every metric, one row per checkpoint and
recording. Each model sees the recordings as it was trained to: as tag
yaws, or as calibrated link angles (common.calibrate).

A flow model's checkpoint can be given as PATH@STEPS, to sample it in
STEPS midpoint steps rather than its config's `flow_steps`; the same
checkpoint may appear at several. Every stochastic model is also scored on
its one-step forecast of the yaw changes, from --yaw-members draws per
window (evaluation_lib.yaw_coverage), which compares models of every
kind; --ensemble-stride scores the ensembles on every Nth window only.
"""
import argparse
import csv
import tomllib
from pathlib import Path

import torch

from common.data_lib import Windows, hanging_yaws, load_recordings
from common.run_lib import pick_device
from world_model.evaluation_lib import evaluate, yaw_coverage
from world_model.model_lib import load_world_model
from world_model.train import MIN_UPRIGHT, upright_subset

HORIZONS = [1, 4, 16, 64, 125]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", nargs="+", help="PATH, or PATH@STEPS for a flow model")
    parser.add_argument("--val", nargs="+", required=True, help="recordings, each its own set")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--stride", type=int, help="every STRIDE-th start position instead of --samples random ones")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=8, help="sampled rollouts per window, stochastic models")
    parser.add_argument("--upright", action="store_true", help="only the windows starting with arms 0 and 1 upright")
    parser.add_argument("--ensemble-stride", type=int, default=1, help="the ensembles on every Nth window only")
    parser.add_argument("--yaw-members", type=int, default=64, help="draws per window for the yaw forecast")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    rows = []
    device = pick_device(args.device)

    models, names = {}, {}
    for spec in map(str, args.checkpoints):
        path, at, steps = spec.rpartition("@")
        if not at:
            path, steps = spec, None
        model = load_world_model(Path(path), device)
        if steps:
            model.flow_steps = int(steps)
        models[spec] = model
        names[spec] = Path(path).parent.name + (f"@{steps}" if steps else "")
    window = max(m.window for m in models.values())
    length = window + max(HORIZONS)
    if args.upright:
        # The hanging yaws, from the recordings the world models trained on.
        with open("world_model/configs/base.toml", "rb") as f:
            hanging = hanging_yaws(load_recordings(args.data_dir, tomllib.load(f)["train"])).to(device)
    width = max(len(n) for n in names.values())
    observations = sorted({m.observation for m in models.values()})
    for rec in args.val:
        # Each model sees the recording as it was trained to (yaw or
        # calibrated angles); both have the same steps, so the same windows.
        seen_as = {o: Windows(load_recordings(args.data_dir, [rec], o), length, device) for o in observations}
        windows = seen_as.get("yaw", seen_as[observations[0]])
        if args.stride:
            index = torch.arange(0, len(windows), args.stride, device=device)
        else:
            index = windows.sample(min(args.samples, len(windows)), torch.Generator().manual_seed(args.seed))
        if args.upright:
            index = upright_subset(windows, index, window, hanging)
            if len(index) < MIN_UPRIGHT:
                print(f"\n{rec}: only {len(index)} windows start upright; skipped")
                continue
        print(f"\n{rec}: 1 - R² by horizon (frames), and one-step bce")
        print(" " * width + "".join(f"{f'h{h:03d}':>8s}" for h in HORIZONS) + f"{'bce':>8s}")
        for path, model in models.items():
            # A model with a shorter window sees the end of each longer one.
            trimmed = _Trimmed(seen_as[model.observation], window - model.window)
            m = evaluate(model, trimmed, index, HORIZONS, args.ensemble, ensemble_stride=args.ensemble_stride)
            if model.stochastic:
                m |= yaw_coverage(model, trimmed.gather(index[::args.ensemble_stride]), args.yaw_members)
            row = [m[f"one_minus_r2/h{h:03d}"] for h in HORIZONS] + [m["bce"]]
            print(f"{names[path]:{width}s}" + "".join(f"{v:8.4f}" for v in row))
            for key in ("ensemble/one_minus_r2", "ensemble/crps", "ensemble/spread_skill"):
                if f"{key}/h001" in m:
                    label = f"  {key.split('/')[1]}"
                    print(f"{label:{width}s}" + "".join(f"{m[f'{key}/h{h:03d}']:8.4f}" for h in HORIZONS))
            if model.stochastic:
                print(f"{'  yaw one step':{width}s}" + "".join(f"  {k.split('/')[1]} {m[k]:.3f}" for k in
                      ("yaw/within_68", "yaw/within_95", "yaw/crps"))
                      + (f"  nll {m['nll']:.3f} median {m['median_nll']:.3f}" if "fm" in m or "fit" in m else ""))
            rows.append({"checkpoint": names[path], "recording": rec, **m})
    if args.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="") as f:
            out = csv.DictWriter(f, keys)
            out.writeheader()
            out.writerows(rows)


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
