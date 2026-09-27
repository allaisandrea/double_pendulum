"""Tabulates the final metrics of world model runs from W&B.

    uv run python -m world_model.report batch-
    uv run python -m world_model.report wm-w --sets val_policy_wm3 val_policy_wm2 --horizons 16 64
    uv run python -m world_model.report rollout --horizons 16 64 125 --csv rollout.csv

For every run in the W&B project whose name starts with PREFIX: 1 - R² at
--horizons on each of --sets, on all its windows and on those starting
with arms 0 and 1 upright (u), with the parameters and the windows it
trained on. Lower is better. --csv also writes them, with the runs'
training settings, one row per run: columns <set>/h016,
<set>/upright/h016 and so on.
"""
import argparse
import csv

from pathlib import Path

import wandb

PROJECT = "allais-andrea-personal/double-pendulum-world-model"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prefix")
    parser.add_argument("--sets", nargs="+", default=["val_policy_wm3", "val_policy_wm2", "val_random_walk"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--csv", type=Path, help="also write the table here")
    args = parser.parse_args()
    runs = [r for r in wandb.Api().runs(PROJECT) if r.name.startswith(args.prefix)]
    columns = [(s, sub, h) for s in args.sets for sub in ("", "upright/") for h in args.horizons]
    heads = [f"{s.removeprefix('val_')[:10]}{' u' if sub else ''} h{h}" for s, sub, h in columns]
    width = max([len(r.name) for r in runs] + [4])
    print(f"{'run':{width}s} {'state':>8s} {'params':>8s} {'M windows':>9s} " + " ".join(f"{h:>17s}" for h in heads))
    for r in sorted(runs, key=lambda r: r.name):
        s, c = r.summary, r.config
        step = s.get("_step", 0)
        seen = step * c.get("batch_size", 0) / 1e6
        values = [s.get(f"{st}/{sub}one_minus_r2/h{h:03d}") for st, sub, h in columns]
        cells = " ".join(f"{v:17.4f}" if isinstance(v, (int, float)) else f"{'-':>17s}" for v in values)
        print(f"{r.name:{width}s} {r.state:>8s} {c.get('parameters', 0) / 1e6:7.2f}M {seen:9.0f} {cells}")
    if args.csv:
        write_csv(args.csv, sorted(runs, key=lambda r: r.name), columns)


# Training settings written with each run, as world_model.train logs them.
SETTINGS = ["hidden", "layers", "window", "parameters", "batch_size", "lr", "bf16", "compile",
            "rollout_train", "max_grad_norm", "steps", "seed"]


def write_csv(path: Path, runs, columns):
    """One row per run: name, state, steps done, the step it branched from
    (world_model.scale's cooldowns, else empty), SETTINGS and the metrics."""
    heads = [f"{st}/{sub}h{h:03d}" for st, sub, h in columns]
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["run", "state", "step", "branch_step", *SETTINGS, *heads])
        for r in runs:
            s, c = r.summary, r.config
            values = [s.get(f"{st}/{sub}one_minus_r2/h{h:03d}") for st, sub, h in columns]
            out.writerow([r.name, r.state, s.get("_step", ""), (c.get("branch_of") or {}).get("step", ""),
                          *(c.get(k, "") for k in SETTINGS),
                          *(v if isinstance(v, (int, float)) else "" for v in values)])


if __name__ == "__main__":
    main()
