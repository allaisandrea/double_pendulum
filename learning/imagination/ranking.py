"""Ranks policies on the rig and in world models, and scores the world models' rankings.

    uv run python -m imagination.ranking export
    uv run python -m imagination.ranking imagine ../recordings/ranking-*-seed0.tsv ../recordings/ranking-*-seed1.tsv \
        --world-models runs/world_model/rollout3-k8/model.pt runs/world_model/rollout-k16/model.pt \
        --scores runs/ranking/scores.csv
    uv run python -m imagination.ranking agreement runs/ranking/scores.csv

export writes each policy in the pool (imagination/ranking/pool.txt) as
the harness's JSON, to runs/ranking/<name>.json, for
imagination/ranking/collect.sh to run on the rig.

imagine reads the manifests that script writes (policy name,
recording), pools each policy's episodes across all its recordings,
scores it on the rig over them (imagination.ranking_lib), and in each
world model from the same starts, greedy or sampling as the recording's
policy did. It prints the scores, writes them to --scores, and prints
their agreement, as agreement does.

agreement reads the scores and prints, per world model, how well they
agree with the rig's: Pearson and Spearman correlation, mean maximum rank
violation (MMRV), mean absolute error and bias (mean overrating), over
all policies, those new to the rig, and, for a world model some policies
were trained in, the others (held out) and those (own), with its
exploitation gap: how much more it overrates its own policies than the
others. It takes no rollouts, so new metrics need no new imagining.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common.data_lib import hanging_yaws, load_recordings
from common.run_lib import pick_device
from imagination.export_lib import export_policy
from imagination.agent_lib import Agent
from imagination.ranking_lib import (Scores, agreements, episodes, exploitation_gap, imagined, read_scores,
                                     sampled, write_agreements, write_scores)
from imagination.rollout_lib import load_policy
from imagination.sim2real import HANGING
from world_model.model_lib import load_world_model

POOL = Path("imagination/ranking/pool.txt")
OUT = Path("runs/ranking")


def read_pool(path: Path) -> dict[str, tuple[Path, bool]]:
    pool = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, checkpoint, seen = line.split()
            pool[name] = (Path(checkpoint), seen == "seen")
    return pool


def export(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (checkpoint, _) in read_pool(args.pool).items():
        out = OUT / f"{name}.json"
        if out.exists():
            continue
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        agent = Agent(ck["features"], cfg["hidden"], cfg["layers"], len(ck["levels"]))
        agent.load_state_dict(ck["agent"])
        export_policy(agent, ck["levels"], cfg["policy_window"], out,
                      {"checkpoint": str(checkpoint), "iteration": ck.get("iteration"), "ranking": name})
        print(f"{out}: from {checkpoint}")


def world_model_spec(spec: str, tau: float) -> tuple[str, Path, float]:
    """A --world-models entry, PATH or PATH@TAU: its column name (the run's
    directory, with @TAU if given), its checkpoint and its tau."""
    path, at, value = spec.rpartition("@")
    if at and path.endswith(".pt"):
        return f"{Path(path).parent.name}@{value}", Path(path), float(value)
    return Path(spec).parent.name, Path(spec), tau


def imagine(args):
    device = pick_device(args.device)
    pool = read_pool(args.pool)
    hanging = hanging_yaws(load_recordings(Path("data"), HANGING)).numpy()
    recordings: dict[str, list[str]] = {}
    for manifest in args.manifests:
        for line in manifest.read_text().splitlines():
            if line.strip():
                name, recording = line.split("\t")
                recordings.setdefault(name, []).append(recording)
    specs = [world_model_spec(w, args.tau) for w in args.world_models]
    models = {name: load_world_model(path, device) for name, path, _ in specs}
    taus = {name: tau for name, _, tau in specs}
    names, seen, counts, trained_in, real, sem, predicted = [], [], [], [], [], [], {m: [] for m in models}
    print(f"{'policy':18s} {'eps':>4s} {'rig':>14s} " + " ".join(f"{m:>18s}" for m in models))
    for name, paths in recordings.items():
        checkpoint, was_seen = pool[name]
        agent, env, cfg = load_policy(checkpoint, device)
        groups = []  # (episodes, greedy) per recording
        for recording in paths:
            frames = Path(recording) / "frames.arrows"
            groups.append((episodes(frames, hanging, env.history, seconds=args.seconds), not sampled(frames)))
        eps = [e for group, _ in groups for e in group]
        scores = np.array([e.real for e in eps])
        row = []
        for m, model in models.items():
            env.model, env.tau = model, taus[m]
            # Each recording's starts, weighted by how many episodes it has.
            parts = [(imagined(agent, env, [e.start for e in group], args.seconds, args.rollouts, args.seed, greedy), len(group))
                     for group, greedy in groups if group]
            v = sum(value * n for value, n in parts) / sum(n for _, n in parts)
            predicted[m].append(v)
            row.append(v)
        names.append(name)
        seen.append(was_seen)
        counts.append(len(eps))
        trained_in.append(Path(cfg["world_model"]).parent.name)
        real.append(scores.mean())
        sem.append(scores.std(ddof=1) / len(scores) ** 0.5 if len(scores) > 1 else float("nan"))
        print(f"{name:18s} {len(eps):4d} {real[-1]:+7.2f} ± {sem[-1]:.2f} " + " ".join(f"{v:+18.2f}" for v in row),
              flush=True)
    s = Scores(names, trained_in, np.array(seen), np.array(counts), np.array(real), np.array(sem),
               {m: np.array(v) for m, v in predicted.items()})
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    write_scores(args.scores, s)
    print(f"\nscores: {args.scores}")
    report(s, None)


def agreement(args):
    report(read_scores(args.scores), args.csv)


def report(s: Scores, csv_path: Path | None):
    """Prints each world model's agreements() and exploitation gap, and
    writes the agreements to `csv_path` if given."""
    rows = agreements(s)
    print(f"\n{'world model':22s} {'policies':>9s} {'n':>3s} {'Pearson':>8s} {'Spearman':>9s} {'MMRV':>6s} "
          f"{'MAE':>6s} {'bias':>6s}")
    for m in s.predicted:
        for a in (a for a in rows if a.world_model == m):
            x = a.metrics
            print(f"{m:22s} {a.policies:>9s} {a.n:3d} {x['pearson']:8.2f} {x['spearman']:9.2f} "
                  f"{x['mmrv']:6.2f} {x['mae']:6.2f} {x['bias']:+6.2f}")
        gap = exploitation_gap(s, m)
        if gap is not None:
            print(f"{m:22s} exploitation gap (bias on own - bias on held out): {gap:+.2f}")
    if csv_path:
        write_agreements(csv_path, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="export the pool's policies for the harness")
    e.add_argument("--pool", type=Path, default=POOL)
    v = sub.add_parser("imagine", help="score the pool on the rig and in world models")
    v.add_argument("manifests", type=Path, nargs="+", help="TSVs collect.sh wrote: policy name, recording")
    v.add_argument("--world-models", nargs="+", required=True,
                   help="checkpoints, each PATH or PATH@TAU for a stochastic model's own tau")
    v.add_argument("--scores", type=Path, required=True, help="the CSV to write the scores to")
    v.add_argument("--pool", type=Path, default=POOL)
    v.add_argument("--seconds", type=float, default=10.0)
    v.add_argument("--rollouts", type=int, default=8, help="per episode start")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--tau", type=float, default=1.0,
                   help="the spread of stochastic world models' frames (0: their mean), where PATH@TAU does not say")
    v.add_argument("--device", default="auto")
    a = sub.add_parser("agreement", help="score world models' rankings from imagine's scores")
    a.add_argument("scores", type=Path)
    a.add_argument("--csv", type=Path, help="also write the table here")
    args = parser.parse_args()
    {"export": export, "imagine": imagine, "agreement": agreement}[args.command](args)


if __name__ == "__main__":
    main()
