"""Ranks policies on the rig and in world models, and scores the world models' rankings.

    uv run python -m imagination.ranking export
    uv run python -m imagination.ranking evaluate ../recordings/ranking-*-seed0.tsv ../recordings/ranking-*-seed1.tsv \
        --world-models runs/world_model/rollout3-k8/model.pt runs/world_model/rollout-k16/model.pt

export writes each policy in the pool (imagination/ranking/pool.txt) as
the harness's JSON, to runs/ranking/<name>.json, for
imagination/ranking/collect.sh to run on the rig. evaluate reads the
manifests that script writes (policy name, recording), pools each
policy's episodes across all its recordings, scores it on the rig over
them (imagination.ranking_lib), and
in each world model from the same starts, greedy or sampling as the
recording's policy did, then prints, per world model,
how well its scores agree with the rig's: Pearson and Spearman
correlation, mean maximum rank violation (MMRV) and mean absolute error,
over all policies and over those new to the rig.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common.data_lib import hanging_yaws, load_recordings
from common.run_lib import pick_device
from imagination.export_lib import export_policy
from imagination.agent_lib import Agent
from imagination.ranking_lib import agreement, episodes, imagined, sampled
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


def evaluate(args):
    device = pick_device(args.device)
    pool = read_pool(args.pool)
    hanging = hanging_yaws(load_recordings(Path("data"), HANGING)).numpy()
    recordings: dict[str, list[str]] = {}
    for manifest in args.manifests:
        for line in manifest.read_text().splitlines():
            if line.strip():
                name, recording = line.split("\t")
                recordings.setdefault(name, []).append(recording)
    models = {p.parent.name: load_world_model(p, device) for p in args.world_models}
    names, seen, real, sem, predicted = [], [], [], [], {m: [] for m in models}
    print(f"{'policy':18s} {'eps':>4s} {'rig':>14s} " + " ".join(f"{m:>18s}" for m in models))
    for name, paths in recordings.items():
        checkpoint, was_seen = pool[name]
        agent, env, _ = load_policy(checkpoint, device)
        groups = []  # (episodes, greedy) per recording
        for recording in paths:
            frames = Path(recording) / "frames.arrows"
            groups.append((episodes(frames, hanging, env.history, seconds=args.seconds), not sampled(frames)))
        eps = [e for group, _ in groups for e in group]
        scores = np.array([e.real for e in eps])
        row = []
        for m, model in models.items():
            env.model = model
            # Each recording's starts, weighted by how many episodes it has.
            parts = [(imagined(agent, env, [e.start for e in group], args.seconds, args.rollouts, args.seed, greedy), len(group))
                     for group, greedy in groups if group]
            v = sum(value * n for value, n in parts) / sum(n for _, n in parts)
            predicted[m].append(v)
            row.append(v)
        names.append(name)
        seen.append(was_seen)
        real.append(scores.mean())
        sem.append(scores.std(ddof=1) / len(scores) ** 0.5 if len(scores) > 1 else float("nan"))
        print(f"{name:18s} {len(eps):4d} {real[-1]:+7.2f} ± {sem[-1]:.2f} " + " ".join(f"{v:+18.2f}" for v in row),
              flush=True)
    real, new = np.array(real), ~np.array(seen)
    print(f"\n{'world model':22s} {'policies':>9s} {'Pearson':>8s} {'Spearman':>9s} {'MMRV':>6s} {'MAE':>6s}")
    for m in models:
        p = np.array(predicted[m])
        for label, keep in (("all", np.ones_like(new)), ("new to the rig", new)):
            if keep.sum() >= 3:
                a = agreement(real[keep], p[keep])
                print(f"{m:22s} {label:>9s} {a['pearson']:8.2f} {a['spearman']:9.2f} {a['mmrv']:6.2f} {a['mae']:6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="export the pool's policies for the harness")
    e.add_argument("--pool", type=Path, default=POOL)
    v = sub.add_parser("evaluate", help="score world models' rankings against a rig session")
    v.add_argument("manifests", type=Path, nargs="+", help="TSVs collect.sh wrote: policy name, recording")
    v.add_argument("--world-models", type=Path, nargs="+", required=True)
    v.add_argument("--pool", type=Path, default=POOL)
    v.add_argument("--seconds", type=float, default=10.0)
    v.add_argument("--rollouts", type=int, default=8, help="per episode start")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--device", default="auto")
    args = parser.parse_args()
    {"export": export, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
