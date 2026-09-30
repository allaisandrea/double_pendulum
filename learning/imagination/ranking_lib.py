"""Ranks policies on the rig and in world models, and scores how well each
world model's ranking matches the rig's, after SIMPLER (arXiv:2405.05941).

A ranking recording is one policy run greedily on a duty cycle: driving
for `active_s`, resting for `rest_s`, over and over. Each drive after the
first is an episode that starts from the pendulum after a rest. Its real
score is the mean sum of the arms' cosines over the drive's first
`seconds`. Its imagined score comes from rollouts started from the same
recorded window, the frames just before the drive, so that the world
model sees exactly the state the rig was in, arm still swinging and all.
The first drive is left out: it starts before the policy has a history.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch

from common.data_lib import FRAME_S, Batch, correct_yaws, load_recording
from common.render_lib import fill_unseen
from imagination.rollout_lib import closed_loop


@dataclass
class Episode:
    real: float  # mean sum of cosines over the drive's first seconds
    start: Batch  # the `history` recorded steps before the drive, a batch of one


def sampled(recording: Path) -> bool:
    """Whether the recording's policy sampled its actions (collect
    --policy-sample), so that imagination should too."""
    with ipc.open_stream(recording) as f:
        meta = f.schema.metadata or {}
    return meta.get(b"policy_sample", b"false") == b"true"


def duty(recording: Path) -> tuple[float, float]:
    """The recording's duty cycle, seconds driving and resting, from the
    metadata collect writes; 10 and 5 for a recording without it."""
    with ipc.open_stream(recording) as f:
        meta = f.schema.metadata or {}
    if b"active_ns" not in meta:
        return 10.0, 5.0
    return int(meta[b"active_ns"]) / 1e9, int(meta[b"rest_ns"]) / 1e9


def capture_times(recording: Path, steps: int) -> np.ndarray:
    """Seconds since t0 of each step of load_recording's layout (one per
    camera frame, dropped frames interpolated)."""
    with ipc.open_stream(recording) as f:
        table = f.read_all()
    capture = table["t_capture"].cast("int64").to_numpy() / 1e9
    frame = table["frame"].to_numpy()
    return np.interp(np.arange(steps), frame - frame[0], capture)


def episodes(recording: Path, hanging: np.ndarray, history: int, seconds: float | None = None) -> list[Episode]:
    """The episodes of a ranking recording, as the module docstring says: its
    drives after the first, on its own duty cycle, each scored over its
    first `seconds` (default: the whole drive)."""
    active_s, rest_s = duty(recording)
    seconds = active_s if seconds is None else seconds
    r = load_recording(recording)
    at = capture_times(recording, len(r.action))
    corrected = correct_yaws(fill_unseen(r.obs, r.present), hanging)
    cycle, out = active_s + rest_s, []
    k = 1
    while k * cycle + seconds <= at[-1]:
        begin = int(np.searchsorted(at, k * cycle))
        end = int(np.searchsorted(at, k * cycle + seconds))
        if begin >= history:
            window = slice(begin - history, begin)
            out.append(Episode(
                real=float(corrected[begin:end][..., 1].sum(-1).mean()),
                start=Batch(torch.from_numpy(r.obs[window][None]),
                            torch.from_numpy(r.present[window][None]),
                            torch.from_numpy(r.action[window][None])),
            ))
        k += 1
    return out


@torch.no_grad()
def imagined(agent, env, starts: list[Batch], seconds: float, rollouts: int, seed: int,
             greedy: bool = True) -> float:
    """The mean sum of cosines over `seconds` of rollouts from each start,
    `rollouts` each, greedy or sampling the policy, their tag misses drawn
    with `seed` (sampled actions from torch's generator)."""
    device = env.hanging.device
    batch = Batch(*(torch.cat([getattr(s, k) for s in starts]).repeat_interleave(rollouts, 0).to(device)
                    for k in ("obs", "present", "action")))
    env.reset_to(Batch(*(t[:, -env.history:] for t in (batch.obs, batch.present, batch.action))))
    env.generator = torch.Generator().manual_seed(seed)
    try:
        run = closed_loop(agent, env, round(seconds / FRAME_S), greedy)
    finally:
        env.generator = None
    return correct_yaws(run.prediction, env.hanging)[..., 1].sum(-1).mean().item()


def mmrv(real: np.ndarray, predicted: np.ndarray) -> float:
    """Mean maximum rank violation (SIMPLER): for each policy, the largest
    real gap to a policy the prediction orders the other way round,
    averaged. 0 = the same order; it counts a swap of near-equals little
    and one of far-apart policies much."""
    n = len(real)
    worst = [
        max((abs(real[i] - real[j]) for j in range(n)
             if (predicted[i] < predicted[j]) != (real[i] < real[j])), default=0.0)
        for i in range(n)
    ]
    return float(np.mean(worst))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def agreement(real: np.ndarray, predicted: np.ndarray) -> dict:
    """How well predicted scores match real ones, over policies; bias is
    the mean overrating (predicted - real)."""
    return {
        "pearson": float(np.corrcoef(real, predicted)[0, 1]),
        "spearman": spearman(real, predicted),
        "mmrv": mmrv(real, predicted),
        "mae": float(np.abs(predicted - real).mean()),
        "bias": float((predicted - real).mean()),
    }


def subsets(model: str, seen: np.ndarray, trained_in: list[str]) -> dict[str, np.ndarray]:
    """The policies to score `model` over, as masks: all; new, those
    whose data is in no world model's training set; and, when some policy
    was trained in `model`, held_out, the others, and own, those. A world
    model is biased towards its own policies, which PPO has tuned to its
    errors: held_out measures it as an independent judge, own as the
    forecaster of what is trained in it."""
    own = np.array([t == model for t in trained_in])
    out = {"all": np.ones_like(own), "new": ~seen}
    if own.any():
        out |= {"held_out": ~own, "own": own}
    return out


@dataclass
class Scores:
    """Each policy's rig score and each world model's prediction of it, as
    `imagination.ranking imagine` writes them."""
    policy: list[str]
    trained_in: list[str]  # the world model each policy was trained in
    seen: np.ndarray  # [policies] bool: its rig data is in the world models' training sets
    episodes: np.ndarray  # [policies] int
    rig: np.ndarray  # [policies]
    rig_sem: np.ndarray  # [policies]
    predicted: dict[str, np.ndarray]  # world model: [policies]


FIXED = ["policy", "trained_in", "seen", "episodes", "rig", "rig_sem"]


def write_scores(path: Path, s: Scores):
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow([*FIXED, *s.predicted])
        for i, name in enumerate(s.policy):
            out.writerow([name, s.trained_in[i], bool(s.seen[i]), int(s.episodes[i]), round(float(s.rig[i]), 4),
                          round(float(s.rig_sem[i]), 4), *(round(float(p[i]), 4) for p in s.predicted.values())])


def read_scores(path: Path) -> Scores:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    column = lambda k, t=float: np.array([t(r[k]) for r in rows])
    models = [k for k in rows[0] if k not in FIXED]
    return Scores([r["policy"] for r in rows], [r["trained_in"] for r in rows], column("seen", lambda v: v == "True"),
                  column("episodes", int), column("rig"), column("rig_sem"), {m: column(m) for m in models})


@dataclass
class Agreement:
    world_model: str
    policies: str  # the subset, as subsets() names it
    n: int
    metrics: dict  # agreement()'s


def agreements(s: Scores) -> list[Agreement]:
    """agreement() of each world model over each of its subsets() of at
    least 3 policies."""
    out = []
    for m, p in s.predicted.items():
        for label, keep in subsets(m, s.seen, s.trained_in).items():
            if keep.sum() >= 3:
                out.append(Agreement(m, label, int(keep.sum()), agreement(s.rig[keep], p[keep])))
    return out


def exploitation_gap(s: Scores, model: str) -> float | None:
    """How much more `model` overrates the policies trained in it than the
    others, or None if none was."""
    masks = subsets(model, s.seen, s.trained_in)
    if "own" not in masks:
        return None
    over = s.predicted[model] - s.rig
    return float(over[masks["own"]].mean() - over[masks["held_out"]].mean())


def write_agreements(path: Path, rows: list[Agreement]):
    keys = ["pearson", "spearman", "mmrv", "mae", "bias"]
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["world_model", "policies", "n", *keys])
        for a in rows:
            out.writerow([a.world_model, a.policies, a.n, *(round(a.metrics[k], 4) for k in keys)])
