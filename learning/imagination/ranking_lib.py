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


def capture_times(recording: Path, steps: int) -> np.ndarray:
    """Seconds since t0 of each step of load_recording's layout (one per
    camera frame, dropped frames interpolated)."""
    with ipc.open_stream(recording) as f:
        table = f.read_all()
    capture = table["t_capture"].cast("int64").to_numpy() / 1e9
    frame = table["frame"].to_numpy()
    return np.interp(np.arange(steps), frame - frame[0], capture)


def episodes(recording: Path, hanging: np.ndarray, history: int, active_s: float = 10.0,
             rest_s: float = 5.0, seconds: float = 10.0) -> list[Episode]:
    """The episodes of a ranking recording, as the module docstring says."""
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
    """How well predicted scores match real ones, over policies."""
    return {
        "pearson": float(np.corrcoef(real, predicted)[0, 1]),
        "spearman": spearman(real, predicted),
        "mmrv": mmrv(real, predicted),
        "mae": float(np.abs(predicted - real).mean()),
    }
