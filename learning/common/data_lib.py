"""Loads collect's frames tables and cuts them into training windows.

A recording becomes one step per camera frame: the sine and cosine of each
tag's yaw, whether each tag was seen, and the action in effect after the
frame. Frames the camera dropped become steps with no tag seen and the
action carried over, so every step is one frame period (8 ms at 125 fps).

A window is `length` consecutive steps of one recording. Windows are drawn
uniformly from every start position in every recording, so a long
recording contributes in proportion to its length.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch

NUM_TAGS = 3
POSE_LEN = 7
# Actions are int8; dividing by this maps their whole range into [-1, 1),
# leaving room for policies that go beyond the random walk's ±60.
ACTION_SCALE = 128.0


@dataclass
class Recording:
    name: str
    obs: np.ndarray  # [T, NUM_TAGS, 2] float32: sin and cos of each tag's yaw
    present: np.ndarray  # [T, NUM_TAGS] bool: whether each tag was seen
    action: np.ndarray  # [T] float32: the action after each step, / ACTION_SCALE


def yaw(pose: np.ndarray) -> np.ndarray:
    """ZYX yaw of [..., 7] poses (x y z qw qx qy qz), in radians.

    The tag's in-plane angle, as in analysis/plot_yaw.py.
    """
    w, x, y, z = pose[..., 3], pose[..., 4], pose[..., 5], pose[..., 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def load_recording(path: Path) -> Recording:
    """Reads a frames.arrows table into one step per camera frame."""
    with ipc.open_stream(path) as reader:
        table = reader.read_all()
    frame = table["frame"].to_numpy()
    if np.any(np.diff(frame) <= 0):
        raise ValueError(f"{path}: frame numbers do not increase")
    slot = (frame - frame[0]).astype(np.int64)
    steps = int(slot[-1]) + 1

    obs = np.zeros((steps, NUM_TAGS, 2), np.float32)
    present = np.zeros((steps, NUM_TAGS), bool)
    for tag in range(NUM_TAGS):
        col = table[f"tag{tag}_pose"].combine_chunks()
        # A null pose still has its 7 slots in the child array, so the
        # child reshapes to one row per frame; the nulls read as NaN.
        values = col.values.slice(col.offset * POSE_LEN, len(col) * POSE_LEN)
        pose = values.to_numpy(zero_copy_only=False).reshape(-1, POSE_LEN)
        seen = col.is_valid().to_numpy(zero_copy_only=False)
        angle = yaw(pose[seen])
        obs[slot[seen], tag, 0] = np.sin(angle)
        obs[slot[seen], tag, 1] = np.cos(angle)
        present[slot[seen], tag] = True

    # A dropped frame never reached the policy: the action carries over.
    action = np.zeros(steps, np.float32)
    action[slot] = table["action"].to_numpy().astype(np.float32) / ACTION_SCALE
    has_row = np.zeros(steps, bool)
    has_row[slot] = True
    last_row = np.maximum.accumulate(np.where(has_row, np.arange(steps), 0))
    action = action[last_row]

    return Recording(path.parent.name, obs, present, action)


def load_recordings(data_dir: Path, names: list[str]) -> list[Recording]:
    """Loads `data_dir/<name>/frames.arrows` for each name."""
    return [load_recording(Path(data_dir) / n / "frames.arrows") for n in names]


@dataclass
class Batch:
    obs: torch.Tensor  # [B, L, NUM_TAGS, 2]
    present: torch.Tensor  # [B, L, NUM_TAGS] bool
    action: torch.Tensor  # [B, L]


class Windows:
    """Every window of `length` steps in a set of recordings, on one device."""

    def __init__(self, recordings: list[Recording], length: int, device):
        self.length = length
        self.obs = torch.from_numpy(np.concatenate([r.obs for r in recordings])).to(device)
        self.present = torch.from_numpy(
            np.concatenate([r.present for r in recordings])
        ).to(device)
        self.action = torch.from_numpy(
            np.concatenate([r.action for r in recordings])
        ).to(device)
        # Starts are indices into the concatenation; a window never spans
        # two recordings.
        starts, offset = [], 0
        for r in recordings:
            n = len(r.action) - length + 1
            if n > 0:
                starts.append(offset + np.arange(n))
            offset += len(r.action)
        if not starts:
            raise ValueError(f"no recording has {length} steps")
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self._steps = torch.arange(length, device=device)

    def __len__(self) -> int:
        return len(self.starts)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """`n` window indices, uniform and with replacement."""
        i = torch.randint(len(self), (n,), generator=generator, device="cpu")
        return i.to(self.starts.device)

    def gather(self, index: torch.Tensor) -> Batch:
        rows = self.starts[index][:, None] + self._steps
        return Batch(self.obs[rows], self.present[rows], self.action[rows])




def hanging_yaws(recordings: list[Recording], min_rest: int = 500) -> torch.Tensor:
    """The sin and cos [NUM_TAGS, 2] of each tag's yaw with the pendulum
    hanging still: the mean over the last second of every rest of at least
    `min_rest` frames at action 0 (4 s at 125 fps), when the pendulum has
    nearly settled. Averaging the sines and cosines keeps angles either
    side of 180° from cancelling.

    A tag hangs at 180° only if it is mounted square on its arm and the
    camera has no roll; these are what it reads instead.
    """
    sums = np.zeros((NUM_TAGS, 2))
    for r in recordings:
        edges = np.flatnonzero(np.diff(np.r_[0, (r.action == 0).astype(int), 0]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start >= min_rest:
                last = slice(end - 125, end)
                sums += (r.obs[last] * r.present[last][..., None]).sum(0)
    norm = np.linalg.norm(sums, axis=-1, keepdims=True)
    if (norm == 0).any():
        raise ValueError(f"no rest of {min_rest} frames with every tag seen")
    return torch.from_numpy((sums / norm).astype(np.float32))


def correct_yaws(obs, hanging):
    """obs [..., NUM_TAGS, 2] with each tag's yaw turned so that its
    hanging yaw [NUM_TAGS, 2] reads 180° and upright 0°: the pendulum's
    own angles, whatever the tags' mounting. Numpy or torch."""
    s, c = obs[..., 0], obs[..., 1]
    # Turning by 180° - hanging: sin and cos of that are sin h and -cos h.
    hs, hc = hanging[..., 0], -hanging[..., 1]
    stack = np.stack if isinstance(obs, np.ndarray) else torch.stack
    return stack([s * hc + c * hs, c * hc - s * hs], -1)
