"""Loads collect's frames tables and cuts them into training windows.

A recording becomes one step per camera frame: the sine and cosine of each
tag's yaw, whether each tag was seen, and the action in effect after the
frame. Frames the camera dropped become steps with no tag seen and the
action carried over, so every step is one frame period (8 ms at 125 fps).

The observation is one of OBSERVATIONS: "yaw", each tag's in-plane angle
in the camera's image, or "angles", each link's angle from hanging, as
common.calibrate computes them from the tags' poses (the rig's geometry
and the camera's pose calibrated away) and stores them beside the frames
in angles.arrows. Either way it is a (sin, cos) per tag, seen where the
tag was, so everything downstream works on both.

A window is `length` consecutive steps of one recording. Windows are drawn
uniformly from every start position in every recording, so a long
recording contributes in proportion to its length.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

NUM_TAGS = 3
OBSERVATIONS = ("yaw", "angles")
ANGLES_FILE = "angles.arrows"
# Seconds between camera frames (125 fps).
FRAME_S = 0.008
POSE_LEN = 7
# Actions are int8; dividing by this maps their whole range into [-1, 1),
# leaving room for policies that go beyond the random walk's ±60.
ACTION_SCALE = 128.0


@dataclass
class Recording:
    name: str
    obs: np.ndarray  # [T, NUM_TAGS, 2] float32: sin and cos of each tag's yaw, or its link's angle
    present: np.ndarray  # [T, NUM_TAGS] bool: whether each tag was seen
    action: np.ndarray  # [T] float32: the action after each step, / ACTION_SCALE


def yaw(pose: np.ndarray) -> np.ndarray:
    """ZYX yaw of [..., 7] poses (x y z qw qx qy qz), in radians.

    The tag's in-plane angle, as in analysis/plot_yaw.py.
    """
    w, x, y, z = pose[..., 3], pose[..., 4], pose[..., 5], pose[..., 6]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def load_recording(path: Path, observation: str = "yaw") -> Recording:
    """Reads a frames.arrows table into one step per camera frame, observed
    as `observation` (see the module docstring)."""
    if observation not in OBSERVATIONS:
        raise ValueError(f"observation {observation!r}: not one of {OBSERVATIONS}")
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

    if observation == "angles":
        obs, present = read_angles(path.parent / ANGLES_FILE, steps)
    return Recording(path.parent.name, obs, present, action)


def load_recordings(data_dir: Path, names: list[str], observation: str = "yaw") -> list[Recording]:
    """Loads `data_dir/<name>/frames.arrows` for each name."""
    return [load_recording(Path(data_dir) / n / "frames.arrows", observation) for n in names]


def write_angles(path: Path, angles: np.ndarray, metadata: dict[str, str]):
    """Writes each step's link angles [T, NUM_TAGS] (radians from hanging,
    NaN where the link's tag was unseen) as angles.arrows, one row per step
    of load_recording's layout, with `metadata` (how they were calibrated)."""
    cols = {f"link{i}": pa.array(angles[:, i], pa.float64(), mask=np.isnan(angles[:, i])) for i in range(NUM_TAGS)}
    table = pa.table(cols).replace_schema_metadata(metadata)
    with ipc.new_stream(path, table.schema) as w:
        w.write_table(table)


def read_angles(path: Path, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """The (sin, cos) [steps, NUM_TAGS, 2] and seen [steps, NUM_TAGS] of an
    angles.arrows file written for a recording of `steps` steps."""
    if not path.exists():
        raise FileNotFoundError(f"{path}: no calibrated angles; run common.calibrate")
    with ipc.open_stream(path) as reader:
        table = reader.read_all()
    if len(table) != steps:
        raise ValueError(f"{path}: {len(table)} steps, the recording {steps}; calibrate it again")
    angles = np.stack([table[f"link{i}"].to_numpy(zero_copy_only=False) for i in range(NUM_TAGS)], -1)
    present = ~np.isnan(angles)
    angles = np.nan_to_num(angles)
    obs = np.stack([np.sin(angles), np.cos(angles)], -1).astype(np.float32) * present[..., None]
    return obs, present


def angles_metadata(path: Path) -> dict[str, str]:
    """The metadata an angles.arrows file was written with."""
    with ipc.open_stream(path) as reader:
        return {k.decode(): v.decode() for k, v in (reader.schema.metadata or {}).items()}


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
        # two recordings. `local` is each start's step within its recording.
        starts, local, offset = [], [], 0
        for r in recordings:
            n = len(r.action) - length + 1
            if n > 0:
                starts.append(offset + np.arange(n))
                local.append(np.arange(n))
            offset += len(r.action)
        if not starts:
            raise ValueError(f"no recording has {length} steps")
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.local = torch.from_numpy(np.concatenate(local)).to(device)
        self._steps = torch.arange(length, device=device)

    def __len__(self) -> int:
        return len(self.starts)

    def fold_of(self, index: torch.Tensor, block: int) -> torch.Tensor:
        """The fold, 0 or 1, of the windows at `index`: their recordings cut
        into blocks of `block` steps, alternately fold 0 and fold 1, by where
        each window starts. Blocks keep the overlapping windows of one
        stretch of time together, so one fold says little about the other."""
        return (self.local[index] // block) % 2

    def keep_fold(self, block: int, fold: int):
        """Keeps only the windows starting in `fold`'s blocks."""
        keep = self.fold_of(torch.arange(len(self.starts), device=self.starts.device), block) == fold
        self.starts, self.local = self.starts[keep], self.local[keep]

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
