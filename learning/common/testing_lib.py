"""Builds small frames tables for tests."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from common.data_lib import POSE_LEN, NUM_TAGS


def pose(angle):
    """A tag rotated by `angle` about the camera's viewing axis."""
    return [0.0, 0.0, 0.7, np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


def write_frames(path: Path, frames, poses, actions):
    """A minimal frames table: frame numbers, tag poses (None = unseen), actions."""
    pose_type = pa.list_(pa.field("item", pa.float32(), nullable=False), POSE_LEN)
    cols = {"frame": pa.array(frames, pa.uint64())}
    for tag in range(NUM_TAGS):
        cols[f"tag{tag}_pose"] = pa.array([p[tag] for p in poses], pose_type)
    cols["action"] = pa.array(actions, pa.int8())
    table = pa.table(cols)
    with ipc.new_stream(path, table.schema) as w:
        w.write_table(table)
