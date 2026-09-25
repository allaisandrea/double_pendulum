"""Draws the pendulum from its tags' yaws, for videos of real or imagined runs.

The pendulum is drawn as three equal arms chained from a fixed pivot, each
at its tag's yaw. The camera is upside down, so the drawing is its image
turned right side up: an arm at 180° points down. Given each tag's hanging
yaw (`hanging_yaws` in data_lib), the yaws are corrected so that a hanging
arm is drawn straight down, whatever the tag's mounting; without it, the
arms are drawn at the raw yaws.

An arm whose tag was not seen in a frame is drawn faded. Below the
pendulum, a bar shows the action, and a label the time. Videos play slowed
down, 4 times by default, so a swing is easy to follow.
"""
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.data_lib import FRAME_S, correct_yaws

# Reference palette (dataviz skill), as in analysis/plot_yaw.py.
ARMS = [(0x2A, 0x78, 0xD6), (0xEB, 0x68, 0x34), (0x1B, 0xAF, 0x7A)]
SURFACE, TEXT, GRID = (0xFC, 0xFC, 0xFB), (0x0B, 0x0B, 0x0B), (0xE4, 0xE3, 0xDF)
# Frames are 16-pixel multiples, which H.264 encodes without resizing.
SIZE = 512


def fill_unseen(obs: np.ndarray, present: np.ndarray) -> np.ndarray:
    """obs [T, NUM_TAGS, 2] with each unseen tag at its last seen value, for
    recordings, where an unseen tag has no angle. A tag not yet seen hangs
    straight down."""
    out = obs.copy()
    last = np.tile(np.array([0.0, -1.0], obs.dtype), (obs.shape[1], 1))
    for t in range(len(obs)):
        last = np.where(present[t][:, None], obs[t], last)
        out[t] = last
    return out


def _fade(color, amount=0.7):
    return tuple(int(c + (s - c) * amount) for c, s in zip(color, SURFACE))


def render_frame(
    obs, seen, action: float, t: float, slowdown: float = 4, size: int = SIZE
) -> np.ndarray:
    """One frame [size, size, 3] uint8 from obs [NUM_TAGS, 2] (sin and cos
    of each yaw), seen [NUM_TAGS], and the action as a fraction of 128."""
    img = Image.new("RGB", (size, size), SURFACE)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size // 28)
    pivot = np.array([size / 2, size * 0.46])
    arm = size * 0.13
    draw.ellipse([*(pivot - arm * 3.1), *(pivot + arm * 3.1)], outline=GRID)
    width = max(2, size // 48)
    joint = pivot
    for (s, c), was_seen, color in zip(obs, seen, ARMS):
        end = joint + arm * np.array([s, -c])
        draw.line([*joint, *end], fill=color if was_seen else _fade(color), width=width)
        draw.ellipse([*(joint - width), *(joint + width)], fill=TEXT)
        joint = end

    # The action bar grows from the middle, full width at ±128.
    y, half = size * 0.9, size * 0.4
    draw.line([size / 2 - half, y, size / 2 + half, y], fill=GRID, width=width)
    draw.line([size / 2, y, size / 2 + half * action, y], fill=TEXT, width=width * 2)
    label = size // 28
    draw.text((size / 2 - half, y - 2 * label), f"action {round(action * 128):+d}", fill=TEXT, font=font)
    draw.text((label, label), f"{t:5.2f} s", fill=TEXT, font=font)
    if slowdown != 1:
        draw.text((size - label, label), f"1/{slowdown:g} speed", fill=TEXT, font=font, anchor="ra")
    return np.asarray(img)


def render(
    obs, seen, action, stride: int = 1, slowdown: float = 4, size: int = SIZE, hanging=None
) -> np.ndarray:
    """Every `stride`-th frame [F, size, size, 3] of a run: obs [T, NUM_TAGS, 2],
    seen [T, NUM_TAGS], action [T] as fractions of 128, and optionally the
    hanging yaws [NUM_TAGS, 2] to correct obs by."""
    obs, seen, action = (np.asarray(a) for a in (obs, seen, action))
    if hanging is not None:
        obs = correct_yaws(obs, np.asarray(hanging))
    return np.stack(
        [
            render_frame(obs[t], seen[t], float(action[t]), t * FRAME_S, slowdown, size)
            for t in range(0, len(obs), stride)
        ]
    )


def write_video(path: Path, frames: np.ndarray, stride: int = 1, slowdown: float = 4):
    """An mp4 of frames rendered every `stride`-th frame, played `slowdown`
    times slower than real time: 31.25 fps by default."""
    iio.imwrite(path, frames, fps=1 / (FRAME_S * stride * slowdown), codec="libx264")
