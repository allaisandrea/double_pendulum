# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=15", "numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Plots each tag's yaw over time, in randomly chosen windows of a recording.

    uv run analysis/plot_yaw.py recordings/1790029977
    uv run analysis/plot_yaw.py recordings/1790029977 --windows 5 --seconds 10 --seed 1

Yaw is the tag's in-plane angle: its rotation about the camera's viewing
axis (the Z of a ZYX decomposition of the tag-to-camera rotation). For the
pendulum it is each link's swing angle, and the part of the pose that stays
trustworthy; the out-of-plane angles flip between two ambiguous solutions.

One figure per window, one axes per tag. Angles are unwrapped within a
window, so a full turn reads as a climb rather than a jump, and start
inside -180..180. Periods where the stand-in policy rested are shaded.
Figures go next to this script, named after the run and window start.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.ipc as ipc

# Reference palette (dataviz skill): categorical slots 1-3, light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT_2, GRID, REST = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
# Break a line where samples are further apart than this, so a stretch
# where the tag went unseen shows as a gap rather than a straight segment.
MAX_GAP_S = 0.25


def yaw_deg(pose):
    """ZYX yaw of a [x, y, z, qw, qx, qy, qz] pose, in degrees."""
    w, x, y, z = pose[3], pose[4], pose[5], pose[6]
    return np.degrees(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def load(run):
    with ipc.open_stream(run / "observations.arrows") as reader:
        obs = reader.read_all()
    meta = {k.decode(): v.decode() for k, v in obs.schema.metadata.items()}
    t = np.array(obs["t_capture"].cast("int64").to_pylist()) / 1e9
    tags = [int(i) for i in meta["tags"].strip("[]").split(",")]
    yaws = {}
    for tag in tags:
        poses = obs[f"tag{tag}_pose"].to_pylist()
        yaws[tag] = np.array([yaw_deg(p) if p is not None else np.nan for p in poses])
    return t, yaws, meta


def rest_spans(meta, lo, hi):
    """(start, end) seconds of the policy's rest periods within [lo, hi]."""
    active, rest = int(meta.get("active_slots", 1)), int(meta.get("rest_slots", 0))
    if not rest:
        return []
    period = int(meta["period_ns"]) / 1e9
    cycle = (active + rest) * period
    spans, start = [], np.floor(lo / cycle) * cycle
    while start < hi:
        a, b = start + active * period, start + cycle
        if b > lo and a < hi:
            spans.append((max(a, lo), min(b, hi)))
        start += cycle
    return spans


def window_series(t, y, lo, hi):
    """The samples in [lo, hi], unwrapped, with NaN at gaps to break the line."""
    keep = (t >= lo) & (t <= hi) & ~np.isnan(y)
    tw, yw = t[keep], y[keep]
    if len(yw) == 0:
        return tw, yw
    yw = np.degrees(np.unwrap(np.radians(yw)))
    yw -= 360 * np.round(yw[0] / 360)  # start inside -180..180
    breaks = np.where(np.diff(tw) > MAX_GAP_S)[0] + 1
    return np.insert(tw, breaks, np.nan), np.insert(yw, breaks, np.nan)


def plot_window(run, t, yaws, meta, lo, hi, out):
    tags = list(yaws)
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 2.2 * len(tags)), sharex=True,
                             facecolor=SURFACE)
    for ax, tag, color in zip(axes, tags, SERIES):
        ax.set_facecolor(SURFACE)
        for a, b in rest_spans(meta, lo, hi):
            ax.axvspan(a, b, color=REST, lw=0, zorder=0)
        tw, yw = window_series(t, yaws[tag], lo, hi)
        ax.plot(tw, yw, color=color, lw=1.5, zorder=2)
        seen = np.mean(~np.isnan(yaws[tag][(t >= lo) & (t <= hi)])) * 100
        ax.set_ylabel(f"tag {tag}\nyaw (°)", color=TEXT, fontsize=10)
        ax.text(1.0, 1.02, f"seen in {seen:.0f}% of frames", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=TEXT_2)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=9)
    axes[-1].set_xlim(lo, hi)
    axes[-1].set_xlabel("time since t0 (s)", color=TEXT, fontsize=10)
    note = "  ·  grey: policy resting (zero duty)" if rest_spans(meta, lo, hi) else ""
    fig.suptitle(f"Tag yaw, {run.name}, {lo:.1f}–{hi:.1f} s{note}", color=TEXT,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="recording directory")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=None, help="default: random, printed")
    args = ap.parse_args()

    t, yaws, meta = load(args.run)
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1 << 31))
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(t.min(), t.max() - args.seconds, args.windows))
    print(f"{args.run}: {len(t)} observations over {t.max() - t.min():.0f} s; seed {seed}")
    here = Path(__file__).parent
    for lo in starts:
        out = here / f"yaw_{args.run.name}_{lo:07.1f}s.png"
        plot_window(args.run, t, yaws, meta, lo, lo + args.seconds, out)
        print(f"  {out}")


if __name__ == "__main__":
    main()
