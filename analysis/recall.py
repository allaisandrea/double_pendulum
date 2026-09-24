# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=15", "numpy>=1.26"]
# ///
"""Reports each tag's detection recall in one or more recordings, with its uncertainty.

    uv run analysis/recall.py recordings/1790029977 recordings/1790030112 ...

Recall is the fraction of observed frames in which a tag was detected; all
tags are assumed to stay in frame. It is measured while the stand-in policy
is active, since that is where motion blur costs detections; at rest it is
close to 100% whatever the camera settings, so rest periods would only
dilute the comparison. Rest recall is printed separately as a sanity check.

The uncertainty comes from splitting the active frames into --chunks
contiguous pieces of equal size and taking the spread of per-chunk recall.
Consecutive frames are strongly correlated (one fast swing blurs several
in a row), so a binomial error bar per frame would be far too optimistic;
chunks long compared with a swing are close to independent. Reported: the
mean over chunks ± its standard error (std / sqrt(chunks)), and the std.
Two runs differ meaningfully only when their means are further apart than
about 2 * sqrt(se1² + se2²).

Recall assumes the tags stay in frame, so each run is also checked for tags
detected within --edge-px of the image border: a tag seen that close may
leave the frame on other frames, which would count as misses. The corners
are recovered by projecting each pose back through the recorded intrinsics
and tag size, which undoes the pose fit exactly, even when both are wrong.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

# Settings that distinguish runs in a parameter sweep, printed per run, when
# the recording has them.
SETTINGS = ["exposure_us", "gain", "decimate", "policy_range", "policy_step",
            "policy_latency_ns", "plan_every_ns", "seed"]


def table_path(run):
    """frames.arrows; recordings before 2026-09-24 have observations.arrows."""
    new = run / "frames.arrows"
    return new if new.exists() else run / "observations.arrows"


def load(run):
    with ipc.open_stream(table_path(run)) as reader:
        obs = reader.read_all()
    meta = {k.decode(): v.decode() for k, v in obs.schema.metadata.items()}
    t = np.array(obs["t_capture"].cast("int64").to_pylist())
    tags = [int(i) for i in meta["tags"].strip("[]").split(",")]
    poses = {tag: np.array([p if p is not None else [np.nan] * 7
                            for p in obs[f"tag{tag}_pose"].to_pylist()])
             for tag in tags}
    return t, poses, meta


def corners_px(poses, meta):
    """Pixel corners (N, 4, 2) of each tag pose [x, y, z, qw, qx, qy, qz]."""
    k = dict(re.findall(r"(\w+)=([-\d.e]+)", meta["intrinsics"]))
    fx, fy, cx, cy = (float(k[n]) for n in ("fx", "fy", "cx", "cy"))
    h = float(meta["tag_size_m"]) / 2
    corners = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    t, (w, x, y, z) = poses[:, :3], poses[:, 3:].T
    r = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
    cam = np.einsum("nij,cj->nci", r, corners) + t[:, None, :]
    return np.stack([fx * cam[..., 0] / cam[..., 2] + cx,
                     fy * cam[..., 1] / cam[..., 2] + cy], -1)


def edge_distance(poses, meta):
    """Each detection's closest corner to the image border, in pixels; NaN if unseen."""
    width, height = (int(v) for v in re.search(r"(\d+)x(\d+)", meta["camera"]).groups())
    c = corners_px(poses, meta)
    d = np.minimum.reduce([c[..., 0], width - 1 - c[..., 0], c[..., 1], height - 1 - c[..., 1]])
    return d.min(axis=1)


def active_mask(t, meta):
    """True for frames captured while the policy was active. Recordings
    before 2026-09-24 give the cycle in slots of the old action grid."""
    if "active_ns" in meta:
        active, rest = int(meta["active_ns"]), int(meta["rest_ns"])
    else:
        period = int(meta["period_ns"])
        active = int(meta.get("active_slots", 1)) * period
        rest = int(meta.get("rest_slots", 0)) * period
    if not rest:
        return np.ones(len(t), dtype=bool)
    return t % (active + rest) < active


def chunk_stats(seen, chunks):
    """Mean, standard error and std of recall over contiguous equal chunks."""
    per_chunk = np.array([c.mean() for c in np.array_split(seen, chunks)])
    std = per_chunk.std(ddof=1)
    return per_chunk.mean(), std / np.sqrt(chunks), std


def settings(meta):
    out = []
    for k in SETTINGS:
        if k not in meta:
            continue
        v = meta[k]
        if k.endswith("_ns"):
            k, v = k[:-3] + "_ms", f"{int(v) / 1e6:g}"
        out.append(f"{k}={v}")
    return "  ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, nargs="+", help="recording directories")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--edge-px", type=float, default=20,
                    help="warn about tags detected this close to the image border")
    args = ap.parse_args()

    for run in args.runs:
        t, poses, meta = load(run)
        if len(t) == 0:
            print(f"{run}: no observations\n")
            continue
        seen = {tag: ~np.isnan(p[:, 0]) for tag, p in poses.items()}
        on = active_mask(t, meta)
        n_on = int(on.sum())
        duration = (t.max() - t.min()) / 1e9
        print(f"{run}  ({settings(meta)})")
        print(f"  {len(t)} frames over {duration:.0f} s: {n_on} active, "
              f"{len(t) - n_on} resting; {n_on // args.chunks} active frames per chunk")
        if n_on < 2 * args.chunks:
            print("  too few active frames to estimate recall\n")
            continue
        rows = {f"tag {tag}": s for tag, s in seen.items()}
        rows["all tags"] = np.logical_and.reduce(list(seen.values()))
        print(f"  {'':9} {'active recall':>22} {'chunk std':>10} {'rest':>7}")
        for name, s in rows.items():
            mean, se, std = chunk_stats(s[on], args.chunks)
            rest = f"{100 * s[~on].mean():6.1f}%" if (~on).any() else "     —"
            print(f"  {name:9} {100 * mean:12.1f}% ± {100 * se:4.1f}% {100 * std:9.1f}% {rest}")
        for tag, p in poses.items():
            d = edge_distance(p, meta)
            near = np.nan_to_num(d, nan=np.inf) < args.edge_px
            if near.any():
                print(f"  WARNING: tag {tag} detected within {args.edge_px:g} px of the border "
                      f"in {near.sum()} frames (closest {np.nanmin(d):.0f} px); "
                      "it may leave the frame, which counts as a miss")
        closest = {tag: np.nanmin(edge_distance(p, meta)) for tag, p in poses.items()
                   if (~np.isnan(p[:, 0])).any()}
        print("  closest to the border: "
              + ", ".join(f"tag {tag} {d:.0f} px" for tag, d in closest.items()))
        print()


if __name__ == "__main__":
    main()
