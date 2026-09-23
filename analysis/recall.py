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
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

# Settings that distinguish runs in a parameter sweep, printed per run.
SETTINGS = ["exposure_us", "gain", "decimate", "plan_every_ns", "seed"]


def load(run):
    with ipc.open_stream(run / "observations.arrows") as reader:
        obs = reader.read_all()
    meta = {k.decode(): v.decode() for k, v in obs.schema.metadata.items()}
    t = np.array(obs["t_capture"].cast("int64").to_pylist())
    tags = [int(i) for i in meta["tags"].strip("[]").split(",")]
    seen = {tag: np.array([p is not None for p in obs[f"tag{tag}_pose"].to_pylist()])
            for tag in tags}
    return t, seen, meta


def active_mask(t, meta):
    """True for frames captured while the policy was active, by slot."""
    active, rest = int(meta.get("active_slots", 1)), int(meta.get("rest_slots", 0))
    if not rest:
        return np.ones(len(t), dtype=bool)
    slot = t // int(meta["period_ns"])
    return slot % (active + rest) < active


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
    args = ap.parse_args()

    for run in args.runs:
        t, seen, meta = load(run)
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
        print()


if __name__ == "__main__":
    main()
