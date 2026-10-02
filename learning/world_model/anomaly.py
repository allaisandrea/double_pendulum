"""Looks for anomalies in the recordings: stretches where something on the rig
went wrong (a motor without power, a tag that slipped, a hand on the arm).

    uv run python -m world_model.anomaly runs/world_model/rb2-it2/model.pt --out runs/anomaly

Three checks, each written to --out as a CSV, with a figure per recording
that has flagged stretches:

- surprise (new era only): each frame's one-step negative log-likelihood
  under the world model, per tag, given the 16 recorded frames before it,
  and the cross-entropy of its missing-tag predictions. Each is turned into
  a robust z-score against frames in the same state (driving or resting,
  how high the pendulum is, how fast the motor arm turns), so that hard
  but ordinary physics does not count. A stretch is flagged where the
  rolling median of a z-score over --smooth-s stays above --z for at least
  --min-s seconds.
- motor response: the motor arm's median speed over each drive's frames
  with a large action (|action| >= 40); under 0.2 rev/s means the motor did
  not turn the arm.
- hanging pose: each tag's angle over the last second of each rest of at
  least 4 s; a tag that moves on its arm shows as a step.

The model's training recordings are marked in-sample: it fits them better,
so their surprise reads low. The baselines are taken from the other
recordings. --extra adds recordings from ../recordings that are not in
data/ (the scrapped ones, as known faults).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from common.data_lib import ACTION_SCALE, NUM_TAGS, Recording, load_recording
from common.run_lib import pick_device
from imagination.ranking_lib import capture_times, duty
from world_model.model_lib import gaussian_nll, last_seen, load_world_model

FPS = 125
FIRST = 1790789841  # the first recording after a tag on the rig moved
SIGNALS = [f"nll_tag{i}" for i in range(NUM_TAGS)] + ["bce"]


def recording_path(name: str) -> Path:
    local = Path("data") / name / "frames.arrows"
    return local if local.exists() else Path("../recordings") / name / "frames.arrows"


@torch.no_grad()
def surprise(model, r: Recording, device, batch: int = 32768) -> np.ndarray:
    """[T, 4]: each frame's NLL per tag (summed over sin and cos; NaN where
    the tag is unseen or has no reference) and the missing-tag cross-entropy
    summed over tags, NaN for the first `window` frames."""
    w = model.window
    obs = torch.from_numpy(r.obs).to(device)
    present = torch.from_numpy(r.present).to(device)
    action = torch.from_numpy(r.action).to(device)
    out = np.full((len(r.action), len(SIGNALS)), np.nan, np.float32)
    for start in range(w, len(r.action), batch):
        t = torch.arange(start, min(start + batch, len(r.action)), device=device)
        idx = t[:, None] - w + torch.arange(w, device=device)[None]  # [B, w]
        o, p, a = obs[idx], present[idx], action[idx]
        delta, logit, logvar = model(o, p, a)
        ref, has_ref = last_seen(o, p)
        target = (obs[t] - ref) / model.delta_scale
        nll = gaussian_nll(delta - target, logvar).sum(-1)  # [B, NUM_TAGS]
        nll = torch.where(present[t] & has_ref, nll, torch.nan)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, (~present[t]).float(), reduction="none").sum(-1)
        out[t.cpu().numpy(), :NUM_TAGS] = nll.float().cpu().numpy()
        out[t.cpu().numpy(), NUM_TAGS] = bce.float().cpu().numpy()
    return out


def angles(r: Recording) -> np.ndarray:
    """Each tag's yaw [T, NUM_TAGS], NaN where unseen."""
    a = np.arctan2(r.obs[..., 0], r.obs[..., 1])
    return np.where(r.present, a, np.nan)


def arm0_speed(r: Recording) -> np.ndarray:
    """The motor arm's speed [T] in rev/s, NaN where not measurable."""
    a = angles(r)[:, 0]
    s = np.abs(np.angle(np.exp(1j * np.diff(a)))) * FPS / (2 * np.pi)
    return np.r_[np.nan, s]


def phases(path: Path, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Whether each step is in a drive, its drive's index (-1 resting), and
    the cycle length in seconds."""
    active, rest = duty(path)
    t = capture_times(path, steps)
    cycle = active + rest
    driving = (t % cycle) < active
    return driving, np.where(driving, (t // cycle).astype(int), -1), cycle


def state_bins(r: Recording, hanging: np.ndarray, driving: np.ndarray) -> np.ndarray:
    """A coarse state per frame: driving or resting, how high the pendulum is
    (the sum of the arms' cosines from hanging, in 4 bins), and how fast the
    motor arm turns (3 bins)."""
    a = angles(r)
    height = np.nansum(-np.cos(a - hanging[None]), axis=1)
    h = np.digitize(height, [-2.0, 0.0, 2.0])
    s = np.digitize(np.nan_to_num(arm0_speed(r)), [0.3, 1.5])
    return driving.astype(int) * 12 + h * 3 + s


def rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    """Median over a centred window of n steps, ignoring NaNs, on a grid of
    n // 5 steps and held between (exact enough to find long stretches)."""
    step = max(1, n // 5)
    out = np.full(len(x), np.nan)
    for c in range(0, len(x), step):
        win = x[max(0, c - n // 2): c + n // 2]
        if np.isfinite(win).sum() > n // 4:
            out[c: c + step] = np.nanmedian(win)
    return out


def stretches(flag: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
    return [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_len]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("world_model", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/anomaly"))
    parser.add_argument("--extra", nargs="*", default=["1790875867", "1790877098"])
    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--smooth-s", type=float, default=5.0)
    parser.add_argument("--min-s", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    ck = torch.load(args.world_model, map_location="cpu", weights_only=False)
    in_sample = set(ck["config"]["train"])
    model = load_world_model(args.world_model, device)

    names = sorted(p.parent.name for p in Path("data").glob("*/frames.arrows"))
    names += [n for n in args.extra if n not in names]
    new = [n for n in names if int(n) >= FIRST]

    # Hanging pose and motor response, both eras.
    rests, drives, recs = [], [], {}
    for n in names:
        path = recording_path(n)
        r = load_recording(path)
        driving, drive, _ = phases(path, len(r.action))
        a, speed = angles(r), arm0_speed(r)
        edges = np.flatnonzero(np.diff(np.r_[0, (r.action == 0).astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            # A rest that runs to the end is the stop, the arm still swinging.
            if e - s >= 4 * FPS and e < len(r.action):
                last = a[e - FPS: e]
                mean = np.degrees(np.angle(np.nanmean(np.exp(1j * last), axis=0)))
                rests.append({"recording": n, "t_s": e / FPS, **{f"tag{i}_deg": mean[i] for i in range(NUM_TAGS)}})
        big = np.abs(r.action * ACTION_SCALE) >= 40
        for d in np.unique(drive[drive >= 0]):
            m = (drive == d) & big & np.isfinite(speed)
            if m.sum() >= FPS:
                drives.append({"recording": n, "t_s": np.flatnonzero(drive == d)[0] / FPS,
                               "arm0_rev_s": float(np.median(speed[m])), "frames": int(m.sum())})
        if n in new:
            recs[n] = (r, path, driving)
    write(args.out / "rests.csv", rests)
    write(args.out / "drives.csv", drives)

    # Surprise, new era only.
    # Each tag's typical hanging angle in the new era, a circular mean.
    new_rests = np.radians([[x[f"tag{i}_deg"] for i in range(NUM_TAGS)] for x in rests if int(x["recording"]) >= FIRST])
    hanging = np.angle(np.nanmean(np.exp(1j * new_rests), axis=0))
    scores, bins = {}, {}
    for n, (r, path, driving) in recs.items():
        scores[n] = surprise(model, r, device)
        bins[n] = state_bins(r, hanging, driving)
        print(f"scored {n}", flush=True)
    base_names = [n for n in recs if n not in in_sample and n not in args.extra]
    med = np.full((36, len(SIGNALS)), np.nan)
    mad = np.full((36, len(SIGNALS)), np.nan)
    for b in range(36):
        pooled = np.concatenate([scores[n][bins[n] == b] for n in base_names])
        if len(pooled) > 1000:
            med[b] = np.nanmedian(pooled, axis=0)
            mad[b] = 1.4826 * np.nanmedian(np.abs(pooled - med[b]), axis=0)

    flagged, summary = [], []
    for n, (r, path, driving) in recs.items():
        z = (scores[n] - med[bins[n]]) / mad[bins[n]]
        smooth = np.stack([rolling_median(z[:, k], int(args.smooth_s * FPS)) for k in range(len(SIGNALS))], 1)
        rec_flags = []
        for k, sig in enumerate(SIGNALS):
            for s, e in stretches(np.nan_to_num(smooth[:, k]) > args.z, int(args.min_s * FPS)):
                rec_flags.append({"recording": n, "signal": sig, "start_s": round(s / FPS, 1),
                                  "length_s": round((e - s) / FPS, 1), "peak_z": round(float(np.nanmax(smooth[s:e, k])), 1),
                                  "in_sample": n in in_sample, "known_fault": n in args.extra})
        flagged += rec_flags
        summary.append({"recording": n, "minutes": round(len(r.action) / FPS / 60, 1), "in_sample": n in in_sample,
                        **{f"median_z_{sig}": round(float(np.nanmedian(z[:, k])), 2) for k, sig in enumerate(SIGNALS)},
                        **{f"max_smooth_z_{sig}": round(float(np.nanmax(smooth[:, k])), 1) for k, sig in enumerate(SIGNALS)},
                        "flagged_s": round(sum(f["length_s"] for f in rec_flags), 1)})
        if rec_flags:
            plot(args.out / f"{n}.png", n, smooth, driving, rec_flags, args.z)
    write(args.out / "flagged.csv", flagged)
    write(args.out / "recordings.csv", summary)
    print(f"{len(flagged)} flagged stretches in {len({f['recording'] for f in flagged})} recordings; see {args.out}")


def write(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        if rows:
            out = csv.DictWriter(f, fieldnames=list(rows[0]))
            out.writeheader()
            out.writerows(rows)


def plot(path, name, smooth, driving, flags, z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(smooth)) / FPS / 60
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(12, 7), sharex=True)
    for k, (ax, sig) in enumerate(zip(axes, SIGNALS)):
        ax.fill_between(t, -5, 15, where=driving, color="0.92", step="mid", linewidth=0)
        ax.plot(t, smooth[:, k], linewidth=0.8)
        ax.axhline(z, color="tab:red", linewidth=0.6, linestyle="--")
        for f in flags:
            if f["signal"] == sig:
                ax.axvspan(f["start_s"] / 60, (f["start_s"] + f["length_s"]) / 60, color="tab:red", alpha=0.2)
        ax.set_ylim(-3, 12)
        ax.set_ylabel(f"{sig}\nz")
    axes[-1].set_xlabel("minutes (grey: driving)")
    fig.suptitle(f"{name}: rolling-median surprise z-score")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
