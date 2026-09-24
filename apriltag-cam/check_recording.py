# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=15"]
# ///
"""Checks a recording made by `collect` for internal consistency.

    uv run apriltag-cam/check_recording.py recordings/<unix time>

Exits non-zero if anything is inconsistent. Checks, on frames.arrows:
  - frame numbers increase, and each row's times are in pipeline order:
    capture, arrival, detection start and end, then, for frames the policy
    acted on, policy start, policy done and send;
  - frames the policy skipped have no policy times, and carry the action of
    the last frame it acted on;
  - with the stand-in random walk: every action is within the policy range,
    consecutive actions differ by at most the step, and actions decided in a
    rest period are 0.
Reports camera frames that never reached detection, skipped frames, and
the delay from capture to send.
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

TIMES = ["t_capture", "t_arrival", "t_detect_start", "t_detected"]
POLICY_TIMES = ["t_policy_start", "t_policy_done", "t_sent"]


def read(path):
    with ipc.open_stream(path) as reader:
        table = reader.read_all()
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    return table, meta


def ns(col):
    return col.cast(pa.int64()).to_pylist()


def resting(t, active, rest):
    return rest > 0 and t % (active + rest) >= active


def quantiles(v):
    if not v:
        return "none"
    v = sorted(v)
    q = lambda p: v[round((len(v) - 1) * p)]
    return f"min {q(0):.1f}  median {q(0.5):.1f}  p95 {q(0.95):.1f}  max {q(1):.1f} ms"


def main():
    run = Path(sys.argv[1])
    frames, meta = read(run / "frames.arrows")
    problems = []

    seq = frames["frame"].to_pylist()
    times = {c: ns(frames[c]) for c in TIMES + POLICY_TIMES}
    acted = frames["acted"].to_pylist()
    actions = frames["action"].to_pylist()

    if any(b <= a for a, b in zip(seq, seq[1:])):
        problems.append("frame numbers do not strictly increase")
    never_detected = sum(b - a - 1 for a, b in zip(seq, seq[1:]) if b > a)

    rng = int(meta.get("policy_range", 127))
    step = int(meta["policy_step"]) if "policy_step" in meta else None
    active, rest = int(meta.get("active_ns", 1)), int(meta.get("rest_ns", 0))

    last = 0  # the action in effect: 0 until the policy first acts
    delay = []
    for i, f in enumerate(seq):
        row = [times[c][i] for c in TIMES]
        if acted[i]:
            row += [times[c][i] for c in POLICY_TIMES]
        elif any(times[c][i] is not None for c in POLICY_TIMES):
            problems.append(f"frame {f}: skipped, but has policy times")
        if any(b < a for a, b in zip(row, row[1:])):
            problems.append(f"frame {f}: times out of order: {row}")

        a = actions[i]
        if abs(a) > rng:
            problems.append(f"frame {f}: action {a} outside ±{rng}")
        if not acted[i]:
            if a != last:
                problems.append(f"frame {f}: skipped, carries {a}, but {last} was in effect")
            continue
        if resting(times["t_capture"][i], active, rest):
            if a != 0:
                problems.append(f"frame {f}: action {a} decided in a rest period")
        elif step is not None and abs(a - last) > step:
            problems.append(f"frame {f}: action moved {last} -> {a}, more than the step {step}")
        last = a
        delay.append((times["t_sent"][i] - times["t_capture"][i]) / 1e6)

    n, n_acted = len(seq), sum(acted)
    print(f"{run}: {n} frames, policy acted on {n_acted}, skipped {n - n_acted} "
          f"({100 * (n - n_acted) / max(n, 1):.1f}%)")
    print(f"  camera frames that never reached detection: {never_detected}")
    print(f"  capture to send: {quantiles(delay)}")
    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems[:20]:
            print("  " + p)
        sys.exit(1)
    print("  consistent")


if __name__ == "__main__":
    main()
