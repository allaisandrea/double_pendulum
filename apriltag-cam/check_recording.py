# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=15"]
# ///
"""Checks a recording made by `collect` for internal consistency.

    uv run apriltag-cam/check_recording.py recordings/<unix time>

Exits non-zero if anything is inconsistent. Checks:
  - every action taken from a plan matches that plan's chunk, at the slot
    the plan assigns it to;
  - no action came from a plan that had already been superseded by a newer
    one available before the slot began;
  - every plan starts at the first slot at least the offset after its frame;
  - every action in a rest period of the stand-in policy is 0;
and reports gaps, skipped slots and how early plans arrived.
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def read(path):
    with ipc.open_stream(path) as reader:
        table = reader.read_all()
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    return table, meta


def ns(col):
    return col.cast(pa.int64()).to_pylist()


def main():
    run = Path(sys.argv[1])
    obs, meta = read(run / "observations.arrows")
    act, _ = read(run / "actions.arrows")
    period, offset = int(meta["period_ns"]), int(meta["offset_ns"])
    problems = []

    frames = obs["frame"].to_pylist()
    t_capture = ns(obs["t_capture"])
    t_plan = ns(obs["t_plan"])
    k_start = obs["k_start"].to_pylist()
    plans = obs["plan"].to_pylist()
    by_frame = {f: i for i, f in enumerate(frames)}

    def ceil_div(a, b):
        return -(-a // b)

    # Plans start at a fixed offset after their frame.
    for f, tc, ks in zip(frames, t_capture, k_start):
        if ks is not None and ks != ceil_div(tc + offset, period):
            problems.append(f"frame {f}: k_start {ks}, expected {ceil_div(tc + offset, period)}")

    slots = act["slot"].to_pylist()
    actions = act["action"].to_pylist()
    src = act["frame"].to_pylist()
    idx = act["index"].to_pylist()

    holes = sum(b - a - 1 for a, b in zip(slots, slots[1:]) if b - a > 1)
    if any(b <= a for a, b in zip(slots, slots[1:])):
        problems.append("slots are not strictly increasing")

    # A plan is usable from the first slot that it has started by and that
    # began after the plan was ready. Sweep the slots, tracking the newest
    # usable plan.
    usable_from = sorted(
        (max(ks, ceil_div(tp, period)), f)
        for f, ks, tp in zip(frames, k_start, t_plan)
        if ks is not None
    )
    gaps, newest, u = 0, None, 0
    for s, a, f, i in zip(slots, actions, src, idx):
        while u < len(usable_from) and usable_from[u][0] <= s:
            newest = max(newest, usable_from[u][1]) if newest is not None else usable_from[u][1]
            u += 1
        if f is None:
            gaps += 1
            if a != 0:
                problems.append(f"slot {s}: gap but action {a}")
            j = by_frame.get(newest)
            if j is not None and s < k_start[j] + len(plans[j]):
                problems.append(f"slot {s}: gap although plan from frame {newest} covered it")
            continue
        j = by_frame.get(f)
        if j is None:
            problems.append(f"slot {s}: action from frame {f}, which is not in observations")
            continue
        if k_start[j] + i != s:
            problems.append(f"slot {s}: frame {f} index {i} belongs to slot {k_start[j] + i}")
        if plans[j][i] != a:
            problems.append(f"slot {s}: action {a}, but plan {f}[{i}] is {plans[j][i]}")
        if newest is not None and f < newest:
            problems.append(f"slot {s}: used plan {f}, superseded by plan {newest} before the slot began")

    # The stand-in policy's rest periods, when the recording has them.
    active, rest = int(meta.get("active_slots", 1)), int(meta.get("rest_slots", 0))
    if rest:
        for s, a in zip(slots, actions):
            if s % (active + rest) >= active and a != 0:
                problems.append(f"slot {s}: action {a} during a rest period")

    lead = sorted((ks * period - tp) / 1e6 for ks, tp in zip(k_start, t_plan) if ks is not None)
    late = sum(1 for x in lead if x < 0)
    print(f"{run}: {len(frames)} observations, {len(slots)} actions")
    print(f"  period {period / 1e6:g} ms, offset {offset / 1e6:g} ms, chunk {meta['chunk_len']}")
    if rest:
        print(f"  policy: {active * period / 1e9:g} s on, {rest * period / 1e9:g} s rest")
    print(f"  gaps: {gaps} ({100 * gaps / max(len(slots), 1):.1f}%), skipped slots: {holes}")
    if lead:
        print(f"  plan lead before its first slot: min {lead[0]:.1f}  "
              f"median {lead[len(lead) // 2]:.1f} ms; {late} late")
    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems[:20]:
            print("  " + p)
        sys.exit(1)
    print("  consistent")


if __name__ == "__main__":
    main()
