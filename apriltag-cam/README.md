# apriltag-cam

Records the pendulum through a USB camera, detecting AprilTags in every frame
and writing two files side by side: a Motion-JPEG `.mov` with each tag's pose
drawn on it, and a CSV with one row per detection.

```sh
cargo build
./target/debug/apriltag-cam --camera Arducam        # Ctrl-C to stop
./target/debug/apriltag-cam --camera Arducam --duration 60
```

Output lands in `recordings/apriltag-<unix time>.mov` unless `--out` says
otherwise; the CSV takes the same name with a `.csv` extension.

## Rig setup

**Put the camera about 27 inches (0.69 m) from the pendulum.** That framing
holds the whole swing with room to spare, and keeps the tags large enough in
the image to decode reliably.

**Exposure defaults to 500 µs at gain 100**, which works on the Arducam
OV9281 under normal room lighting. Both are applied automatically before the
camera opens — there is nothing to set by hand.

- A short exposure is what freezes the arm. At 500 µs a point moving 1 m/s
  smears half a millimetre, and the OV9281's global shutter means the tag
  squares stay square rather than shearing.
- Gain 100 is needed to get a usable image back at that exposure. It
  amplifies noise too, but AprilTag tolerates noise far better than blur.
- Automatic metering is worse than a mediocre fixed exposure, because it
  hunts as the arm swings through frame, so detection rate ends up depending
  on where the arm happens to be.

`--exposure-us 0` hands metering back to the camera, for cameras that are not
UVC or when the defaults do not suit the light.

## Options worth knowing

| flag | default | meaning |
| --- | --- | --- |
| `--camera` | `0` | index, or part of the name (e.g. `Arducam`) |
| `--exposure-us` | `500` | fixed exposure in µs; `0` leaves it automatic |
| `--gain` | `100` | sensor gain, 0..100 |
| `--tag-size` | `0.040` | black square's edge in metres, 0.8× the sheet's nominal size |
| `--hfov` | `60` | horizontal field of view, used when `--fx/--fy` are absent |
| `--decimate` | `2` | quad search runs on an image this many times smaller |
| `--duration` | — | stop after this many seconds |

Exposure needs `tools/bin/uvc-util`; build it with `make -C tools`. It cannot
go through AVFoundation, which offers no exposure control for external
cameras on macOS — see `tools/README.md` for the traps that come with these
controls.

## What the numbers actually look like

Measured on the Arducam OV9281 on this laptop, 2026-09-18.

**The camera runs at 120 fps; the recording does not.** A 60 s run produced
1660 frames — 27.7 fps — while the camera delivered 7207. The rest are
dropped at the first hand-off, deliberately: the capture thread keeps only
the newest frame, so poses stay fresh instead of queueing up stale. Detection
costs about 16 ms median with three tags in view, and JPEG encoding competes
for the same two cores.

**Frame spacing is uneven** — 8 ms at best, 33 ms median, 274 ms at worst.
A control loop must use the timestamps rather than assume a fixed step.

**Do not trust the tag normals.** At 27 inches a 40 mm tag spans only tens of
pixels, which puts planar pose estimation deep into its ambiguous regime: two
orientations project almost identically, differing only by perspective worth
a fraction of a pixel. Roughly two thirds of poses come back with a second,
nearly-as-good solution, and the reported normal flips between two mirror
images about the line of sight — same tilt magnitude, opposite direction,
7–9° either way. Tags that are physically parallel disagree by 8–9° median
and up to 43°.

The in-plane rotation is unaffected and is good to well under a degree: its
second difference runs 0.8–1.8° against 5–12° for the normal's tilt, on the
same frames. **For a planar pendulum, use the in-plane angle and discard the
out-of-plane degrees of freedom.** The `alt_err` column flags the poses where
a second solution was found.

**The intrinsics are not calibrated.** `--hfov 60` is a guess inherited from
a different camera, and poses carry whatever scale error it brings. The
27-inch placement gives a cheap cross-check: with `--tag-size` correct, the
reported depth should read 0.69 m, and the true focal length is the assumed
one scaled by `0.69 / reported`. Recordings so far disagree with each other
on this, so measure the distance before trusting it. There is also a residual
radial trend in depth — apparent depth drifts as a tag crosses the frame,
correlating −0.2 to −0.5 with image radius — which is uncorrected lens
distortion. The tool has no distortion model at all, so fixing that needs
code, not just better numbers.

## collect: RL training data

`collect` drives the motor while recording what the camera sees, with every
timestamp on one monotonic clock (`CLOCK_UPTIME_RAW`). It needs the Arducam
and the Uno running `arduino/arduino.ino`.

From the repository root:

```sh
cargo build --manifest-path apriltag-cam/Cargo.toml
./apriltag-cam/target/debug/collect --duration 300 --gain 0          # ±30: 20 s on, 5 s rest
./apriltag-cam/target/debug/collect --policy-range 0 --duration 20   # everything but motion
uv run apriltag-cam/check_recording.py recordings/<unix time>
```

**Two loops on a fixed grid.** Slot k starts at `t0 + k × period` (20 ms).
Every `--plan-every-ms` (80), one loop takes a fresh frame, detects tags and
asks the policy for a chunk of actions (8), which starts at the first slot at
least `--offset-ms` (60) after the frame's capture time. Planning slower than
the grid is what lets a chunk play out: each plan runs about 4 actions before
the next takes over. Planned on every frame (~78 fps), each was superseded
after its first action, and detection kept both cores so busy that the
executor stalled. The other, a high-priority executor, wakes at each
slot boundary and writes one byte. A newer plan takes over from its own first
slot; until then the previous plan keeps the slots in between, since a plan
normally arrives before it starts. A slot no plan covers is a **gap** and gets
0. For now the policy is a stand-in: each plan repeats one random action in
±`--policy-range`, seeded by `--seed` (recorded). It runs for `--active-s`
(20) and then rests for `--rest-s` (5), sending zeros, and repeats, so one
recording holds both driven motion and the arm settling afterwards. The
schedule follows the slot grid, so a rest starts on time even inside a chunk.
At zero duty the shield brakes the motor: the arm settles damped, not free.

**Output**: `recordings/<unix time>/observations.arrows` and `actions.arrows`,
Arrow IPC streams. A stream is readable up to its last batch even after a
crash, and batches are written about once a second. Read them with
`pyarrow.ipc.open_stream` or `polars.read_ipc_stream`. Times are
`Duration(ns)` relative to t0; run parameters are in the schema metadata.

| observations | |
| --- | --- |
| `frame` | camera sequence number |
| `t_capture`, `t_detected` | sensor capture, poses ready |
| `tag{id}_pose` | `[x, y, z, qw, qx, qy, qz]`, f32, null when unseen; qw ≥ 0 |
| `tag{id}_err`, `_alt_err`, `_margin` | pose quality; `alt_err` flags the ambiguous poses |
| `t_policy`, `t_plan` | policy invoked, plan available |
| `k_start`, `plan` | the plan's first slot, and all of its actions |

| actions | |
| --- | --- |
| `slot` | its time is `slot × period` |
| `action` | the byte sent |
| `frame`, `index` | which plan supplied it, and where in it; null for a gap |

While it runs, `collect` prints a status line every `--status-every-s` (10):
frames and how often each tag was seen, slots with their gaps, skips and late
writes, late plans, and whether the policy is active or resting.

A slot missing from `actions` was skipped because the executor fell more than
a slot behind; the board kept the previous action through it. The final
summary reports that, the executor's lateness against the grid, how early plans
arrived before their first slot (tune `--offset-ms` with it), and each frame's
age on arrival, which doubles as a check that the camera's timestamps are on
the same clock.

If the Uno speaks after its ready line, it has reset, so `collect` stops and
reports the recording as unreliable.
