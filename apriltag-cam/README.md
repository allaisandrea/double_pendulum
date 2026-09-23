# apriltag-cam

Tracks the pendulum's AprilTags through a USB camera. Two programs:

- **`apriltag-cam`** records what the camera sees: a Motion-JPEG `.mov` with
  each tag's pose drawn on it, and a CSV with one row per detection.
- **`collect`** drives the motor while recording tag poses and the actions
  sent, on one clock, as training data for RL. See [collect](#collect-rl-training-data).

```sh
cargo build
./target/debug/apriltag-cam                  # Ctrl-C to stop
./target/debug/apriltag-cam --duration 60
```

Output lands in `recordings/apriltag-<unix time>.mov` unless `--out` says
otherwise; the CSV takes the same name with a `.csv` extension.

## Setting up a machine

Needs Rust (`rustup`), Xcode's command line tools, and `uv` for the Python
scripts. From the repository root:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # then open a new shell
make -C tools                                     # uvc-util, needed to fix the exposure
cargo build --manifest-path apriltag-cam/Cargo.toml
```

Verified on an M4 MacBook Air with macOS 15, 2026-09-23. The camera shows up
there as `AVCaptureDeviceTypeExternal`; macOS 13 calls it
`ExternalUnknown`, and the vendored nokhwa patch asks for both (see
`vendor/nokhwa-bindings-macos/PATCHED.md`).

## Rig setup

**The camera is mounted upside down, 23.5 inches (0.60 m) from tag 0**
(measured 2026-09-23). Tag 0 is nearest the hub, tag 2 is at the free end.
Each link sits closer to the camera than the one before, so that they can
rotate over each other (see [Pose accuracy](#pose-accuracy)). Tag 2 moves
fastest and is always in front; tags 0 and 1 can be hidden behind the links
in front of them. The arm hanging at rest puts tag 2 near the top of the
image, about 30 px from the edge; the full swing stays inside the frame.

**The tags' black squares are 23 mm** (`--tag-size 0.023`, the default).

**Light the pendulum from the side, not from behind the camera.** A lamp
directly behind the camera caused glare that cost detections at every
setting: at 200 µs and gain 75, moving it took the frames with all three
tags seen from 87% to 91–92%, and stopped gain 50 from blacking out tag 0.

**Exposure defaults to 200 µs at gain 75**, the best setting measured on
this rig (see below). Both are applied through UVC before the camera opens,
and re-checked afterwards; there is nothing to set by hand.

- A short exposure freezes the arm, and the OV9281's global shutter keeps
  the tag squares square rather than sheared.
- The window is narrow and depends on the light. Too short or too little
  gain, and tag 0, in the dimmest part of the frame by the hub, stops
  decoding even at rest. Too long, and the scene saturates: with the lamp
  behind the camera, 300 µs lost a third of the frames with all tags seen
  and 500 µs nearly all of them, even at rest. **Re-measure after changing
  the lighting.**
- Automatic metering is worse than a mediocre fixed exposure, because it
  hunts as the arm swings through frame.

`--exposure-us 0` hands metering back to the camera, for cameras that are not
UVC. Exposure needs `tools/bin/uvc-util`: AVFoundation offers no exposure
control for external cameras on macOS. See `tools/README.md` for the traps
that come with these controls.

## Options worth knowing

| flag | default | meaning |
| --- | --- | --- |
| `--camera` | `Arducam` | part of the name, or an index (order is not stable) |
| `--exposure-us` | `200` | fixed exposure in µs; `0` leaves it automatic |
| `--gain` | `75` | sensor gain, 0..100 |
| `--tag-size` | `0.023` | black square's edge in metres, 0.8× the sheet's nominal size |
| `--hfov` | `72` | horizontal field of view, used when `--fx/--fy` are absent |
| `--decimate` | `2` | quad search runs on an image this many times smaller |
| `--duration` | — | stop after this many seconds |

`collect` takes the same camera and detector flags.

## Detection: what the numbers look like

Measured 2026-09-23 on the M4, with `collect` driving the arm (`--policy-range
60`, seed 1) for 2 minutes per setting. Recall is over the policy's active
periods; at rest it is 99–100% for every tag at the defaults.

**Frames with all three tags detected**, lamp to the side:

| exposure \ gain | 50 | 75 | 100 |
| --- | --- | --- | --- |
| 100 µs | — | 24% | 34% |
| 150 µs | — | **92%** | 88% |
| 200 µs | 88% | **91–92%** | 89% |

Error bars are about ±1 point; repeat runs of the same setting agreed
within them. Per tag at the defaults: tag 0 about 96%, tag 1 about 96%,
tag 2 about 99%.

- **Motion blur no longer limits recall** at these settings: 150 and 200 µs
  tie. Shorter is too dark for tag 0.
- **`--decimate 1` does not help.** Recall is unchanged (91.0% against
  91.1%), and detection takes about 15 ms against 5 ms, so `collect` falls
  to about 68 observations a second and drops 44% of camera frames.
- **The remaining misses are probably occlusion.** They rarely coincide
  across tags: frames with all three tags seen match the product of the
  per-tag rates, where whole-arm blur would make misses coincide. Tags 0
  and 1 miss most, and they are the ones that can pass behind a link.

**Throughput.** The camera delivers 1280×800 at 120 fps, and both programs
keep up: `apriltag-cam` records about 116 fps (3 ms median detection), and
`collect` detects every frame at about 5 ms median with 2 threads. The
capture thread keeps only the newest frame, so if detection ever falls
behind, frames are dropped rather than queued, and the spacing becomes
uneven. Use the timestamps rather than assuming a fixed step.

## Pose accuracy

**Use the in-plane angle; do not trust the tag normals.** A 23 mm tag spans
about 30 pixels here, which puts planar pose estimation deep into its
ambiguous regime: two orientations project almost identically. Roughly two
thirds of poses come back with a second, nearly-as-good solution, and the
reported normal flips between two mirror images about the line of sight,
7–9° either way; physically parallel tags disagree by 8–9° median and up to
43°. The in-plane rotation is unaffected and good to well under a degree
(second difference 0.8–1.8° against 5–12° for the tilt, same frames). For a
planar pendulum, discard the out-of-plane degrees of freedom. The `alt_err`
column flags poses where a second solution was found. (Measured 2026-09-18.)

**Recordings before 2026-09-23 have the wrong scale.** They assumed 40 mm
tags, which makes every position 40/23 = 1.74× too large, and a 60° field
of view (fx 1108.5 px against 881 px now), which makes depth a further 1.26×
too large; x and y do not depend on the focal length. `collect` stores both
in its metadata (`tag_size_m`, `intrinsics`), so rescale x, y and z by
`0.023 / tag_size_m`, and z again by `881 / fx`. The in-plane angle is
unaffected.

**The intrinsics are not calibrated; `--hfov 72` is an estimate.** Depth
scales with the assumed focal length. At the old default of 60° (fx 1108.5
px), with the arm at rest and the correct tag size, the tags reported depths
of 0.745 m (tag 0), 0.725 m (tag 1) and 0.706 m (tag 2), and tag 0
measures 0.597 m from the camera. So the true focal length is the assumed
one times `0.597 / 0.745`: about 890 px, a horizontal field of view of
about 72°. The tape measure's reference point (lens front or sensor) is
uncertain by a centimetre or two, a few percent.

The tags are not coplanar: each link sits closer to the camera than the one
before, so that they can rotate over each other. At the defaults, tag 0
reads 0.592 m, tag 1 0.577 m and tag 2 0.562 m at rest, about 15 mm per
link. Apparent depth also drifts as a single tag crosses the frame
(correlating −0.2 to −0.5 with image radius), which is uncorrected lens
distortion. The tool has no distortion model, so fixing that needs a proper
calibration and code, not just a better field of view.

## collect: RL training data

`collect` drives the motor while recording what the camera sees, with every
timestamp on one monotonic clock (`CLOCK_UPTIME_RAW`). It needs the Arducam
and the Uno running `arduino/arduino.ino`, with the shield's 12 V supply on.

From the repository root:

```sh
./apriltag-cam/target/debug/collect --duration 300                   # ±60: 20 s on, 5 s rest
./apriltag-cam/target/debug/collect --policy-range 0 --duration 20   # everything but motion
uv run apriltag-cam/check_recording.py recordings/<unix time>
uv run analysis/recall.py recordings/<unix time>
```

**If the arm does not move, check the motor supply first.** The Uno runs on
USB power and answers `ready mdd10` without it, so everything looks fine
from the laptop. Both ERR LEDs lit on the shield means undervoltage: the
12 V supply is off or unplugged.

**Two loops on a fixed grid.** Slot k starts at `t0 + k × period` (20 ms).
One loop takes each frame (or one every `--plan-every-ms`, if set), detects
tags, and asks the policy for a chunk of actions (8), which starts at the
first slot at least `--offset-ms` (60) after the frame's capture time. The
other, a real-time executor, wakes at each slot boundary and writes one
byte. A newer plan takes over from its own first slot; until then the
previous plan keeps the slots in between. A slot no plan covers is a
**gap** and gets 0. Planning on every frame, 120 a second, most plans are
superseded after their first action or two. This costs nothing here: the
executor's worst lateness stays under 0.1 ms. On the old two-core laptop it
stalled the executor, which is why `--plan-every-ms` exists; frames between
plans are not detected, so they are missing from the observations.

**The stand-in policy** holds one random action in ±`--policy-range` (60)
for each `--hold-ms` (80) block of slots. The action is a hash of `--seed`
(recorded) and the block, so it does not depend on the planning rate, and
the same seed sends the same action in the same slot on every run. The
double pendulum is chaotic, so the arm's path still differs from run to run,
but repeat runs give matching recall. At ±30 the arm barely swings, because the
direction flips every block or two and the pushes cancel; ±60 gives
vigorous motion. The firmware caps duty at ±80. The policy runs for
`--active-s` (20), then rests for `--rest-s` (5) sending zeros, and repeats,
so one recording holds both driven motion and the arm settling. At zero
duty the shield brakes the motor: the arm settles damped, not free.

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
the same clock. The first few slots of every run are gaps, before the first
plan can arrive.

If the Uno speaks after its ready line, it has reset, so `collect` stops and
reports the recording as unreliable.

## Measuring detection recall

`analysis/recall.py` reports each tag's recall over the active periods of one
or more recordings, with the run's settings, so a parameter sweep reads side
by side. It also warns when a tag was detected near the image border, where
it might leave the frame.

The error bar comes from splitting the active frames into 8 contiguous
chunks and taking the spread of per-chunk recall. Consecutive frames are
strongly correlated, so a per-frame binomial error would be far too
optimistic. To compare settings:

- **Use the same `--seed`** for every run, and run 2 minutes per setting;
  1 minute gave error bars too coarse to separate close settings.
- **Repeat one setting at the start and end** of a sweep, and shuffle the
  rest, so drift (lighting, the motor warming) shows up rather than
  masquerading as an effect. Before the lighting was sorted out, two runs
  of the same setting once differed by 10 points; since, repeats agree
  within about 1.
- **Vary exposure and gain on a grid**, so each can be read with the other
  held fixed.
- **Check recall at rest.** A setting that fails at rest is a lighting
  problem (too dark, glare or saturation), not motion blur.
