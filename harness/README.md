# harness

The laptop side of the pendulum rig: it tracks the pendulum's AprilTags
through a USB camera and drives the motor. Two programs:

- **`record`** records what the camera sees: a Motion-JPEG `.mov` with
  each tag's pose drawn on it, and a CSV with one row per detection.
- **`collect`** drives the motor while recording tag poses and the actions
  sent, on one clock, as training data for RL. See [collect](#collect-rl-training-data).

```sh
cargo build
./target/debug/record                        # Ctrl-C to stop
./target/debug/record --duration 60
```

Output lands in `recordings/video-<unix time>.mov` unless `--out` says
otherwise; the CSV takes the same name with a `.csv` extension.

## Setting up a machine

Needs Rust (`rustup`), Xcode's command line tools, and `uv` for the Python
scripts. From the repository root:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # then open a new shell
make -C tools                                     # uvc-util, needed to fix the exposure
cargo build --manifest-path harness/Cargo.toml
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

**The tags' black squares are 23 mm** (`TAG_SIZE_M` in `src/constants.rs`,
along with the tag family and the camera's field of view).

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
| `--decimate` | `2` | quad search runs on an image this many times smaller |
| `--duration` | — | stop after this many seconds |

`collect` takes the same camera and detector flags.

## Detection: what the numbers look like

Measured 2026-09-23 on the M4, with `collect` driving the arm (`--policy-range
60`, seed 1) for 2 minutes per setting, under the stand-in policy of the time:
a new random action every 80 ms. Recall is over the policy's active
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
keep up: `record` records about 116 fps (3 ms median detection), and
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

**The intrinsics are not calibrated; `HFOV_DEG` (72) is an estimate.** Depth
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
./harness/target/debug/collect --duration 300                   # ±60: 20 s on, 5 s rest
./harness/target/debug/collect --policy-range 0 --duration 20   # everything but motion
uv run harness/check_recording.py recordings/<unix time>
uv run analysis/recall.py recordings/<unix time>
```

**If the arm does not move, check the motor supply first.** The Uno runs on
USB power and answers `ready mdd10` without it, so everything looks fine
from the laptop. Both ERR LEDs lit on the shield means undervoltage: the
12 V supply is off or unplugged.

**Three threads, no clock to wait on.**

    capture -> [newest frame wins] -> detect -> [every detection] -> policy -> serial
                                                                       \-> frames.arrows

Capture drains the camera and keeps only the newest frame, so detection
never works on a stale one. Detection passes every result on. The policy
thread takes everything that has arrived, acts on the newest frame, and
writes one byte to the motor at once. Frames that arrived while it was busy
are **skipped**: they still go into its history and the table, with the
action that was in effect. With 7 ms of simulated inference, detection at
about 5 ms and a frame every 8.3 ms, nothing is skipped and the motor gets
a new command 27 ms after capture (median; 31 ms at p95). The previous
design queued each command for a fixed 60 ms.

If frames stop arriving, the last command stays in force until the
firmware's watchdog zeroes it, 300 ms after the last byte. When `collect`
stops, the policy thread sends a final zero.

**The policy** is anything implementing `Policy` in `src/policy.rs`. It gets
exactly `HISTORY_LENGTH` (3) frames, newest first: the current frame and the
ones before it, each with its number, capture time since t0, the poses of
tags 0, 1 and 2 (none when unseen), and the action in effect after it (none
for the current frame). It returns one action. It is first called once a
recording has that many frames; the earlier ones only fill its history, and
the motor stays at 0. The length is recorded as `history` in the metadata.

**The stand-in policy** is a random walk: each frame the action moves by a
step drawn uniformly from ±`--policy-step` (10), reflecting at
±`--policy-range` (60). It takes about 3·(range/step)² frames to wander from
0 to a limit, about 0.9 s at the defaults. It keeps no state: the walk
continues from the previous frame's action, which is in the history, and
each step is a hash of `--seed` (recorded) and the frame number. It spends
`--policy-latency-ms` (7) per call, standing in for inference; a plain sleep
overshoots by a quarter or more on macOS, so it sleeps most of the time and
spins for the last millisecond. It runs for `--active-s` (20), then rests
for `--rest-s` (5) sending zeros, and repeats, so one recording holds both
driven motion and the arm settling. The cycle runs on capture time since
t0. At zero duty the shield brakes the motor: the arm settles damped, not
free. The firmware caps duty at ±80.

**Output**: `recordings/<unix time>/frames.arrows`, an Arrow IPC stream,
one row per frame that went through detection. A stream is readable up to
its last batch even after a crash, and batches are written about once a
second. Read it with `pyarrow.ipc.open_stream` or `polars.read_ipc_stream`.
Times are `Duration(ns)` relative to t0, which is taken just before the
camera starts; run parameters are in the schema metadata.

| column | |
| --- | --- |
| `frame` | camera sequence number; a jump means frames dropped before detection |
| `t_capture`, `t_arrival` | sensor capture, frame reached the program |
| `t_detect_start`, `t_detected` | detection ran |
| `tag{id}_pose` | `[x, y, z, qw, qx, qy, qz]`, f32, null when unseen; qw ≥ 0 |
| `tag{id}_err`, `_alt_err`, `_margin` | pose quality; `alt_err` flags the ambiguous poses |
| `acted` | whether the policy acted on this frame, or skipped it while busy |
| `t_policy_start`, `t_policy_done`, `t_sent` | policy ran, action written; null when skipped |
| `action` | the action in effect after this frame |

While it runs, `collect` prints a status line every `--status-every-s` (10):
frames and how often each tag was seen, actions sent and frames skipped, the
slowest detection and the slowest capture-to-send, and whether the policy is
active or resting. The final summary adds each frame's age on arrival,
which doubles as a check that the camera's timestamps are on the same clock,
and the time frames spent queued, in inference, and from capture to send.
The first frames after the camera starts can arrive a few hundred ms late;
they are recorded with their true times.

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
