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
