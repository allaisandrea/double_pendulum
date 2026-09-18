# Camera tools

Helpers for checking and configuring USB (UVC) cameras on macOS. Build with
`make -C tools`; the binaries land in `tools/bin/` (git-ignored).

## avprobe

Lists cameras, or streams one and prints its mean brightness and frame rate
twice a second. Use it to confirm an exposure change actually reached the
sensor: a UVC setting can read back correctly and still be ignored.

    tools/bin/avprobe                 # list: name, AVFoundation type, unique id
    tools/bin/avprobe SK-C201 10      # stream 10 s from the first camera whose name matches

Brightness is video-range luma, so black reads 16, not 0.

## uvc-util

[jtfrey/uvc-util](https://github.com/jtfrey/uvc-util) at commit `8110da7`,
vendored unmodified (MIT, see `uvc-util/LICENSE`). It sets UVC controls
directly, which is the only way to set exposure time on macOS: AVFoundation
offers no such control.

    tools/bin/uvc-util -d                          # list UVC cameras
    tools/bin/uvc-util -I 0 -c                     # controls camera 0 implements
    tools/bin/uvc-util -I 0 -S exposure-time-abs   # range, default and current value

`-I` is uvc-util's own camera index, unrelated to the recorder's `--camera`.

Manual exposure, in units of 0.1 ms (so 5 is 0.5 ms):

    tools/bin/uvc-util -I 0 -s auto-exposure-mode=1 -s exposure-time-abs=5

Back to auto: restore the exposure first, while still in manual, because the
camera ignores exposure writes once auto is on. Use the defaults `-S` reports
(39 and 8 on the SK-C201):

    tools/bin/uvc-util -I 0 -s exposure-time-abs=39 && tools/bin/uvc-util -I 0 -s auto-exposure-mode=8

Verified on this Mac (macOS 13, SK-C201 webcam, 2026-09-10): settings take
effect on the live stream within about half a second and survive the camera
being opened, so set the exposure, then start the recorder. The built-in
FaceTime camera is a PCIe device, not UVC, so none of this reaches it.

### Setting exposure reliably

Two traps, both found the hard way on the Arducam OV9281 (2026-09-18):

- **A rejected write looks like a successful one.** While the camera is in
  auto mode it ignores writes to `exposure-time-abs`, reports success, and
  reads back the value it was given while still metering automatically.
  Always set `auto-exposure-mode=1` first, and read every value back.
- **Batching does not work.** Passing several `-s` flags in one invocation
  dropped later writes silently. One control per invocation.

The settings live in the camera, not the host, so a USB re-enumeration
restores its defaults. That happened mid-session here — the kernel logged
`terminateDevice ... connect change interrupt` and the camera came back
metering automatically, so recordings made afterwards had no fixed exposure
and nothing said so.

`apriltag-cam --exposure-us` does all of this, including re-checking
afterwards; prefer it over setting the controls by hand before a recording.
