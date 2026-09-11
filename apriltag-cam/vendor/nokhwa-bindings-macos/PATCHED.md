# nokhwa-bindings-macos 0.2.4, patched

Vendored from crates.io and swapped in with `[patch.crates-io]` in
`apriltag-cam/Cargo.toml`. Apache-2.0: the published crate omits its license
file, so `LICENSE` is copied from the nokhwa crate it belongs to.

## Why

nokhwa finds external (USB) cameras by asking AVFoundation for
`AVCaptureDeviceTypeExternal`, which only exists from macOS 14. Up to macOS 13
external cameras have type `AVCaptureDeviceTypeExternalUnknown`, so on this
Mac (macOS 13) no USB camera was visible. Upstream switched from one name to
the other in PRs #172 and #176 (see issue #155).

Separately, upstream lists cameras with one set of device types and opens
them by index with another, so an index from the listing could name a
different camera.

Opening a camera then crashed: `set_all` picks the device format by size
alone, overwriting its choice with every format of that size while keeping
the first matching frame-rate range. The SK-C201 webcam offers 1920x1080 at
30 fps (`420v`) and at 5 fps (`yuvs`), so it got the 5 fps format with a
30 fps frame duration, which AVFoundation rejects with an uncaught exception.

## Change

Search `src/lib.rs` for `PATCH (apriltag-cam)`:

- adds `AVCaptureDeviceType::ExternalUnknown`;
- adds `AVCaptureDeviceType::all()`, used by both discovery sessions: the
  listing's original types plus `ExternalUnknown`;
- `set_all` takes the format and frame-rate range together, from the first
  format of the requested size that supports the requested frame rate;
- `#![allow(warnings)]`, since cargo only caps lints for registry crates.

Drop the patch once upstream asks for both external types, or once this
machine runs macOS 14 or later.
