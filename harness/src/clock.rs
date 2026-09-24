//! The one clock every timestamp is taken on.
//!
//! `CLOCK_UPTIME_RAW` is the mach_absolute_time clock in nanoseconds: it
//! only moves forward, is never slewed by NTP, and is the clock AVFoundation
//! stamps camera frames with. With the vendored nokhwa patch, a frame's
//! `capture_timestamp()` is a reading of this same clock, so capture times
//! and anything measured here can be compared directly.

use std::time::Duration;

/// Now, on the monotonic clock, as time since boot (excluding sleep).
pub fn mono() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the call to fill.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_UPTIME_RAW) failed");
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}
