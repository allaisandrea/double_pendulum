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
    read(libc::CLOCK_UPTIME_RAW)
}

/// How long the machine has spent asleep since boot: how far
/// `CLOCK_MONOTONIC_RAW`, which keeps running through sleep, has pulled
/// ahead of [`mono`], which stops. It grows only while the machine sleeps.
pub fn asleep() -> Duration {
    read(libc::CLOCK_MONOTONIC_RAW).saturating_sub(mono())
}

fn read(clock: libc::clockid_t) -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the call to fill.
    let rc = unsafe { libc::clock_gettime(clock, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime({clock}) failed");
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_asleep_does_not_grow_while_awake() {
        let before = asleep();
        std::thread::sleep(Duration::from_millis(50));
        let grew = asleep().saturating_sub(before);
        assert!(
            grew < Duration::from_millis(5),
            "grew by {grew:?} while awake"
        );
    }
}
