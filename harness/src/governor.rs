//! A speed limit on the motor's arm, whatever the policy asks for.
//!
//! The governor measures the speed of the arm the motor drives, tag 0's,
//! from the change in its yaw over the last few frames, and while that is
//! over the limit it sends 0 in place of the policy's action: at zero duty
//! the shield brakes the motor. It needs tag 0 in the newest frame and in
//! the one `span` frames back or earlier; without them it lets the policy's
//! action through.

use crate::policy::Step;

/// The arm whose speed is limited: the one the motor drives.
const TAG: usize = 0;

pub struct Governor {
    /// Revolutions per second the arm may turn at before the motor brakes.
    pub max_rev_s: f64,
    /// Frames back to measure the turn over.
    pub span: usize,
}

impl Governor {
    /// Frames of history `speed` needs, counting the newest.
    pub fn history_length(&self) -> usize {
        self.span + 1
    }

    /// The arm's speed in revolutions per second at the newest frame of
    /// `history` (newest first), from the earliest frame within `span` of
    /// it where the tag was seen; None without one, or without the tag in
    /// the newest frame.
    pub fn speed(&self, history: &[Step]) -> Option<f64> {
        let now = history.first()?;
        let yaw_now = yaw(&now.poses[TAG]?);
        let then = history[1..]
            .iter()
            .take(self.span)
            .filter(|s| s.poses[TAG].is_some())
            .last()?;
        let turned = wrap(yaw_now - yaw(&then.poses[TAG]?));
        let seconds = now.t.checked_sub(then.t)?.as_secs_f64();
        (seconds > 0.0).then(|| turned.abs() / std::f64::consts::TAU / seconds)
    }

    /// The action to send in place of `action`, and whether the governor
    /// overrode it.
    pub fn limit(&self, history: &[Step], action: i8) -> (i8, bool) {
        match self.speed(history) {
            Some(speed) if speed > self.max_rev_s => (0, true),
            _ => (action, false),
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "arm (tag {TAG}) over {} rev/s, measured over {} frames: motor braked at 0",
            self.max_rev_s, self.span
        )
    }
}

/// ZYX yaw of a pose (x y z qw qx qy qz): the tag's in-plane angle.
fn yaw(p: &[f32; 7]) -> f64 {
    let [w, x, y, z] = [p[3], p[4], p[5], p[6]].map(f64::from);
    (2.0 * (x * y + w * z)).atan2(1.0 - 2.0 * (y * y + z * z))
}

/// An angle difference brought into (-π, π].
fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    a - t * ((a + std::f64::consts::PI) / t).ceil() + t
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A history, newest first, of frames 8 ms apart with tag 0 turning at
    /// `rev_s`, seen where `seen` says.
    fn turning(rev_s: f64, seen: &[bool]) -> Vec<Step> {
        let n = seen.len();
        (0..n)
            .map(|i| {
                let k = (n - 1 - i) as f64; // frames since the oldest
                let yaw = 1.0 + std::f64::consts::TAU * rev_s * 0.008 * k;
                let pose = [0.0, 0.0, 0.7, (yaw / 2.0).cos() as f32, 0.0, 0.0, (yaw / 2.0).sin() as f32];
                Step {
                    frame: 100 + k as u64,
                    t: Duration::from_micros(8000 * k as u64),
                    poses: [seen[i].then_some(pose), None, None],
                    action: (i > 0).then_some(64),
                }
            })
            .collect()
    }

    const GOVERNOR: Governor = Governor { max_rev_s: 2.0, span: 3 };

    #[test]
    fn it_measures_speed_in_either_direction_across_the_wrap() {
        for rev_s in [0.5, -1.5, 3.0, -7.0] {
            let speed = GOVERNOR.speed(&turning(rev_s, &[true; 4])).unwrap();
            assert!((speed - rev_s.abs()).abs() < 1e-3, "{rev_s}: {speed}");
        }
    }

    #[test]
    fn it_brakes_only_over_the_limit() {
        assert_eq!(GOVERNOR.limit(&turning(1.9, &[true; 4]), 64), (64, false));
        assert_eq!(GOVERNOR.limit(&turning(2.1, &[true; 4]), 64), (0, true));
        assert_eq!(GOVERNOR.limit(&turning(-2.1, &[true; 4]), -48), (0, true));
    }

    #[test]
    fn it_measures_from_the_earliest_frame_the_tag_was_seen_in() {
        let speed = GOVERNOR.speed(&turning(3.0, &[true, true, false, false])).unwrap();
        assert!((speed - 3.0).abs() < 1e-3);
    }

    #[test]
    fn without_the_tag_it_lets_the_action_through() {
        assert_eq!(GOVERNOR.limit(&turning(5.0, &[false, true, true, true]), 64), (64, false));
        assert_eq!(GOVERNOR.limit(&turning(5.0, &[true, false, false, false]), 64), (64, false));
    }
}
