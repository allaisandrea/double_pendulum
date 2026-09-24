//! The policy interface `collect` drives the motor through, and a stand-in.
//!
//! A policy sees the newest detected frame and the ones before it, and
//! returns one action, which is sent to the motor at once. It is called on
//! the newest frame only; frames that arrived while it was busy still appear
//! in the history it gets next time, with the action that was in effect.

use std::time::Duration;

/// How many frames a policy sees: the current one and the two before it.
pub const HISTORY: usize = 3;

/// One detected frame, as a policy sees it.
#[derive(Clone, Debug)]
pub struct Step {
    /// Camera sequence number.
    pub frame: u64,
    /// Capture time, since the recording's t0.
    pub t: Duration,
    /// One pose per recorded tag, in `--tags` order: x y z in metres, then
    /// the quaternion w x y z with w >= 0. None when the tag was not seen.
    pub poses: Vec<Option<[f32; 7]>>,
    /// The action in effect after this frame: the one the policy chose for
    /// it, or the one carried over if it was skipped. None for the frame
    /// being decided.
    pub action: Option<i8>,
}

/// Chooses a motor action from recent frames.
pub trait Policy: Send {
    /// `history` is newest first: the frame to act on, then up to
    /// `HISTORY - 1` earlier ones (fewer at the start of a recording).
    fn act(&mut self, history: &[Step]) -> i8;

    /// A one-line description, recorded in the recording's metadata.
    fn describe(&self) -> String;
}

/// Alternating active and rest periods, in time since t0.
#[derive(Clone, Copy, Debug)]
pub struct DutyCycle {
    pub active: Duration,
    /// Zero never rests.
    pub rest: Duration,
}

impl DutyCycle {
    pub fn resting(&self, t: Duration) -> bool {
        !self.rest.is_zero()
            && t.as_nanos() % (self.active + self.rest).as_nanos() >= self.active.as_nanos()
    }
}

/// Stand-in for a real policy: a random walk, reflected at ±`range`, that
/// moves by a step drawn uniformly from -`step`..=`step` on each frame, and
/// is zero while resting. It spends `latency` on each call, standing in for
/// inference.
///
/// It keeps no state: the walk continues from the previous frame's action,
/// which is in the history, and each step is a hash of the seed and the
/// frame number rather than a draw from a running generator.
pub struct RandomWalk {
    pub seed: u64,
    pub range: i8,
    pub step: u8,
    pub latency: Duration,
    pub duty: DutyCycle,
}

impl RandomWalk {
    /// The action for `now` given the previous frame's action, without the
    /// simulated latency.
    fn decide(&self, now: &Step, prev: i8) -> i8 {
        if self.duty.resting(now.t) {
            return 0;
        }
        let span = 2 * self.step as u64 + 1;
        let da = (splitmix64(self.seed ^ splitmix64(now.frame)) % span) as i64 - self.step as i64;
        let r = self.range as i64;
        let mut a = prev as i64 + da;
        if a > r {
            a = 2 * r - a;
        } else if a < -r {
            a = -2 * r - a;
        }
        // A step larger than the whole range could reflect past the far end.
        a.clamp(-r, r) as i8
    }
}

impl Policy for RandomWalk {
    fn act(&mut self, history: &[Step]) -> i8 {
        wait(self.latency);
        let prev = history.get(1).and_then(|s| s.action).unwrap_or(0);
        self.decide(&history[0], prev)
    }

    fn describe(&self) -> String {
        format!(
            "random walk: step uniform in ±{} per frame, reflected at ±{}, from a hash of \
             seed and frame; zero while resting; {} ms simulated latency",
            self.step,
            self.range,
            self.latency.as_secs_f64() * 1e3
        )
    }
}

/// Waits for `d`, precisely. macOS stretches an ordinary sleep: 7 ms takes
/// 9-10 ms here, more than a camera frame. So this sleeps half of what is
/// left while that is more than the stretch could matter, then spins for the
/// last millisecond or so.
fn wait(d: Duration) {
    let deadline = std::time::Instant::now() + d;
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return;
        }
        if left > Duration::from_micros(1500) {
            std::thread::sleep(left / 2);
        } else {
            std::hint::spin_loop();
        }
    }
}

/// A 64-bit mixing function (SplitMix64's finaliser): nearby inputs give
/// unrelated outputs, and it is fixed here, independent of any crate.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn step(frame: u64, t: Duration, action: Option<i8>) -> Step {
        Step { frame, t, poses: vec![], action }
    }

    fn walk(range: i8, step: u8, duty: DutyCycle) -> RandomWalk {
        RandomWalk { seed: 1, range, step, latency: Duration::ZERO, duty }
    }

    const ALWAYS: DutyCycle = DutyCycle { active: Duration::from_secs(1), rest: Duration::ZERO };

    /// Runs the walk over `n` frames 8 ms apart, feeding each action back.
    fn run(p: &mut RandomWalk, n: u64) -> Vec<i8> {
        let mut prev: Option<Step> = None;
        let mut out = Vec::new();
        for f in 0..n {
            let now = step(f, ms(8 * f), None);
            let history: Vec<Step> = std::iter::once(now.clone()).chain(prev.clone()).collect();
            let a = p.act(&history);
            out.push(a);
            prev = Some(Step { action: Some(a), ..now });
        }
        out
    }

    #[test]
    fn simulated_latency_is_close_to_what_was_asked() {
        for want in [ms(0), ms(1), ms(7)] {
            let t = std::time::Instant::now();
            wait(want);
            let took = t.elapsed();
            assert!(took >= want && took < want + Duration::from_micros(500), "{want:?} took {took:?}");
        }
    }

    #[test]
    fn rests_start_and_end_on_time() {
        let d = DutyCycle { active: ms(40), rest: ms(60) };
        let pattern: Vec<bool> = (0..10).map(|i| d.resting(ms(20 * i))).collect();
        let (f, t) = (false, true);
        assert_eq!(pattern, [f, f, t, t, t, f, f, t, t, t]);
        assert!((0..1000).all(|i| !ALWAYS.resting(ms(i))));
    }

    #[test]
    fn the_walk_moves_by_at_most_a_step_and_stays_in_range() {
        let a = run(&mut walk(20, 5, ALWAYS), 5000);
        assert!(a.windows(2).all(|w| (w[1] as i16 - w[0] as i16).abs() <= 5));
        assert!(a.iter().all(|x| (-20..=20).contains(x)));
        // It gets to both ends, and reflection keeps it from sticking there.
        assert!(a.contains(&20) && a.contains(&-20));
        let at_ends = a.iter().filter(|x| x.abs() == 20).count();
        assert!(at_ends < a.len() / 10, "{at_ends} of {} at the limits", a.len());
    }

    #[test]
    fn the_same_inputs_give_the_same_action() {
        let p = walk(60, 10, ALWAYS);
        let now = step(123, ms(5000), None);
        assert_eq!(p.decide(&now, 17), p.decide(&now, 17));
        // Frame numbers pick the step, so different frames differ somewhere.
        assert!((0..50).any(|f| p.decide(&step(f, ms(0), None), 0) != p.decide(&now, 0)));
    }

    #[test]
    fn it_rests_at_zero_and_restarts_from_zero() {
        // 400 ms on, 400 ms off; frames every 8 ms.
        let a = run(&mut walk(60, 10, DutyCycle { active: ms(400), rest: ms(400) }), 150);
        assert!(a[50..100].iter().all(|&x| x == 0), "not zero while resting");
        assert!(a[..50].iter().any(|&x| x != 0), "never moved while active");
        assert!(a[100].abs() <= 10, "did not restart from zero: {}", a[100]);
    }

    #[test]
    fn a_skipped_frame_continues_from_the_action_carried_through_it() {
        let mut p = walk(60, 10, ALWAYS);
        // Frame 2 was skipped: its action is the one carried from frame 1.
        let history = [step(3, ms(24), None), step(2, ms(16), Some(40)), step(1, ms(8), Some(40))];
        let a = p.act(&history);
        assert!((30..=50).contains(&a));
    }
}
