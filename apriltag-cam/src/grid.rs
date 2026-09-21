//! The action grid and the rule that splices plans onto it.
//!
//! Slot `k` is active from `t0 + k * period`. A plan is a chunk of actions
//! whose first one belongs to slot `k_start`, a fixed offset after the
//! capture time of the frame the plan was computed from. Plans supersede
//! each other from their `k_start` onwards.
//!
//! Because a plan starts some way after its frame was captured, a new plan
//! usually arrives *before* its first slot. Until then the previous plan
//! still owns the slots in between, so the executor cannot just take the
//! newest plan: it takes the newest one that has started.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    /// When slot 0 begins, on [`crate::clock::mono`].
    pub t0: Duration,
    pub period: Duration,
}

impl Grid {
    /// When slot `k` begins.
    pub fn slot_time(&self, k: i64) -> Duration {
        self.t0 + self.period * u32::try_from(k).expect("slots before 0 are never scheduled")
    }

    /// The first slot starting at or after `t_capture + offset`.
    pub fn k_start(&self, t_capture: Duration, offset: Duration) -> i64 {
        let target = (t_capture + offset).as_nanos() as i128 - self.t0.as_nanos() as i128;
        let p = self.period.as_nanos() as i128;
        // Ceiling division that also rounds up correctly for negative targets.
        (target.div_euclid(p) + i128::from(target.rem_euclid(p) != 0)) as i64
    }

    /// The slot in progress at `t`, or None before slot 0.
    pub fn slot_at(&self, t: Duration) -> Option<i64> {
        let since = t.checked_sub(self.t0)?;
        Some((since.as_nanos() / self.period.as_nanos()) as i64)
    }
}

/// A chunk of actions and where it sits on the grid.
#[derive(Debug)]
pub struct Plan {
    /// The camera frame the plan was computed from.
    pub frame: u64,
    pub k_start: i64,
    pub actions: Vec<i8>,
}

/// Where a slot's action comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Action `index` of the plan computed from `frame`.
    Plan { frame: u64, index: u8, action: i8 },
    /// No plan covers the slot: the executor sends 0.
    Gap,
}

impl Pick {
    pub fn action(&self) -> i8 {
        match *self {
            Pick::Plan { action, .. } => action,
            Pick::Gap => 0,
        }
    }
}

/// The plans that may still own a current or future slot, oldest first.
/// Plans are pushed in frame order, so their `k_start`s never decrease.
#[derive(Default)]
pub struct Schedule {
    plans: VecDeque<Arc<Plan>>,
}

impl Schedule {
    pub fn push(&mut self, plan: Arc<Plan>) {
        self.plans.push_back(plan);
    }

    /// The action for slot `k`: from the newest plan that has started by `k`,
    /// if it still covers `k`, else a gap. Slots must be asked for in
    /// increasing order, because plans superseded at `k` are dropped.
    pub fn pick(&mut self, k: i64) -> Pick {
        // The oldest plan is superseded once the next one has started.
        while self.plans.len() >= 2 && self.plans[1].k_start <= k {
            self.plans.pop_front();
        }
        match self.plans.front() {
            Some(p) if p.k_start <= k && k < p.k_start + p.actions.len() as i64 => {
                let index = (k - p.k_start) as usize;
                Pick::Plan {
                    frame: p.frame,
                    index: index as u8,
                    action: p.actions[index],
                }
            }
            _ => Pick::Gap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(frame: u64, k_start: i64, actions: &[i8]) -> Arc<Plan> {
        Arc::new(Plan {
            frame,
            k_start,
            actions: actions.to_vec(),
        })
    }

    fn actions(s: &mut Schedule, ks: std::ops::Range<i64>) -> Vec<i8> {
        ks.map(|k| s.pick(k).action()).collect()
    }

    #[test]
    fn k_start_rounds_up_to_the_next_slot() {
        let g = Grid {
            t0: Duration::from_millis(1000),
            period: Duration::from_millis(20),
        };
        let off = Duration::from_millis(60);
        assert_eq!(g.k_start(Duration::from_millis(1000), off), 3); // exactly on a slot
        assert_eq!(g.k_start(Duration::from_millis(1001), off), 4); // just after: next one
        assert_eq!(g.k_start(Duration::from_millis(1019), off), 4);
        assert_eq!(g.k_start(Duration::from_millis(900), off), -2); // before t0
        assert_eq!(g.slot_at(Duration::from_millis(1039)), Some(1));
        assert_eq!(g.slot_at(Duration::from_millis(999)), None);
    }

    #[test]
    fn a_newer_plan_takes_over_from_its_start() {
        let mut s = Schedule::default();
        s.push(plan(1, 0, &[1, 1, 1, 1, 1, 1]));
        s.push(plan(2, 3, &[2, 2, 2, 2]));
        // Plan 2 arrived before slot 3; plan 1 keeps slots 0..3.
        assert_eq!(actions(&mut s, 0..8), [1, 1, 1, 2, 2, 2, 2, 0]);
    }

    #[test]
    fn several_plans_can_be_waiting() {
        let mut s = Schedule::default();
        s.push(plan(1, 0, &[1; 8]));
        s.push(plan(2, 2, &[2; 8]));
        s.push(plan(3, 4, &[3; 8]));
        assert_eq!(actions(&mut s, 0..6), [1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn a_late_plan_loses_the_slots_already_past() {
        let mut s = Schedule::default();
        s.push(plan(1, 0, &[1; 4]));
        assert_eq!(actions(&mut s, 0..3), [1, 1, 1]);
        // Starts at slot 2, but only arrives before slot 3.
        s.push(plan(2, 2, &[20, 21, 22, 23]));
        assert_eq!(
            s.pick(3),
            Pick::Plan {
                frame: 2,
                index: 1,
                action: 21
            }
        );
    }

    #[test]
    fn running_off_the_end_is_a_gap_until_the_next_plan() {
        let mut s = Schedule::default();
        assert_eq!(s.pick(0), Pick::Gap); // nothing yet
        s.push(plan(1, 1, &[5, 5]));
        assert_eq!(actions(&mut s, 1..5), [5, 5, 0, 0]);
        s.push(plan(2, 6, &[7]));
        assert_eq!(actions(&mut s, 5..8), [0, 7, 0]);
    }

    #[test]
    fn equal_starts_go_to_the_newer_plan() {
        let mut s = Schedule::default();
        s.push(plan(1, 2, &[1; 4]));
        s.push(plan(2, 2, &[2; 4]));
        assert_eq!(s.pick(2).action(), 2);
    }
}
