//! Exploration around a trained policy: short bursts of a random offset
//! added to its actions, so that the recordings show the world model what
//! larger and opposing actions do in the states the policy visits, without
//! the violence of a wide random walk.
//!
//! A burst starts on a frame with probability `rate`, lasts
//! `min_frames..=max_frames` camera frames, and adds one offset, drawn
//! uniformly from `-max_offset..=max_offset`, to every action in it; the
//! sum is clipped to `±limit`. Whether a frame is in a burst, and its
//! offset, are hashes of the seed and frame numbers, so a run can be
//! repeated from its seed and nothing is kept between calls. Where bursts
//! overlap, the latest one still covering a frame sets its offset.

use crate::policy::splitmix64;

#[derive(Clone, Copy, Debug)]
pub struct Perturbation {
    pub seed: u64,
    /// Chance that a burst starts on any one frame.
    pub rate: f64,
    pub min_frames: u64,
    pub max_frames: u64,
    pub max_offset: u8,
    pub limit: u8,
}

/// Salts that keep the three draws of a frame independent.
const START: u64 = 0x5354_4152_5421;
const LENGTH: u64 = 0x4c45_4e47_5448;
const OFFSET: u64 = 0x4f46_4653_4554;

impl Perturbation {
    fn draw(&self, salt: u64, frame: u64) -> u64 {
        splitmix64(self.seed ^ salt ^ splitmix64(frame))
    }

    fn starts(&self, frame: u64) -> bool {
        let u = (self.draw(START, frame) >> 11) as f64 / (1u64 << 53) as f64;
        u < self.rate
    }

    /// The burst starting on `frame`, if one does: its length in frames
    /// and its offset.
    pub fn burst(&self, frame: u64) -> Option<(u64, i16)> {
        self.starts(frame).then(|| {
            let span = self.max_frames - self.min_frames + 1;
            let width = 2 * self.max_offset as u64 + 1;
            (
                self.min_frames + self.draw(LENGTH, frame) % span,
                (self.draw(OFFSET, frame) % width) as i16 - self.max_offset as i16,
            )
        })
    }

    /// The offset in effect on `frame`, from the latest burst covering it,
    /// or None outside bursts.
    pub fn offset(&self, frame: u64) -> Option<i16> {
        (0..self.max_frames.min(frame + 1))
            .map(|back| frame - back)
            .find_map(|start| self.burst(start).filter(|&(len, _)| frame - start < len))
            .map(|(_, offset)| offset)
    }

    /// `action` on `frame`, with the offset added and clipped to ±limit if
    /// a burst covers it; and whether one did.
    pub fn apply(&self, frame: u64, action: i8) -> (i8, bool) {
        match self.offset(frame) {
            None => (action, false),
            Some(o) => {
                let l = self.limit as i16;
                ((action as i16 + o).clamp(-l, l) as i8, true)
            }
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "bursts of {}..={} frames starting with probability {} a frame, adding an offset \
             uniform in ±{}, clipped to ±{}",
            self.min_frames, self.max_frames, self.rate, self.max_offset, self.limit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(seed: u64) -> Perturbation {
        Perturbation {
            seed,
            rate: 0.02,
            min_frames: 3,
            max_frames: 8,
            max_offset: 48,
            limit: 96,
        }
    }

    #[test]
    fn bursts_are_bounded_and_set_the_offset_of_the_frames_they_cover() {
        let q = p(3);
        let n = 200_000;
        let bursts: Vec<(u64, u64, i16)> =
            (0..n).filter_map(|f| q.burst(f).map(|(len, o)| (f, len, o))).collect();
        assert!(bursts.iter().all(|&(_, len, o)| (3..=8).contains(&len) && o.abs() <= 48));
        let rate = bursts.len() as f64 / n as f64;
        assert!((0.018..0.022).contains(&rate), "{rate}");
        // Each frame takes the offset of the latest burst still covering it.
        let offsets: Vec<Option<i16>> = (0..n).map(|f| q.offset(f)).collect();
        let latest = |g: u64| {
            bursts.iter().rev().find(|&&(f, len, _)| f <= g && g < f + len).map(|&(_, _, o)| o)
        };
        assert!((0..n).step_by(97).all(|g| offsets[g as usize] == latest(g)));
        let share = offsets.iter().filter(|o| o.is_some()).count() as f64 / n as f64;
        // About rate * mean length (5.5 frames), less the overlaps.
        assert!((0.08..0.12).contains(&share), "{share}");
        let distinct: std::collections::HashSet<_> = offsets.iter().flatten().collect();
        assert!(distinct.len() > 80, "offsets barely vary: {}", distinct.len());
    }

    #[test]
    fn the_same_seed_repeats_and_another_differs() {
        let (a, b, c) = (p(5), p(5), p(6));
        let run = |q: &Perturbation| (0..5000).map(|f| q.offset(f)).collect::<Vec<_>>();
        assert_eq!(run(&a), run(&b));
        assert_ne!(run(&a), run(&c));
    }

    #[test]
    fn perturbed_actions_are_clipped_and_marked() {
        let q = Perturbation { rate: 1.0, max_offset: 48, limit: 96, ..p(1) };
        let f = (0..1000).find(|&f| q.offset(f).unwrap() > 40).unwrap();
        let (a, marked) = q.apply(f, 64);
        assert!(marked);
        assert_eq!(a, 96, "64 + {} clipped", q.offset(f).unwrap());
        let off = Perturbation { rate: 0.0, ..p(1) };
        assert_eq!(off.apply(f, 64), (64, false));
    }
}
