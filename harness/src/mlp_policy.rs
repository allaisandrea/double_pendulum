//! A policy trained in the world model (`learning/imagination`), run here.
//!
//! `learning`'s `imagination.export` writes the policy as JSON: the actor's
//! layers (tanh between them, none after the last), the int8 action for
//! each output, and the policy window. The input is built as in training
//! (`ImaginedEnv.observe`): for each of the latest `window` frames, the sine
//! and cosine of each tag's yaw (zero where the tag was not seen), whether
//! each tag was seen, and the action in effect before the frame, / 128.
//!
//! Frames are laid out by camera frame number, as the recordings are when
//! read for training: a frame the camera dropped counts as one with no tag
//! seen, and the action carries through it. So the input needs the history
//! back to the frame before the window's first; `window + 1` frames always
//! reach that far.
//!
//! The export also holds test cases, histories with the logits PyTorch
//! gives for them, computed through the training data pipeline. Loading
//! recomputes every case and refuses the policy if any differs.
//!
//! [`TrainedPolicy`] drives the motor with one, in `collect`.

use crate::perturb::Perturbation;
use crate::policy::{splitmix64, wait_until, DutyCycle, Policy, Step};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use std::path::Path;
use std::time::{Duration, Instant};

const FORMAT: &str = "pendulum-policy-v1";
const TAGS: usize = 3;
/// Per frame: sin and cos per tag, seen per tag, the action before it.
const STEP_FEATURES: usize = TAGS * 2 + TAGS + 1;
const ACTION_SCALE: f32 = 128.0;
/// How far a recomputed logit may be from PyTorch's.
const TOLERANCE: f32 = 1e-4;

#[derive(Deserialize)]
struct Export {
    format: String,
    source: serde_json::Value,
    policy_window: usize,
    levels: Vec<i8>,
    layers: Vec<LayerJson>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct LayerJson {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    history: Vec<CaseStep>,
    logits: Vec<f32>,
    action: i8,
}

#[derive(Deserialize)]
struct CaseStep {
    frame: u64,
    poses: [Option<[f32; 7]>; TAGS],
    action: Option<i8>,
}

/// A linear layer, its weight row-major [outputs][inputs].
struct Layer {
    inputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Layer {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.weight
            .chunks_exact(self.inputs)
            .zip(&self.bias)
            .map(|(row, b)| b + row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>())
            .collect()
    }
}

pub struct MlpPolicy {
    window: usize,
    levels: Vec<i8>,
    layers: Vec<Layer>,
    source: String,
}

impl MlpPolicy {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("loading {}", path.display()))
    }

    /// Parses an export and checks it against its test cases.
    pub fn from_json(text: &str) -> Result<Self> {
        let export: Export = serde_json::from_str(text)?;
        ensure!(
            export.format == FORMAT,
            "format {:?}, not {FORMAT:?}",
            export.format
        );
        ensure!(export.policy_window > 0, "an empty policy window");
        let mut inputs = export.policy_window * STEP_FEATURES;
        let mut layers = Vec::new();
        for (i, l) in export.layers.into_iter().enumerate() {
            ensure!(
                l.weight.iter().all(|row| row.len() == inputs),
                "layer {i} does not take {inputs} inputs"
            );
            ensure!(l.bias.len() == l.weight.len(), "layer {i}'s bias and weight differ");
            let outputs = l.bias.len();
            layers.push(Layer {
                inputs,
                weight: l.weight.concat(),
                bias: l.bias,
            });
            inputs = outputs;
        }
        ensure!(
            !layers.is_empty() && inputs == export.levels.len(),
            "the last layer gives {inputs} logits for {} actions",
            export.levels.len()
        );
        let policy = Self {
            window: export.policy_window,
            levels: export.levels,
            layers,
            source: export.source.to_string(),
        };
        ensure!(!export.cases.is_empty(), "no test cases to check the policy against");
        for (i, case) in export.cases.iter().enumerate() {
            policy.check(case).with_context(|| format!("test case {i}"))?;
        }
        Ok(policy)
    }

    fn check(&self, case: &Case) -> Result<()> {
        let history: Vec<Step> = case
            .history
            .iter()
            .map(|s| Step {
                frame: s.frame,
                t: Duration::ZERO,
                poses: s.poses,
                action: s.action,
            })
            .collect();
        let logits = self.logits(&self.features(&history)?);
        let worst = logits
            .iter()
            .zip(&case.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        ensure!(
            logits.len() == case.logits.len() && worst <= TOLERANCE,
            "logits {logits:?}, PyTorch's {:?}",
            case.logits
        );
        let action = self.greedy(&history)?;
        ensure!(action == case.action, "action {action}, PyTorch's {}", case.action);
        Ok(())
    }

    /// Frames of history the input needs, counting the one to act on.
    pub fn history_length(&self) -> usize {
        self.window + 1
    }

    /// The policy's input from `history`, newest first: the frame to act on,
    /// then the ones before it, as `collect` keeps them.
    pub fn features(&self, history: &[Step]) -> Result<Vec<f32>> {
        ensure!(!history.is_empty(), "no frames");
        ensure!(
            history.windows(2).all(|w| w[0].frame > w[1].frame),
            "frames are not newest first"
        );
        let newest = history[0].frame;
        let window = self.window as u64;
        ensure!(newest >= window, "frame {newest} is too early for the window");
        // The action in effect after `frame`: that of the latest frame at or
        // before it, carried through any the camera dropped.
        let action_after = |frame: u64| -> Result<f32> {
            let step = history[1..]
                .iter()
                .find(|s| s.frame <= frame)
                .with_context(|| format!("the history does not reach back to frame {frame}"))?;
            let a = step.action.context("an earlier frame without its action")?;
            Ok(a as f32 / ACTION_SCALE)
        };
        let mut x = Vec::with_capacity(self.window * STEP_FEATURES);
        for frame in newest + 1 - window..=newest {
            let poses = history
                .iter()
                .find(|s| s.frame == frame)
                .map_or([None; TAGS], |s| s.poses);
            for pose in &poses {
                let (s, c) = pose.map_or((0.0, 0.0), |p| {
                    let yaw = yaw(&p);
                    (yaw.sin(), yaw.cos())
                });
                x.extend([s, c]);
            }
            x.extend(poses.iter().map(|p| if p.is_some() { 1.0 } else { 0.0 }));
            x.push(action_after(frame - 1)?);
        }
        Ok(x)
    }

    /// The actor's output, one logit per action, for input `x`.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.apply(&h);
            if i + 1 < self.layers.len() {
                h.iter_mut().for_each(|v| *v = v.tanh());
            }
        }
        h
    }

    /// The likeliest action for `history`, newest first.
    pub fn greedy(&self, history: &[Step]) -> Result<i8> {
        let logits = self.logits(&self.features(history)?);
        let mut best = 0;
        for (i, l) in logits.iter().enumerate() {
            if *l > logits[best] {
                best = i;
            }
        }
        self.level(best)
    }

    /// An action for `history` drawn from the policy's distribution, by
    /// inverting its CDF at `u`, uniform in [0, 1).
    pub fn sample(&self, history: &[Step], u: f64) -> Result<i8> {
        let logits = self.logits(&self.features(history)?);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = logits.iter().map(|l| ((l - max) as f64).exp()).collect();
        let mut left = u * weights.iter().sum::<f64>();
        for (i, w) in weights.iter().enumerate() {
            if left < *w {
                return self.level(i);
            }
            left -= w;
        }
        self.level(weights.len() - 1)
    }

    fn level(&self, output: usize) -> Result<i8> {
        match self.levels.get(output) {
            Some(&a) => Ok(a),
            None => bail!("no action for output {output}"),
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "MLP policy over {} frames, actions {:?}, from {}",
            self.window, self.levels, self.source
        )
    }
}

/// A trained policy driving the motor: the network's action for the newest
/// frame, sent `latency` after the policy was called, as the random walk's
/// were in the recordings the world model learned from, so that the policy
/// meets the delay it was trained with. The network itself takes tens of
/// microseconds.
///
/// Like the random walk, it rests at 0 during the duty cycle's rest periods,
/// which leave the pendulum hanging for the next swing-up. It takes the
/// likeliest action, or with `sample_seed` draws one from the policy's
/// distribution, from a hash of the seed and the frame number, so a run
/// can be repeated from its seed. With `perturbation`, bursts of a random
/// offset are added to its actions while driving, for exploration.
pub struct TrainedPolicy {
    pub net: MlpPolicy,
    pub latency: Duration,
    pub duty: DutyCycle,
    pub sample_seed: Option<u64>,
    pub perturbation: Option<Perturbation>,
    /// Whether the last action was perturbed.
    pub last_perturbed: bool,
}

impl TrainedPolicy {
    /// The action for the newest frame, and whether it was perturbed.
    fn decide(&self, history: &[Step]) -> Result<(i8, bool)> {
        let now = &history[0];
        if self.duty.resting(now.t) {
            return Ok((0, false));
        }
        let action = match self.sample_seed {
            None => self.net.greedy(history)?,
            Some(seed) => {
                let bits = splitmix64(seed ^ splitmix64(now.frame)) >> 11;
                self.net.sample(history, bits as f64 / (1u64 << 53) as f64)?
            }
        };
        Ok(match &self.perturbation {
            Some(p) => p.apply(now.frame, action),
            None => (action, false),
        })
    }
}

impl Policy for TrainedPolicy {
    fn history_length(&self) -> usize {
        self.net.history_length()
    }

    fn act(&mut self, history: &[Step]) -> i8 {
        let deadline = Instant::now() + self.latency;
        // collect always hands over `history_length` frames, newest first,
        // which reach back far enough; anything else is a bug there.
        let (action, perturbed) = self.decide(history).expect("a history the policy can read");
        self.last_perturbed = perturbed;
        wait_until(deadline);
        action
    }

    fn perturbed(&self) -> bool {
        self.last_perturbed
    }

    fn describe(&self) -> String {
        let how = match self.sample_seed {
            None => "likeliest action".to_string(),
            Some(_) => "actions sampled from a hash of seed and frame".to_string(),
        };
        let perturb = self
            .perturbation
            .map_or(String::new(), |p| format!("; perturbed: {}", p.describe()));
        format!(
            "{}; {how}{perturb}; zero while resting; sent {} ms after the call",
            self.net.describe(),
            self.latency.as_secs_f64() * 1e3
        )
    }
}

/// ZYX yaw of a pose (x y z qw qx qy qz): the tag's in-plane angle, as in
/// `learning`'s `data_lib.yaw`.
fn yaw(p: &[f32; 7]) -> f32 {
    let (w, x, y, z) = (p[3], p[4], p[5], p[6]);
    (2.0 * (x * y + w * z)).atan2(1.0 - 2.0 * (y * y + z * z))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("testdata/mlp_policy.json");

    fn step(frame: u64, yaw_deg: Option<f32>, action: Option<i8>) -> Step {
        let pose = yaw_deg.map(|d| {
            let h = d.to_radians() / 2.0;
            [0.0, 0.0, 0.7, h.cos(), 0.0, 0.0, h.sin()]
        });
        Step {
            frame,
            t: Duration::ZERO,
            poses: [pose; TAGS],
            action,
        }
    }

    #[test]
    fn the_fixture_matches_pytorch_on_every_case() {
        let p = MlpPolicy::from_json(FIXTURE).unwrap();
        assert_eq!(p.history_length(), 5);
    }

    #[test]
    fn a_policy_that_disagrees_with_its_cases_is_refused() {
        let mut export: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let w = &mut export["layers"][0]["weight"][0][0];
        *w = serde_json::json!(w.as_f64().unwrap() + 0.5);
        let err = MlpPolicy::from_json(&export.to_string()).err().unwrap();
        assert!(format!("{err:#}").contains("test case"), "{err:#}");
    }

    #[test]
    fn a_dropped_frame_is_unseen_and_carries_the_action() {
        let p = MlpPolicy::from_json(FIXTURE).unwrap(); // window 4
        // Frame 12 was dropped: frames 10, 11, 13 and 14 arrived.
        let history = [
            step(14, Some(90.0), None),
            step(13, Some(0.0), Some(32)),
            step(11, Some(180.0), Some(-64)),
            step(10, None, Some(16)),
        ];
        let x = p.features(&history).unwrap();
        let frame = |i: usize| &x[i * STEP_FEATURES..(i + 1) * STEP_FEATURES];
        // Frames 11 to 14, oldest first, each with the action before it.
        assert!((frame(0)[1] + 1.0).abs() < 1e-6, "cos 180° is -1");
        assert_eq!(frame(0)[9], 16.0 / 128.0);
        assert_eq!(frame(1)[..9], [0.0; 9], "frame 12 was never seen");
        assert_eq!(frame(1)[9], -64.0 / 128.0);
        assert_eq!(frame(2)[9], -64.0 / 128.0, "carried through the drop");
        assert!((frame(3)[0] - 1.0).abs() < 1e-6, "sin 90° is 1");
        assert_eq!(frame(3)[9], 32.0 / 128.0);
    }

    fn trained(sample_seed: Option<u64>, latency: Duration) -> TrainedPolicy {
        TrainedPolicy {
            net: MlpPolicy::from_json(FIXTURE).unwrap(),
            latency,
            duty: DutyCycle {
                active: Duration::from_secs(1),
                rest: Duration::from_secs(1),
            },
            sample_seed,
            perturbation: None,
            last_perturbed: false,
        }
    }

    fn history_at(t: Duration, newest: u64) -> Vec<Step> {
        (0..5)
            .map(|i| Step {
                t,
                ..step(newest - i, Some(30.0 * i as f32), (i > 0).then_some(16))
            })
            .collect()
    }

    #[test]
    fn a_trained_policy_acts_greedily_rests_and_takes_its_latency() {
        let mut p = trained(None, Duration::from_millis(5));
        let history = history_at(Duration::from_millis(500), 100);
        let started = Instant::now();
        let a = p.act(&history);
        let took = started.elapsed();
        assert!(
            took >= Duration::from_millis(5) && took < Duration::from_millis(6),
            "{took:?}"
        );
        assert_eq!(a, p.net.greedy(&history).unwrap());
        assert!(!p.perturbed());
        let resting = history_at(Duration::from_millis(1500), 100);
        assert_eq!(p.act(&resting), 0);
    }

    #[test]
    fn sampled_actions_follow_the_seed_and_the_policy() {
        let p = trained(Some(7), Duration::ZERO);
        let q = trained(Some(7), Duration::ZERO);
        let actions: Vec<i8> = (100..400)
            .map(|f| p.decide(&history_at(Duration::ZERO, f)).unwrap().0)
            .collect();
        let again: Vec<i8> = (100..400)
            .map(|f| q.decide(&history_at(Duration::ZERO, f)).unwrap().0)
            .collect();
        assert_eq!(actions, again, "the same seed and frames give the same actions");
        let distinct: std::collections::HashSet<_> = actions.iter().collect();
        assert!(distinct.len() > 1, "sampling never varied: {distinct:?}");
        // Inverting the CDF at the ends picks the first and last actions
        // the policy gives weight to.
        let h = history_at(Duration::ZERO, 100);
        assert!(p.net.levels.contains(&p.net.sample(&h, 0.0).unwrap()));
        assert!(p.net.levels.contains(&p.net.sample(&h, 0.999_999).unwrap()));
    }

    #[test]
    fn perturbations_offset_driving_actions_only() {
        let mut p = trained(None, Duration::ZERO);
        p.perturbation = Some(Perturbation {
            seed: 2,
            rate: 1.0,
            min_frames: 8,
            max_frames: 8,
            max_offset: 48,
            limit: 96,
        });
        let driving = history_at(Duration::from_millis(500), 100);
        let plain = p.net.greedy(&driving).unwrap();
        let offset = p.perturbation.unwrap().offset(100).unwrap();
        let a = p.act(&driving);
        assert!(p.perturbed());
        assert_eq!(a as i16, (plain as i16 + offset).clamp(-96, 96));
        let resting = history_at(Duration::from_millis(1500), 100);
        assert_eq!(p.act(&resting), 0);
        assert!(!p.perturbed(), "no bursts while resting");
    }

    #[test]
    fn a_history_too_short_for_the_window_is_an_error() {
        let p = MlpPolicy::from_json(FIXTURE).unwrap();
        let history = [
            step(14, Some(0.0), None),
            step(13, Some(0.0), Some(0)),
            step(12, Some(0.0), Some(0)),
        ];
        assert!(p.features(&history).is_err());
    }
}
