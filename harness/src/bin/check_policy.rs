//! Loads a policy exported by `learning`'s `imagination.export`, which
//! checks it against the test cases it carries, and times its forward pass.
//!
//!     cargo run --release --bin check_policy -- ../learning/runs/policy/ppo-persistent/policy.json

use anyhow::Result;
use clap::Parser;
use harness::mlp_policy::MlpPolicy;
use harness::policy::Step;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Args {
    /// The policy's JSON export
    policy: PathBuf,
    /// Forward passes to time
    #[arg(long, default_value_t = 10_000)]
    passes: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let policy = MlpPolicy::load(&args.policy)?;
    println!("{}: matches PyTorch on all its test cases", args.policy.display());
    println!("{}", policy.describe());

    // A history of the pendulum hanging, every tag seen, the motor at 0.
    let n = policy.history_length() as u64;
    let pose = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0];
    let history: Vec<Step> = (0..n)
        .map(|i| Step {
            frame: 1000 - i,
            t: Duration::ZERO,
            poses: [Some(pose); 3],
            action: (i > 0).then_some(0),
        })
        .collect();

    let mut times: Vec<Duration> = (0..args.passes)
        .map(|_| {
            let started = Instant::now();
            std::hint::black_box(policy.greedy(std::hint::black_box(&history)).unwrap());
            started.elapsed()
        })
        .collect();
    times.sort();
    let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
    println!(
        "input and forward pass: median {:.0} µs, 99th percentile {:.0} µs, worst {:.0} µs",
        at(0.5),
        at(0.99),
        at(1.0)
    );
    println!("action from hanging still: {}", policy.greedy(&history)?);
    Ok(())
}
