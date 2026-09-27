# learning

The Python learning code, as one uv project. Run everything from this
directory.

- `common/`: code the other packages share. `data_lib.py` loads the frames
  tables `collect` writes (schema in `harness/src/table.rs`) as steps;
  `render_lib.py` draws the pendulum from its yaws into videos, for real or
  imagined runs; `testing_lib.py` builds small tables for tests.
- `world_model/`: an auto-regressive world model of the pendulum. Given a
  window of steps, it predicts the next frame's observation.
- `imagination/`: trains a policy with PPO, using the world model as the
  simulator.

The world model experiments so far (scaling, rollout training, closed-loop
fidelity) are written up in `docs/world_model_experiments.md`.

Library modules end in `_lib`, and each has its tests next to it in
`<module>_test.py`. A package's run configs are in its `configs/`.

## The world model

- **A step** is one camera frame (8 ms at 125 fps): the sine and cosine of
  each tag's yaw (its in-plane angle), whether each tag was seen, and the
  action in effect after the frame, divided by 128. A frame the camera
  dropped becomes a step with no tag seen and the action carried over.
- **The model** is an MLP over the flattened window. Per tag, it outputs
  the change in sine and cosine since the tag was last seen in the window,
  and the logit of the tag going unseen in the next frame.
- **The loss** is the mean squared error of the change, over the tags seen
  in the next frame, plus `bce_weight` times the cross-entropy of the
  missing logits. Training windows are drawn uniformly from every start in
  every training recording.

### Running

```sh
aws --profile andrea-personal s3 sync s3://allais-andrea-store/double_pendulum/data/ data/
uv run wandb login                                                 # once
uv run python -m world_model.train world_model/configs/base.toml --name base
uv run python -m world_model.train world_model/configs/base.toml --name smoke --wandb disabled --steps 200   # smoke test
uv run pytest                                                      # every *_test.py
```

The config names the training and validation recordings by directory
name. `val` can be a table of named validation sets, each reported under
its name: `base.toml` validates on a random-walk recording
(`val_random_walk`) and on one collected with a trained policy
(`val_policy`). `random_walk_only.toml` is `base.toml` without the trained
policy's training recording, for comparison. The run's evaluation sample and latest checkpoint go to
`runs/world_model/<name>/`, which must not exist yet.

### Metrics

Logged every `log_every` steps, on the batch just trained on:

- `train/batch_mse`: the squared error of the predicted change, in units
  of `delta_scale` (the typical one-frame change), over the tags seen in
  the next frame.
- `train/batch_bce`: the cross-entropy of the missing-tag logits.
- `lr`: the learning rate, following the trapezoid.

Every `eval_every` steps, training pauses to evaluate on a fixed sample of
windows from each split (`train` and `val`): the same sample every time,
drawn from `seed`. From each window it rolls the model forward open loop
for `max(horizons)` frames with the recorded actions. Each prediction is
fed back as seen, with its sine and cosine put back on the unit circle.

- `{train,val}/mse`, `{train,val}/bce`: the two loss terms at one step, as
  above.
- `{train,val}/one_minus_r2/hNNN`: 1 − R² at horizon N frames, where R²
  measures the predicted change against the actual change since each tag
  was last seen in the starting window, over the tags seen at that frame.
  0 is perfect.
- `{train,val}/copy_last/hNNN`: the same score for predicting no change,
  which comes out at about 1. It is the reference line.

`world_model.compare` evaluates several checkpoints on the same fixed
windows of any recordings, so models trained or validated on different
data compare like for like:

```sh
uv run python -m world_model.compare runs/world_model/wm3/model.pt runs/world_model/wm-with-policy-50k/model.pt --val 1790378604
```

Because the change being predicted grows with the horizon, 1 − R² need not
increase monotonically with it. Compare one horizon across runs rather than
reading it as an error curve.

## Training in imagination

`imagination/train.py` runs PPO (after CleanRL's
`ppo_continuous_action_isaacgym.py`) in a batch of environments that the
world model steps.

- **Episodes** start from real windows of the recordings. The `num_envs`
  environments carry on across iterations, each iteration running them
  for `rollout_steps` frames. After each frame, an environment starts
  again from a new window with probability (1 − `gamma`) /
  `reset_horizons`, so episodes last `reset_horizons` / (1 − `gamma`)
  frames on average (400, 3.2 s, in `base.toml`). With `reset_horizons` 1,
  states are weighted as the discounted objective weights them; larger
  values give the long run more weight. Episodes end only by truncation,
  so returns are bootstrapped from the critic's value of the state an
  episode would have gone on to, at a reset as at the end of an iteration.
- **Upright starts**: `upright_start_fraction` of the episode starts are
  drawn from the recorded windows whose last frame has every arm seen and
  within 30° of upright (0 in `base.toml`), so the critic can learn what
  staying up is worth. The collections with trained policies hold about
  19,000 such windows.
- **The policy** sees the latest `policy_window` frames, each with the
  action before it. It picks one of `action_bins` int8 actions,
  `action_step` apart and centred on 0 (−64 … 64 in `base.toml`, just past
  the ±60 of the recordings).
- **Each imagined frame** is the world model's prediction. Each tag is
  dropped as unseen with the probability the model gives, so the policy
  meets misses as it will on the rig.
- **The reward** is the sum of the cosines of the tags' yaws, each turned
  so that its hanging yaw reads 180°: a hanging arm scores −1 and an
  upright one +1. The hanging yaws (`hanging_yaws`) are measured over the
  ends of the recorded rests, since a tag mounted slightly turned reads a
  few degrees off 180° there; they are saved in the policy checkpoint. The
  world model still works in raw camera yaws.
- **The speed penalty**: each arm turning faster than its limit in
  `speed_limits_rev_s` costs `speed_penalty` times the square of the
  excess in revolutions per second, per frame, measured between
  consecutive imagined frames. The limits (2, 2.5 and 6.5 rev/s in
  `base.toml`) sit a little under the random walk's 90th percentiles, the
  speeds the world model has data for; the first arm's matches the
  harness's speed governor. The policy maximises the reward net of it.

```sh
uv run python -m imagination.train imagination/configs/base.toml --name first
uv run python -m imagination.train imagination/configs/base.toml --name smoke --wandb disabled --iterations 15
uv run python -m imagination.train --resume runs/policy/first/checkpoints/iter_000100.pt
```

`--set KEY=VALUE` overrides one config value for a new run (VALUE as
TOML, e.g. `--set gamma=0.998`), for experiments that change one thing.

The evaluation during training is one rollout, which a chaotic system
makes noisy. `imagination.evaluate` runs many from hanging (64 by
default), each with its own tag misses from one seeded generator, and
reports the mean sum of cosines with its standard error, the upright
share, the speed penalty and the first arm's speeds, for several policies
in one world model:

```sh
uv run python -m imagination.evaluate runs/policy/*/policy.pt --world-model runs/world_model/wm3/model.pt
```

Every `checkpoint_every` iterations, and at the end, the whole training
state (policy and critic, optimizer, schedule, iteration, the
environments, random generators, times and W&B run id) goes to
`runs/policy/<name>/checkpoints/iter_NNNNNN.pt`, and a copy to
`runs/policy/<name>/policy.pt`. `--resume` continues from a checkpoint
with the config it saved, in the same directory and W&B run, and
repeats what an uninterrupted run would have done. W&B does not take a
step twice, so iterations it already logged past the checkpoint are not
logged again.

`world_model` in the config names the checkpoint to use. Every
`eval_every` iterations the checkpoint goes to
`runs/policy/<name>/policy.pt` and the policy is evaluated: one rollout of
`eval_steps` frames (10 s) from the pendulum hanging still, at each tag's
mean yaw at the end of the recorded rests. The policy acts greedily. Tag
misses are drawn from the world model's probabilities with a generator
seeded by `eval_seed`, so evaluating the same policy twice on the same
device gives the same numbers.

At the end of training, the last evaluation's rollout becomes a video,
`runs/policy/<name>/rollout.mp4`, logged to W&B as `rollout_from_hanging`.
To make one from any checkpoint:

```sh
uv run python -m imagination.video runs/policy/ppo-base/policy.pt
uv run python -m imagination.video runs/policy/ppo-base/policy.pt --seconds 20 --sample --world-model runs/world_model/base/model.pt --out spin.mp4
```

The video draws three equal arms chained from the pivot, each at its tag's
yaw, turned right side up, with an arm faded in frames where its tag was
dropped as unseen, and the action below. Every frame is shown, played 4
times slower than real time (`--slowdown`).

To run a policy in the harness, export it as JSON:

```sh
uv run python -m imagination.export runs/policy/ppo-base/policy.pt
```

The export (`policy.json` next to the checkpoint) holds the actor's layers,
the actions and the window, and test cases: histories of frames as the
harness sees them, with the logits the policy gives for each, computed by
writing each history as a frames table and reading it back as training
data. The harness recomputes them when it loads the policy; see its
README, "Trained policies".

### Metrics

Rewards are per frame: the sum of the three arms' corrected cosines, from
−3 (all hanging) to +3 (all upright), less any speed penalty. Values and returns are in the
critic's units, discounted sums of rewards scaled by 1 − `gamma`, so they
share that range: a policy that holds 0 per frame is worth about 0.

Logged every iteration, from its `rollout_steps` frames in each of the
`num_envs` environments, with actions sampled from the policy:

- `train/reward`: the mean reward per frame over the iteration, the one
  the policy maximises: `train/cos_sum` less `train/speed_penalty`.
- `train/cos_sum`: the mean sum of the arms' cosines per frame, how high
  the pendulum is, whatever else the reward counts. Without speed limits
  it equals `train/reward`.
- `train/speed_penalty`: the mean speed penalty per frame.
- `train/resets`: how many environments started again from a recorded
  window during the iteration.
- `train/upright`: the share of frames with every arm within 30° of
  upright.
- `train/value_mean`, `train/return_mean`: the mean of the critic's values
  over the iteration, and of the return targets it is trained towards
  (GAE's advantages plus the values).
- `train/explained_variance`: 1 − Var(target − value) / Var(target), with
  the values from before the update. 1 means the critic predicts its
  targets, 0 no better than a constant. The targets lean on the critic's
  own later values, so a critic can score well here while wrong about the
  long run; `eval/value_error` checks that.

Averaged over the minibatch updates of each iteration:

- `train/pg_loss`: PPO's clipped policy loss.
- `train/v_loss`: half the critic's squared error against the return
  targets.
- `train/entropy`: the entropy of the policy's action distribution, in
  nats: ln 9 ≈ 2.20 when uniform over the 9 bins, 0 when certain.
- `train/approx_kl`: an estimate of how far each update moved the policy
  from the one that collected the iteration's frames.
- `train/clipfrac`: the share of samples whose probability ratio left
  1 ± `clip`, where the loss stops rewarding the change.

And, for the run itself:

- `lr`: the learning rate, following the trapezoid over iterations.
- `time/<phase>_s`: seconds spent this iteration in each phase. The
  device is synchronised at each phase's start and end, so the times are
  the work done, not merely queued.
  - `time/rollout_s`: collecting the iteration's frames: per frame, the
    policy's and critic's forward passes, the world model's step, and
    restarting the environments that reset.
  - `time/advantages_s`: GAE over the iteration's frames.
  - `time/update_s`: PPO's `update_epochs` passes over the batch in
    `minibatches` minibatches, and the schedule's step.
  - `time/eval_s`: the evaluation rollout and its metrics, on iterations
    that evaluate.
  - `time/checkpoint_s`: writing the checkpoint and `policy.pt`, on
    iterations that save one.
- `time/iteration_s`: the sum of this iteration's phases. It leaves out
  computing and logging the training metrics, which is small.
- `time/share/<phase>`: each phase's share of all the time timed since
  the run began, carried across resumes; `time/share/eval` answers what
  the evaluation costs relative to training.
- `time/total_s`: that time, in seconds.

Logged at each evaluation (see above):

- `eval/reward`: the mean reward per frame over the rollout, the one the
  policy maximises: `eval/cos_sum` less `eval/speed_penalty`.
- `eval/cos_sum`: the mean sum of the arms' cosines per frame. Without
  speed limits it equals `eval/reward`.
- `eval/speed_penalty`: the mean speed penalty per frame.
- `eval/speed/arm<i>`: each arm's median speed, in revolutions per second.
- `eval/upright`: the share of its frames with every arm within 30° of
  upright.
- `eval/mean_abs_action`: the mean |action|, in int8 units (64 means always
  at a limit).
- `eval/value_hanging`: the critic's value of the hanging start.
- `eval/value_error`, `eval/value_bias`: the mean absolute and mean signed
  difference between the critic's value along the rollout and the
  discounted return, net of the speed penalty, the rollout actually
  earned from there. Only frames
  with enough of the rollout ahead to measure that return count (the
  discount falls to 1%): the first 2.6 s of 10 at `gamma` 0.995. The
  rollout is greedy while the critic values the sampling policy, so some
  bias is expected.
- `eval/value_vs_return`: the two plotted against time from hanging.

At the end of training:

- `rollout_from_hanging`: the video of the last evaluation's rollout.

## Training in the cloud

`cloud/` runs training jobs on EC2 instances of their own, so that
training need not share the Mac with the rig. `cloud/setup.sh`, run once,
creates the IAM role the instances run as and stores the W&B key from
`~/.netrc` in Secrets Manager. Then, from `learning/`:

```sh
uv run python cloud/launch.py bench bench-g6 --instance g6.xlarge --follow
uv run python cloud/launch.py world_model wm4 world_model/configs/base.toml --follow
uv run python cloud/launch.py policy ppo-wm4 imagination/configs/base.toml -- --set seed=1
```

`launch.py` packages the current commit (the working tree must be clean)
to `s3://allais-andrea-store/double_pendulum/code/`, and starts a spot
instance (`--on-demand` for one that is not) from AWS's Deep Learning Base
GPU AMI. On it, `cloud/job.sh` installs `uv`, syncs the data, fetches the
world model a policy config names from the bucket's `runs/`, trains, and
syncs the run directory and its log to
`s3://allais-andrea-store/double_pendulum/runs/<kind>/<name>/` every 5
minutes and at the end, with `job_status`. Then the instance terminates
itself, as it does after `--max-hours` (12) whatever happens. `--follow`
prints the log as it arrives.

A policy job launched again with the same name resumes from its latest
checkpoint there, which is how to carry on after a spot interruption. A
world model job starts over. A `bench` job times 100 policy iterations
and 5000 world model steps, without W&B, to compare instances with the
Mac.

A policy config's `world_model` must be in the bucket's `runs/` too:
`aws s3 sync runs/world_model/<name> s3://allais-andrea-store/double_pendulum/runs/world_model/<name>`.
