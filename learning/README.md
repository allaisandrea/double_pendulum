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
name. The run's evaluation sample and latest checkpoint go to
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

Because the change being predicted grows with the horizon, 1 − R² need not
increase monotonically with it. Compare one horizon across runs rather than
reading it as an error curve.

## Training in imagination

`imagination/train.py` runs PPO (after CleanRL's
`ppo_continuous_action_isaacgym.py`) in a batch of environments that the
world model steps.

- **An episode** starts from a real window of the recordings and runs
  `episode_steps` frames. Episodes end only by truncation, so returns are
  bootstrapped from the critic's value of the state after the last step.
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

```sh
uv run python -m imagination.train imagination/configs/base.toml --name first
uv run python -m imagination.train imagination/configs/base.toml --name smoke --wandb disabled --iterations 15
uv run python -m imagination.train --resume runs/policy/first/checkpoints/iter_000100.pt
```

Every `checkpoint_every` iterations, and at the end, the whole training
state (policy and critic, optimizer, schedule, iteration, random
generators, times and W&B run id) goes to
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

### Metrics

Rewards are per frame: the sum of the three arms' corrected cosines, from
−3 (all hanging) to +3 (all upright). Values and returns are in the
critic's units, discounted sums of rewards scaled by 1 − `gamma`, so they
share that range: a policy that holds 0 per frame is worth about 0.

Logged every iteration, from its `num_envs` episodes of `episode_steps`
frames, started from random windows of the recordings, with actions
sampled from the policy:

- `train/reward`: the mean reward per frame over the episodes.
- `train/reward_last_step`: the mean reward on their last frame, once the
  policy has had the episode to act.
- `train/upright`: the share of frames with every arm within 30° of
  upright.
- `train/value_mean`, `train/return_mean`: the mean of the critic's values
  over the episodes, and of the return targets it is trained towards
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
  from the one that collected the episodes.
- `train/clipfrac`: the share of samples whose probability ratio left
  1 ± `clip`, where the loss stops rewarding the change.

And, for the run itself:

- `lr`: the learning rate, following the trapezoid over iterations.
- `time/<phase>_s`: seconds spent this iteration in each phase. The
  device is synchronised at each phase's start and end, so the times are
  the work done, not merely queued.
  - `time/rollout_s`: collecting the episodes: resetting the
    environments, and per frame the policy's and critic's forward passes
    and the world model's step.
  - `time/advantages_s`: GAE over the episodes.
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

- `eval/reward`: the mean reward per frame over the rollout.
- `eval/upright`: the share of its frames with every arm within 30° of
  upright.
- `eval/mean_abs_action`: the mean |action|, in int8 units (64 means always
  at a limit).
- `eval/value_hanging`: the critic's value of the hanging start.
- `eval/value_error`, `eval/value_bias`: the mean absolute and mean signed
  difference between the critic's value along the rollout and the
  discounted return the rollout actually earned from there. Only frames
  with enough of the rollout ahead to measure that return count (the
  discount falls to 1%): the first 2.6 s of 10 at `gamma` 0.995. The
  rollout is greedy while the critic values the sampling policy, so some
  bias is expected.
- `eval/value_vs_return`: the two plotted against time from hanging.

At the end of training:

- `rollout_from_hanging`: the video of the last evaluation's rollout.
