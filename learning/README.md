# learning

The Python learning code, as one uv project. Run everything from this
directory.

- `common/`: code the other packages share. `data_lib.py` loads the frames
  tables `collect` writes (schema in `harness/src/table.rs`) as steps;
  `render_lib.py` draws the pendulum from its yaws into videos, for real or
  imagined runs; `schedule_lib.py` is the trapezoidal learning-rate
  schedule; `run_lib.py` the device, timing and random-state helpers the
  trainers share; `testing_lib.py` builds small tables for tests.
- `world_model/`: an auto-regressive world model of the pendulum. Given a
  window of steps, it predicts the next frame's observation.
- `imagination/`: trains a policy with PPO, using the world model as the
  simulator, and compares world models' predictions with the rig.
- `cloud/`: runs training jobs on EC2 GPU instances.
- `docs/world_model_experiments.md`: the world model experiments so far
  (profiling, scaling, rollout training, closed-loop fidelity, policy
  ranking, the stochastic world model and its scaling, the loop of world
  model, policy and data, annealing the model's noise in policy training).
  Its figures are drawn by `docs/figures.py` from the CSVs in
  `docs/results`, which `docs/results.py` regenerates from W&B, the
  profiling logs and the rig recordings:
  `uv run python -m docs.results && uv run --group docs python docs/figures.py`.

**Two eras of data.** On 2026-09-30 a tag on the rig moved, and policies
trained on the earlier data stopped working. Recordings from `1790789841`
on are of the rig as it is now; everything before is of the rig as it
was. The two are not mixed: the `rebootstrap.toml` configs (world model
and policy) train on the new recordings alone, which the loop that
collects them names with `--set`. Every other config, the experiments
doc, and the ranking pools are of the earlier rig.

**The loop** (`imagination/loop.py`) collects on the rig while AWS
trains: the rig worker records sampled drives of the latest policy in
fixed-length chunks, staging each to `data/` and S3, and tests each new
policy greedily before collecting with it; the training worker trains a
world model on every new-era recording finished so far, then a policy in
it, exports it and checks the export. Its state and log are in
`runs/loop/`, and it resumes from them; its docstring has the options.

```sh
caffeinate -is uv run python -m imagination.loop                    # both workers
caffeinate -is uv run python -m imagination.loop --rig --chunks 8 --chunk-s 1800
```

Library modules end in `_lib`, and each has its tests next to it in
`<module>_test.py`. A package's run configs are in its `configs/`.

```sh
aws --profile andrea-personal s3 sync s3://allais-andrea-store/double_pendulum/data/ data/
uv run wandb login                                                 # once
uv run pytest                                                      # every *_test.py
```

## The world model

- **A step** is one camera frame (8 ms at 125 fps): the sine and cosine of
  each tag's yaw (its in-plane angle), whether each tag was seen, and the
  action in effect after the frame, divided by 128. A frame the camera
  dropped becomes a step with no tag seen and the action carried over.
- **The model** is an MLP (`hidden` wide, `layers` deep) over the
  flattened `window` of steps. Per tag, it outputs the change in sine and
  cosine since the tag was last seen in the window, and the logit of the
  tag going unseen in the next frame. A `stochastic` model also outputs
  the change's log-variance: a normal distribution with a diagonal
  covariance, from which imagination draws each frame, its spread scaled
  by `tau` (1; 0 is its mean).
- **The loss** is the mean squared error of the change, over the tags seen
  in the next frame, plus `bce_weight` times the cross-entropy of the
  missing logits; for a stochastic model, the change's negative
  log-likelihood instead, weighted by its variance to the power `nll_beta`
  (beta-NLL; 0 is the plain likelihood). The log-variance is soft-clamped
  above `logvar_min`. With `rollout_train` K above 1, it is averaged over K
  frames of rollout: the model predicts each frame from its own previous
  predictions, with gradients through the whole rollout
  (`model_lib.rollout_losses`). Training windows are drawn uniformly from
  every start in every training recording.
- **The learning rate** follows the trapezoid (`common/schedule_lib.py`):
  a linear warmup over `warmup_steps`, a constant rate, and a 1 − sqrt
  cooldown over the last `cooldown_fraction` of `steps` (or
  `cooldown_steps`). With no cooldown the rate stays constant, and
  `--branch` starts a cooldown from any checkpoint: one long run gives
  finished models at many lengths.

- **A two-stage model** (`TwoStageWorldModel`) is a stochastic model whose
  mean is a frozen deterministic `mean_model`, and whose variance is
  trained on the residuals of `mean_folds`: two more mean models, each
  trained on one half of the data (`train_fold` 0 or 1, in alternating
  blocks of `fold_block` steps), each window's residual coming from the
  one that did not see it. It was an attempt at the variance overfitting
  large stochastic models show; it did not beat the plain recipe.

- **A flow model** (`flow`, `world_model/flow_lib.py`) draws the next
  frame without a set distribution: the change in each tag's yaw (not in
  its sine and cosine, which given the last frame lie on a circle, where a
  density is degenerate), all three jointly, carried from standard normal
  noise along a velocity field the network learns by flow matching. The
  body encodes the window once; a head `flow_hidden` wide and
  `flow_layers` deep gives the velocity, integrated in `flow_steps`
  midpoint steps (2 × `flow_steps` head passes per frame). `tau` scales the
  starting noise; at 0 the flow starts from zero, which stands in for the
  mean. It trains one step ahead.

The configs:

- `base.toml` is the one-step deterministic recipe the early models used
  (256 × 3, batch 1024); `rollout3-k8`, 512 × 3 trained on 8-frame
  rollouts (`world_model/sweeps/rollout3/rollout3-k8.txt`), was the best
  deterministic model.
- `stochastic.toml` is the stochastic recipe of the scaling study: β-NLL
  (0.5), batch 4096, lr 2e-3, compiled and in bf16, gradients clipped at
  1, and every validation recording pooled into `val_all`. The model
  chosen from the study, `sc-w256-d3-cd200k`, is 256 × 3 trained for
  240k steps (the last 40k cooling down); the standard policy recipe
  trains in it.
- `stochastic-2.toml` and `stochastic-3.toml` are the same with the
  30-second-drive collections added (`s256-long-drives-2` and, at 512 × 3,
  `s512x3-long-drives-2`); neither scored better, open loop or through
  the policies trained in them.
- `rebootstrap.toml` is the recipe for the rig as it is now: 256 × 3,
  stochastic, no validation sets, its training recordings given with
  `--set train=[...]` (50k steps by default; the pipelined loop runs
  250k).
- `flow.toml` is a flow model of the rig as it is now, 256 × 3 with a
  256 × 2 head, on every collection since the tag moved (greedy tests left
  out) but two 20-minute collections of 2026-10-01, held out as `val_all`,
  for 250k steps;
  `--set flow=false` trains the rebootstrap's Gaussian on the same split,
  to compare with.

`docs/world_model_experiments.md` says why.

### Running

```sh
uv run python -m world_model.train world_model/configs/base.toml --name base
uv run python -m world_model.train world_model/configs/base.toml --name smoke --wandb disabled --steps 2000
uv run python -m world_model.train world_model/configs/base.toml --name k8 --set rollout_train=8 --set hidden=512
uv run python -m world_model.train --resume runs/world_model/k8/checkpoints/step_0050000.pt
uv run python -m world_model.train --branch runs/world_model/long/checkpoints/step_0050000.pt \
    --cooldown-steps 10000 --name long-cd50k
```

The config names the training recordings, and in `val` the validation
sets, each a list of recordings reported under its own name (an empty
table, `val = {}`, for none): `base.toml` has `val_random_walk` and one
set per policy collection, `val_policy` (ppo-speed), `val_policy_wm2` and
`val_policy_wm3`; `stochastic.toml` pools all nine validation recordings
of the earlier rig into `val_all`, the set to judge by, and keeps a few
of them as sets of their own. `random_walk_only.toml` trains on the
random-walk recordings alone, for comparison. `--set KEY=VALUE` overrides
a config value for a new run (VALUE as TOML); besides the config's own
keys it takes the optional `rollout_train` (1), `max_grad_norm` (clips
the gradient; long rollouts and wide stochastic models need it),
`compile` and `bf16` (training steps through `torch.compile`, and in
bfloat16 on CUDA; evaluation stays float32), `checkpoint_every`,
`checkpoint_at` (a list of steps), `cooldown_steps`, `stochastic`,
`nll_beta` (0) and `logvar_min`, `eval_stride` and `eval_ensemble` (see
Metrics), the two-stage keys `mean_model`, `mean_folds`,
`train_fold` and `fold_block`, and the flow keys `flow`, `flow_hidden`,
`flow_layers`, `flow_steps`, `nll_steps` and `eval_ensemble_stride`. Policies take a stochastic model's `tau`
from their config, and `imagination.ranking imagine` from `--tau`.

Every `checkpoint_every` steps, at the steps in `checkpoint_at`, and at
the end, the whole training state (model, optimizer, step, random
generators, times, W&B run id) goes to
`runs/world_model/<name>/checkpoints/step_NNNNNNN.pt`, and the weights to
`runs/world_model/<name>/model.pt`, which is what everything else loads.
`--resume` continues from a checkpoint in the same directory and W&B run,
repeating what an uninterrupted run would have done. `--branch` is a new
run, whose steps continue from the checkpoint's. A new run's directory
must not exist yet.

Tools around it:

```sh
uv run python -m world_model.scale world_model/configs/base.toml --name wm-w512-d3 \
    --steps 200000 --branches 12500 25000 50000 100000 200000 -- --set hidden=512
uv run python -m world_model.sweep world_model/sweeps/rollout3/rollout3-k8.txt
uv run python -m world_model.profile --widths 256 1024 --batches 1024 4096 --variants plain compile+bf16
uv run python -m world_model.report rollout3 --horizons 16 64 125
uv run python -m world_model.compare runs/world_model/wm3/model.pt runs/world_model/rollout3-k8/model.pt --val 1790464558
```

- `scale` trains one long constant-rate run and a cooldown branch from
  each of `--branches` (`<name>-cd<k>k`), and carries on after an
  interruption from whatever it finds.
- `sweep` runs a file of `world_model.train` argument lines in turn,
  skipping finished runs and resuming unfinished ones; `sweeps/` holds the
  batch, rollout and follow-up sweeps.
- `profile` times training steps across widths, depths, batch sizes and
  variants (plain, compile, bf16) on the real data; `--csv` also writes them.
  With `--flow`, it profiles the config's flow model, and times its whole
  evaluation, part by part, with its share of the time at `eval_every`.
- `report` tabulates the final metrics of every W&B run with a name
  prefix, on each validation set and its upright subset; `--csv` also
  writes them, with each run's training settings.
- `compare` evaluates several checkpoints on the same fixed windows of any
  recordings, so models trained on different data compare like for like;
  `--upright` keeps the windows starting upright, a stochastic model is
  also scored as an ensemble of `--ensemble` (8) sampled rollouts per
  window (1 − R² of its mean, CRPS over copying, spread over error), and
  `--csv` writes every metric.

### Metrics

Logged every `log_every` steps, on the batch just trained on:

- `train/batch_mse`: the squared error of the predicted change, in units
  of `delta_scale` (the typical one-frame change), over the tags seen in
  the next frame, averaged over the rollout when `rollout_train` > 1.
- `train/batch_bce`: the cross-entropy of the missing-tag logits.
- `train/batch_fm`: for a flow model, instead of `train/batch_mse`, the
  flow-matching loss: the squared error of the velocity, per element.
- `lr`: the learning rate.

Every `eval_every` steps, training pauses to evaluate on a fixed sample of
`eval_samples` windows from the training recordings (`train`) and from
each validation set (under its name): the same sample every time, drawn
from `seed`. With `eval_stride`, a validation set's windows are instead
every `eval_stride`-th start, uniform in time over its recordings, so a
pooled set weighs each recording by its length. From each window it
rolls the model forward open loop (a stochastic model on its mean) for
`max(horizons)` frames with the recorded actions, each prediction fed
back as seen, with its sine and cosine put back on the unit circle. For
each set `<set>`:

- `<set>/mse`, `<set>/bce`: the two loss terms at one step, as above.
- `<set>/one_minus_r2/hNNN`: 1 − R² at horizon N frames, where R²
  measures the predicted change against the actual change since each tag
  was last seen in the starting window, over the tags seen at that frame.
  0 is perfect.
- `<set>/copy_last/hNNN`: the same score for predicting no change, which
  comes out at about 1. It is the reference line.
- For a stochastic model, `<set>/nll` (the one-step loss as trained:
  β-NLL per element), `<set>/median_nll` (the plain NLL's median, which a
  few confident misses cannot move as they move the mean),
  `<set>/within_1sd`, `<set>/within_2sd` (the share of one-step changes
  within 1 and 2 predicted standard deviations: 0.683 and 0.954 if
  calibrated) and `<set>/median_sd`.
- For a flow model, instead of `mse` and the Gaussian's metrics,
  `<set>/fm` (the flow-matching loss, on noise from a fixed seed), and
  `<set>/nll` and `<set>/median_nll`: the negative log-likelihood of the
  next frame's yaw changes, per tag, in radians, over the windows whose
  next frame has every tag seen, integrated back through the flow in
  `nll_steps` (32) RK4 steps with the exact divergence. It is a density
  over angles, not over the sine and cosine, so it does not compare with
  the Gaussian's NLL; CRPS and 1 − R² do. Its rollouts follow the flow
  from zero noise.
- For a stochastic model, `<set>/ensemble/…`: `eval_ensemble` (8)
  sampled rollouts from each window (or from every
  `eval_ensemble_stride`-th), at τ = 1, scored as an ensemble:
  `one_minus_r2` of the ensemble's mean, `crps` (the continuous ranked
  probability score over copying the last frame's error: 1 is no better
  than copying, 0 perfect; a proper score, it rewards spread only where
  the future is uncertain) and `spread_skill` (the ensemble's spread over
  the error of its mean: 1 if calibrated, below 1 overconfident).
- `<set>/upright/…`: the same metrics on the set's windows whose last
  input frame has arms 0 and 1 seen and within 30° of upright, the regime
  the policies now live in (when at least 64 such windows are in the
  sample).
- `time/<phase>_s`, `time/share/<phase>`, `time/total_s`: seconds in
  training, evaluation and checkpointing since the last evaluation, their
  shares of all the time so far, and that time, as in policy training.

Because the change being predicted grows with the horizon, 1 − R² need not
increase monotonically with it. Compare one horizon across runs rather than
reading it as an error curve. Open-loop 1 − R² does not by itself pick the
best world model to train policies in; `imagination.sim2real` and
`imagination.ranking` below do.

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
- **Each imagined frame** is the world model's prediction: for a
  stochastic model, a draw with its spread scaled by `tau` (0 is the
  mean). With `tau_start` and `tau_end`, tau rises linearly from one to
  the other over the run, for training and evaluation alike: the policy
  learns to balance in the calm mean model, then meets the model's noise.
  Each tag is dropped as unseen with the probability the model gives, so
  the policy meets misses as it will on the rig.
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

The configs:

- `base.toml` trains in `wm3` for 2000 iterations at lr 1e-3, which found
  the balancing solution in about an hour on the Mac; `wm-k1.toml`,
  `wm-k6.toml` and `wm-k8.toml` are the same in the deterministic models
  of the ranking study, and `wm-stoch.toml` in the first stochastic one.
- `wm-s256-anneal.toml` is the standard recipe for the earlier rig: the
  256 × 3 stochastic model `sc-w256-d3-cd200k`, tau annealed from 0 to 1,
  15 actions 18 apart (±126, the motor board's whole range), actor and
  critic each 512 × 3, 5000 iterations. Its policy,
  `ppo-s256-anneal-512x3`, scored +2.42 on the rig's 30-second drives.
  The doc's last section has what else was tried (entropy, episode
  length, gamma, the input window).
- `rebootstrap.toml` is the recipe for the rig as it is now: the same,
  but 29 actions 9 apart (±126), and by default 256 × 2 for 2000
  iterations; the loop passes `world_model` and `recordings` with `--set`,
  and for its larger policies `hidden=512`, `layers=3` and
  `iterations=5000`.

```sh
uv run python -m imagination.train imagination/configs/wm-s256-anneal.toml --name first
uv run python -m imagination.train imagination/configs/base.toml --name smoke --wandb disabled --iterations 15
uv run python -m imagination.train imagination/configs/wm-k8.toml --name longer --iterations 5000 --set seed=1
uv run python -m imagination.train imagination/configs/wm-s256-anneal.toml --name warm \
    --init runs/policy/first/checkpoints/iter_003000.pt
uv run python -m imagination.train --resume runs/policy/first/checkpoints/iter_000100.pt
```

`world_model` in the config names the world model checkpoint.
`--set KEY=VALUE` overrides one config value for a new run (VALUE as
TOML, e.g. `--set gamma=0.998`), for experiments that change one thing.
`--init` starts a new run from another run's policy and critic (a
`policy.pt` or a checkpoint), which must have the same inputs and
actions; the optimizer, schedule and environments start afresh.

Every `checkpoint_every` iterations, and at the end, the whole training
state (policy and critic, optimizer, schedule, iteration, the
environments, random generators, times and W&B run id) goes to
`runs/policy/<name>/checkpoints/iter_NNNNNN.pt`, and a copy to
`runs/policy/<name>/policy.pt`. `--resume` continues from a checkpoint
with the config it saved, in the same directory and W&B run, and
repeats what an uninterrupted run would have done. W&B does not take a
step twice, so iterations it already logged past the checkpoint are not
logged again.

Every `eval_every` iterations the policy is evaluated: one rollout of
`eval_steps` frames (10 s) from the pendulum hanging still, at each tag's
mean yaw at the end of the recorded rests. The policy acts greedily. Tag
misses are drawn from the world model's probabilities with a generator
seeded by `eval_seed`, so evaluating the same policy twice on the same
device gives the same numbers. At the end of training, the last
evaluation's rollout becomes a video, `runs/policy/<name>/rollout.mp4`,
logged to W&B as `rollout_from_hanging`. With `eval_rollouts` N above 1,
N more rollouts from hanging, from the same seed, give `eval/mean/cos_sum`
with its standard error and `eval/mean/upright`: in a stochastic world
model one rollout is one draw.

### Evaluating, exporting and watching policies

```sh
uv run python -m imagination.evaluate runs/policy/*/policy.pt --world-model runs/world_model/rollout3-k8/model.pt
uv run python -m imagination.sim2real runs/world_model/rollout3-k8/model.pt runs/world_model/wm3/model.pt
uv run python -m imagination.video runs/policy/ppo-k8/policy.pt
uv run python -m imagination.video runs/policy/ppo-k8/policy.pt --seconds 20 --sample --world-model runs/world_model/wm3/model.pt --out k8-in-wm3.mp4
uv run python -m imagination.export runs/policy/ppo-k8/policy.pt
```

- `evaluate` runs many rollouts from hanging (64 by default), each with
  its own tag misses from one seeded generator, and reports the mean sum
  of cosines with its standard error, the upright share, the speed
  penalty and the first arm's speeds, for several policies in one world
  model. The evaluation during training is a single rollout, which a
  chaotic system makes noisy.
- `sim2real` compares world models' predictions for the four policies
  tested greedily on the rig (ppo-persistent, ppo-speed, ppo-wm2,
  ppo-wm3) with their recordings: the mean sum of cosines over the first
  10 s from hanging, per policy, and per world model the mean error, the
  correlation, and whether it ranks them as the rig does; `--csv` also
  writes the table.
- `video` draws a closed-loop rollout from hanging: three equal arms
  chained from the pivot, each at its tag's yaw, turned right side up, an
  arm faded in frames where its tag was dropped as unseen, and the action
  below. Every frame is shown, played 4 times slower than real time
  (`--slowdown`).
- `export` writes a policy for the harness, as `policy.json` next to the
  checkpoint: the actor's layers, the actions and the window, and test
  cases, histories of frames as the harness sees them with the logits the
  policy gives for each, computed by writing each history as a frames
  table and reading it back as training data. The harness recomputes them
  when it loads the policy; see its README, "Trained policies".

### Metrics

Rewards are per frame: the sum of the three arms' corrected cosines, from
−3 (all hanging) to +3 (all upright), less any speed penalty. Values and
returns are in the critic's units, discounted sums of rewards scaled by
1 − `gamma`, so they share that range: a policy that holds 0 per frame is
worth about 0.

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
  nats: ln `action_bins` when uniform (ln 15 ≈ 2.71, ln 29 ≈ 3.37), 0
  when certain.
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
- `eval/mean_abs_action`: the mean |action|, in int8 units (the largest
  action level means always at a limit).
- `eval/value_hanging`: the critic's value of the hanging start.
- `eval/value_error`, `eval/value_bias`: the mean absolute and mean signed
  difference between the critic's value along the rollout and the
  discounted return, net of the speed penalty, the rollout actually
  earned from there. Only frames with enough of the rollout ahead to
  measure that return count (the discount falls to 1%): the first 2.6 s
  of 10 at `gamma` 0.995. The rollout is greedy while the critic values
  the sampling policy, so some bias is expected.
- `eval/value_vs_return`: the two plotted against time from hanging.

At the end of training:

- `rollout_from_hanging`: the video of the last evaluation's rollout.

## Ranking world models against the rig

A world model is good to train policies in if it ranks policies the way
the rig does (after SIMPLER, arXiv:2405.05941). The pool,
`imagination/ranking/pool.txt`, holds the policies tested so far, from
+0.08 to about +2.3 on the rig; each line is a name, a checkpoint, and
whether the policy was on the rig before the pool was made (`seen` or
`new`). `pool-2.txt` and `pool-3.txt` name the policies of the second
and third sessions, for `collect.sh`.

```sh
uv run python -m imagination.ranking export                # the pool, as runs/ranking/<name>.json
learning/imagination/ranking/collect.sh 120 0              # from the repository root, at the rig
learning/imagination/ranking/collect.sh 120 1              # a second pass, in another order
learning/imagination/ranking/collect.sh 300 0 POOL 30 8    # 5 minutes each, 30 s drives, 8 s rests
uv run python -m imagination.ranking imagine ../recordings/ranking-*-seed0.tsv ../recordings/ranking-*-seed1.tsv \
    --world-models runs/world_model/sc-w256-d3-cd200k/model.pt runs/world_model/rollout3-k8/model.pt@0 \
    --scores runs/ranking/scores.csv
uv run python -m imagination.ranking agreement runs/ranking/scores.csv   # the metrics again, no rollouts
```

`collect.sh [SECONDS] [SEED] [POOL] [ACTIVE] [REST]` runs each policy in
POOL greedily for SECONDS (120), driving ACTIVE s (10) and resting REST s
(5), in an order shuffled with SEED, without the speed governor; two
passes in different orders average out drift over a session; `kill -INT
$(cat recordings/ranking.pid)` stops it. `imagine` takes any number of
manifests, pools each policy's recordings, splits them into their drives
after a rest, on each recording's own duty cycle (read from its
metadata), drops the first, and scores each on the rig (the mean sum of
cosines over the drive, or its first `--seconds`) and in each world
model, from rollouts started at the same recorded frames, greedy or
sampled as the recording was. A stochastic world model draws at `--tau`
(1), or at the TAU of a `PATH@TAU` entry. It prints each policy's rig
score with its standard error and the predictions, and writes them, with
the world model each policy trained in, to `--scores`. This is the slow
part: about 40 s per world model for 13 policies on the Mac.

`agreement` reads those scores and prints, per world model, the Pearson
and Spearman correlations, the mean maximum rank violation (MMRV), the
mean absolute error and the bias (mean overrating), over all policies,
those new to the rig and, for a world model some policies were trained
in, the others (held out) and those (own), with its exploitation gap: how
much more it overrates its own policies than the others. `imagine` ends
by printing the same; `--csv` also writes the table.

## Training in the cloud

`cloud/` runs training jobs on EC2 instances of their own, so that
training need not share the Mac with the rig. `cloud/setup.sh`, run once,
creates the IAM role the instances run as and stores the W&B key from
`~/.netrc` in Secrets Manager. Then, from `learning/`:

```sh
uv run python cloud/launch.py bench bench-g6 --instance g6.xlarge --follow
uv run python cloud/launch.py world_model wm4 world_model/configs/base.toml --follow
uv run python cloud/launch.py policy ppo-wm4 imagination/configs/wm-k8.toml -- --set seed=1
uv run python cloud/launch.py sweep rollout3-k8 world_model/sweeps/rollout3/rollout3-k8.txt
uv run python cloud/launch.py scale wm-w512-d3 world_model/configs/base.toml -- \
    --steps 200000 --branches 12500 25000 50000 -- --set hidden=512
uv run python cloud/launch.py run profile-g6 world_model.profile -- --batches 1024 4096
uv run python cloud/queue.py cloud/queues/scaling.txt --spot 2 --on-demand 1
```

`launch.py` packages the current commit (the working tree must be clean)
to `s3://allais-andrea-store/double_pendulum/code/`, and starts a spot
instance (`--on-demand` for one that is not) from AWS's Deep Learning Base
GPU AMI, trying each of `--instance`'s types (g6, g5 and g6e xlarge by
default) in each of the account's public subnets until one has capacity.
Arguments after `--` go to the job. On the instance, `cloud/job.sh`
installs `uv`, syncs the data, fetches from the bucket's `runs/` the
world models the config or a `--set` names and a policy's `--init`
checkpoint, trains, and syncs the run directory and
its log to `s3://allais-andrea-store/double_pendulum/runs/<kind>/<name>/`
every 5 minutes and at the end, with `job_status`. Then the instance
terminates itself, as it does after `--max-hours` (12) whatever happens.
`--follow` prints the log as it arrives.

The kinds of job: `world_model` and `policy` train one run; `sweep` runs a
`world_model.sweep` file and `scale` a `world_model.scale` model, whose
runs must be named `<name>-*` and go to `runs/world_model/`; `run` runs
any module (`python -m CONFIG ARGS`); `bench` times 100 policy iterations
and 5000 world model steps without W&B. A job launched again with the
same name carries on from its checkpoints on S3, which is how to recover
from a spot interruption.

A job is pinned to the commit it was first launched with, recorded in
`job.json` in its run directory: launched again, as the queue does after
an interruption, it runs that commit's code, whatever has been committed
since, and needs no clean working tree. `--code HEAD` (or a commit) runs
and pins other code instead, to carry on under a fix, say.

`queue.py` runs a file of such jobs (`cloud/queues/`), keeping `--spot`
spot and `--on-demand` on-demand instances busy: it starts pending jobs
on spot while slots are free, else on demand, waits on any already
running (found by its instance's name tag, so a queue stopped and
started again picks up where it was), starts again any whose instance
vanished without a status, and leaves failed ones alone. It launches
every new job at one commit, HEAD when it starts or `--code`. Spot
capacity for these GPUs is often unavailable, and the account's
on-demand quota (8 vCPUs) runs two xlarge instances at a time; policy
queues add `g4dn.xlarge` (T4, slower) to `--instance` as a fallback.

A policy config's `world_model` must be in the bucket's `runs/` too:
`aws s3 sync runs/world_model/<name> s3://allais-andrea-store/double_pendulum/runs/world_model/<name>`.
