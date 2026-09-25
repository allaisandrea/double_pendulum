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
`runs/<name>/`, which must not exist yet.

### Metrics

Every `eval_every` steps, training pauses to evaluate on a fixed sample of
windows from each split: the same sample every time, drawn from `seed`.
From each window it rolls the model forward open loop for `max(horizons)`
frames with the recorded actions. Each prediction is fed back as seen,
with its sine and cosine put back on the unit circle.

- `{train,val}/one_minus_r2/hNNN`: 1 − R² at horizon N, where R² measures
  the predicted change against the actual change since each tag was last
  seen in the starting window. 0 is perfect.
- `{train,val}/copy_last/hNNN`: the same score for predicting no change,
  which comes out at about 1. It is the reference line.
- `{train,val}/mse`, `{train,val}/bce`: the two loss terms at one step.

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
```

`world_model` in the config names the checkpoint to use. Every
`eval_every` iterations the checkpoint goes to `runs/<name>/policy.pt` and
the policy is evaluated: one rollout of `eval_steps` frames (10 s) from
the pendulum hanging still, at each tag's mean yaw at the end of the
recorded rests. The policy acts greedily. Tag misses are drawn from the
world model's probabilities with a generator seeded by `eval_seed`, so
evaluating the same policy twice on the same device gives the same
numbers.

- `eval/reward`: the mean reward per frame, from −3 (all hanging) to +3.
- `eval/upright`: the share of frames with every tag within 30° of upright.
- `eval/mean_abs_action`: the mean |action|, in int8 units.

At the end of training, the last evaluation's rollout becomes a video,
`runs/<name>/rollout.mp4`, logged to W&B as `rollout_from_hanging`. To make one from any checkpoint:

```sh
uv run python -m imagination.video runs/ppo-base/policy.pt
uv run python -m imagination.video runs/ppo-base/policy.pt --seconds 20 --sample --world-model runs/base/model.pt --out spin.mp4
```

The video draws three equal arms chained from the pivot, each at its tag's
yaw, turned right side up, with an arm faded in frames where its tag was
dropped as unseen, and the action below. Every frame is shown, played 4
times slower than real time (`--slowdown`).
