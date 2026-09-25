# learning

The Python learning code, as one uv project. Run everything from this
directory.

- `common/`: code the other packages share. `data_lib.py` loads the frames
  tables `collect` writes (schema in `harness/src/table.rs`) as steps;
  `testing_lib.py` builds small ones for tests.
- `world_model/`: an auto-regressive world model of the pendulum. Given a
  window of steps, it predicts the next frame's observation.

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
uv run python -m world_model.train world_model/configs/base.toml
uv run python -m world_model.train world_model/configs/base.toml --wandb disabled --steps 200   # smoke test
uv run pytest                                                      # every *_test.py
```

The config names the training and validation recordings by directory
name. The run's evaluation sample and latest checkpoint go to
`runs/<run name>/`.

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
