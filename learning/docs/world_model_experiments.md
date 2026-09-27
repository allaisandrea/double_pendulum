# Double pendulum world model experiments

As of 2026-09-27. Shared version, with charts: <https://claude.ai/code/artifact/4ffc27df-3c87-4337-a18b-35ba2bf28011>

## Summary

The best world model for training policies is a 512 × 3 MLP trained on 8-frame rollouts of its own predictions: it predicts four policies' real closed-loop results within 0.05 on average. The policy trained in it, `ppo-k8`, is forecast at about +2.2 sum of cosines per frame on the rig, against +1.22 for the current best, `ppo-wm3`.

- **Scaling helps little.** Across 8 sizes (0.18M to 17M parameters) and 5 training lengths, short-horizon error improved about 15%; larger models overfit the 5 hours of data. No size fixed the long horizon near upright.
- **Rollout training is the big lever for open-loop accuracy.** Training on K-frame rollouts cut the 0.5 s error from upright from 1.59 (worse than copying the last frame) to 0.32 at K = 32.
- **Open-loop accuracy is not what policies need.** One-step models overrate a balancing policy; long-rollout models underrate it, because they learn to predict the average future. K = 8 sits at the crossover, and a second seed agrees.
- **Use the closed-loop comparison (`imagination.sim2real`) to choose world models**, not 1 − R² alone.

## Setup

All models predict the next camera frame (8 ms) from a 16-frame window of each arm's yaw (sine and cosine), which tags were seen, and the motor action.

**Data**: 5.2 hours of training recordings, and three policy validation sets plus one random-walk set.

| Recordings | Collected with | Training | Validation set |
| --- | --- | --- | --- |
| 1790282998, 1790279628 | Random walk | 35 min | — |
| 1790281442 | Random walk | — | val_random_walk (10 min) |
| 1790372784, 1790374602 | ppo-speed, greedy | 29 min | val_policy (5 min) |
| 1790378922, 1790378604 | ppo-wm2, sampled | 116 min | val_policy_wm2 (5 min) |
| 1790464876, 1790464558 | ppo-wm3, sampled (balancing) | 116 min | val_policy_wm3 (5 min) |

**Open-loop metric**: 1 − R² of the predicted change at a horizon of 16, 64 or 125 frames, rolling the model on its own predictions with the recorded actions. 0 is perfect; copying the last frame scores about 1. Each set is also scored on its windows that start with arms 0 and 1 within 30° of upright ("upright"), the regime the policies now live in.

**Closed-loop metric**: for four policies tested greedily on the rig from hanging, each world model's predicted sum of cosines over the first 10 s against the real one (`imagination.sim2real`).

**Infrastructure**: an L4 on-demand GPU, spot L4 and L40S GPUs when available, and the Mac, driven by a job queue (`cloud/queue.py`). Long runs checkpoint and resume on S3 (two spot interruptions were survived), and one constant-rate run per model size branches into cooldowns at five lengths (`world_model.scale`).

## Profiling and training settings

An L4 GPU with batch 4096, `torch.compile` and bf16 trains 6–13× faster than the Mac, and batch 4096 at lr 2e-3 matches batch 1024 per window seen.

Milliseconds per training step (`world_model.profile`):

| Model (parameters) | Batch | Mac (M4) | L4 | L4, compile + bf16 | T4 |
| --- | --- | --- | --- | --- | --- |
| 256 × 3 (0.18M) | 1,024 | 2.8 | 2.1 | 2.2 | 2.6 |
| 256 × 3 | 16,384 | 15.4 | 3.8 | 2.4 | 6.9 |
| 1024 × 3 (2.3M) | 4,096 | 27.9 | 7.5 | 2.9 | — |
| 1024 × 5 (4.4M) | 4,096 | 51.2 | 14.0 | 4.6 | 27.4 |

- At batch 1024 every device is bound by kernel launches (about 2 ms); large batches and compilation are where the GPU pays.
- The T4 lacks bf16, which makes it slower; it was dropped from the instance list.
- Evaluation of one set (4,096 windows, 125 frames) takes 0.2 s on the Mac and 0.06 s on the L4.

The batch sweep (`sweeps/batch.txt`) trained the 256 × 3 model on 100M windows each (1 − R² on val_policy_wm3):

| Batch | lr | Precision | 16 frames | 64 frames |
| --- | --- | --- | --- | --- |
| 1,024 | 1e-3 | fp32 | 0.0233 | 0.701 |
| 4,096 | 2e-3 | fp32 | 0.0235 | 0.712 |
| 4,096 | 2e-3 | bf16 | 0.0226 | 0.725 |
| 16,384 | 4e-3 | fp32 | 0.0256 | 0.722 |

All later runs use batch 4096, lr 2e-3, compile and bf16 on the GPU.

## Phase 1: scaling model size and training length

With one-step training, the best model was 512 × 3 after 205M windows (0.0199 at 16 frames), about 15% better than the original 175k-parameter model; nothing fixed the long horizon near upright.

Each of 8 sizes (widths 256–2048, depths 3 and 5) trained one 200k-step constant-rate run, with cooldown branches at 12.5k to 200k steps (`cloud/queues/scaling.txt`): 40 finished models for about 1.4× the cost of the longest runs.

1 − R² at 16 frames on val_policy_wm3 (lower is better), by training windows:

| Width × depth | 51M | 102M | 205M | 410M | 819M |
| --- | --- | --- | --- | --- | --- |
| 256 × 3 | 0.0242 | 0.0224 | 0.0217 | 0.0212 | 0.0213 |
| 256 × 5 | 0.0257 | 0.0238 | 0.0222 | 0.0223 | 0.0217 |
| 512 × 3 | 0.0227 | 0.0207 | **0.0199** | 0.0205 | 0.0206 |
| 512 × 5 | 0.0232 | 0.0216 | 0.0224 | 0.0222 | 0.0234 |
| 1024 × 3 | 0.0221 | 0.0208 | 0.0205 | 0.0224 | 0.0227 |
| 1024 × 5 | 0.0227 | 0.0221 | 0.0220 | 0.0229 | 0.0277 |
| 2048 × 3 | 0.0212 | 0.0214 | 0.0222 | 0.0236 | 0.0245 |
| 2048 × 5 | 0.0221 | 0.0213 | 0.0221 | 0.0245 | 0.0272 |

- Small models improve with length to the end; models of 2M parameters and more peak at 51–102M windows, then get worse as they overfit.
- On val_random_walk the same pattern holds: the largest models go from about 0.036 to 0.056 at 16 frames over the longer runs.
- From upright, 0.5 s ahead, every one of the 40 models scores 1 − R² between 1.54 and 1.64: worse than predicting no change. Capacity and length do not fix it.

## Rollout training

Training on K-frame rollouts of the model's own predictions (`rollout_train`), with gradients through the whole rollout, cut the 0.5 s error from upright up to 5×; longer rollouts help the long horizon and cost some short-horizon accuracy.

All runs: 512 × 3, 50k steps (205M windows), batch 4096, lr 2e-3, unless noted. 1 − R² on val_policy_wm3, lower is better; "from upright" is its upright subset.

| Run | 16 frames | 64 frames | 125 frames | From upright, 64 | From upright, 125 | Random walk, 16 |
| --- | --- | --- | --- | --- | --- | --- |
| K = 1 (one-step) | 0.0201 | 0.689 | 0.998 | 1.594 | 1.618 | 0.040 |
| K = 4 | 0.0156 | 0.658 | 0.982 | 1.543 | 1.649 | 0.031 |
| K = 8 | 0.0114 | 0.549 | 0.973 | 1.261 | 1.636 | 0.035 |
| K = 12 | 0.0098 | 0.315 | 0.876 | 0.687 | 1.450 | 0.041 |
| K = 16 | 0.0101 | 0.258 | 0.797 | 0.520 | 1.298 | 0.044 |
| K = 32 | 0.0131 | 0.191 | 0.712 | 0.317 | 1.126 | 0.063 |
| K = 32, gradient clipping | 0.0144 | 0.200 | 0.678 | 0.351 | 1.073 | 0.069 |
| K = 64, clipping, lr 1e-3 | 0.0191 | 0.173 | 0.522 | 0.287 | 0.815 | 0.091 |
| K = 64, lr 2e-3 | did not train | | | | | |
| K = 16, 100k steps | 0.0116 | 0.275 | 0.824 | 0.576 | 1.349 | 0.045 |
| K = 16, width 1024 | 0.0112 | 0.272 | 0.797 | 0.525 | 1.286 | 0.046 |
| K = 16, 32-frame window (Mac) | 0.0102 | 0.261 | — | 0.496 | — | 0.057 |

- K = 12–16 give the best short horizon (about 2× better than one-step); K = 64 gives the best long horizon, and is the only model below 1 at 1 s from upright.
- K = 64 diverged at lr 2e-3; gradient clipping at norm 1.0 (`max_grad_norm`) and lr 1e-3 made it train. Clipping changes little at K = 32.
- At K = 16, more steps, width or history add nothing: data and objective, not capacity, limit these models.

## Closed-loop fidelity

The model trained on 8-frame rollouts predicts how policies really perform on the rig: +1.19 for the balancing policy ppo-wm3 against +1.22 real, and a mean error of 0.05 over four policies.

Predicted sum of cosines per frame for ppo-wm3 (first 10 s from hanging, mean of 32 rollouts; the rig: +1.22), and the mean absolute error over all four policies, by the rollout length K used in training:

| K | 1 | 4 | 6 | **8** | 10 | 12 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Predicted ppo-wm3 | +1.96 | +1.85 | +1.51 | **+1.19** | +0.95 | +0.84 | +0.67 | +0.46 | +0.47 |
| Mean error, 4 policies | 0.25 | 0.22 | 0.09 | **0.05** | 0.11 | 0.17 | 0.17 | 0.26 | 0.44 |

K = 8 with another seed predicts +1.21 (mean error 0.07).

The prediction for ppo-wm3 falls steadily with K: one-step models overrate balancing (the original wm3: +2.14), because they learn a false stable point near upright; long-rollout models underrate it, because averaging the future blurs the small corrections that balancing needs. The open-loop table above ranks K = 8 near the bottom, so it cannot choose a world model for policies on its own.

The three policies that do not balance are predicted well by every model:

| Policy | Rig | wm3 (one-step, 0.18M) | K = 8 | K = 16 |
| --- | --- | --- | --- | --- |
| ppo-persistent | +0.16 | +0.02 | +0.10 | +0.16 |
| ppo-speed | +0.06 | +0.12 | +0.08 | +0.10 |
| ppo-wm2 | +0.47 | +0.52 | +0.55 | +0.56 |

The evidence is thin in one place: only one of the four policies balances, with 10–30 s of rig data each.

## A policy trained in the K = 8 model

`ppo-k8` scores about twice `ppo-wm3` in every rollout-trained world model, including a K = 8 model it never trained in, which forecasts roughly +2.2 per frame on the rig.

It was trained in the K = 8 model (`imagination/configs/wm-k8.toml`) with the recipe that balanced best before (lr 1e-3, 5,000 iterations, speed penalty), on a spot L40S. Sum of cosines per frame and share of frames with all three arms within 30° of upright, 64 rollouts of 10 s from hanging:

| World model | ppo-wm3 | ppo-k8 |
| --- | --- | --- |
| K = 8 (ppo-k8 trained in it) | +1.19, 6% | +2.30, 56% |
| K = 8, another seed (never seen) | +1.18, 7% | +2.15, 44% |
| K = 16 (pessimistic) | +0.66, 3% | +1.44, 18% |
| wm3 (one-step; ppo-wm3 trained in it) | +2.14, 37% | +1.06, 10% |

- It loses little in the independent K = 8 model, so it is not exploiting quirks of one model, unlike `ppo-wm3-lr1e3-5k`, whose three-arm balance in wm3 collapsed to +0.47 in rollout-trained models.
- Its video shows the first two arms held upright with visible corrections, and the outer arm mostly up, wobbling and recovering.
- It is exported (`learning/runs/policy/ppo-k8/policy.json`, checked against PyTorch, 29 µs a decision) and untested on the rig.

## Costs and next steps

The program used roughly 15–20 GPU instance-hours, an estimated $15 (Cost Explorer lags a day), and left 8.4 GB of runs on S3, about $0.20 a month.

1. **Test ppo-k8 on the rig**, 30 s greedy from hanging. A result near +2.2 confirms the K = 8 model; then collect with it, sampled, to cover the held-upright states.
2. **Stochastic world models** (ensembles, or distributional outputs) could keep K = 32–64's long-horizon accuracy without the averaging that makes them pessimistic in closed loop.
3. **More closed-loop evidence**: each new policy tested on the rig adds a row to `imagination.sim2real`; one balancing policy is thin ground for choosing K.
4. **Faster policy training on GPUs**: on the L40S it ran at the Mac's speed, and single-rollout evaluation took 20% of the time.

The code is in `learning/` (main, as of commit 5a04fd7), and every run is in W&B (projects double-pendulum-world-model and double-pendulum-imagination) and on S3 under `double_pendulum/runs/`.
