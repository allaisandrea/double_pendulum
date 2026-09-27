# Double pendulum world model experiments

As of 2026-09-27, after the policy ranking on the rig. Every figure is drawn by `docs/figures.py` from the CSV files in `docs/results`, which `docs/results.py` regenerates from the sources: W&B, `world_model.profile` and its cloud logs, `imagination.sim2real` and `imagination.ranking` (`uv run python -m docs.results`, then `uv run --group docs python docs/figures.py`). An earlier shared version, from before the ranking: <https://claude.ai/code/artifact/4ffc27df-3c87-4337-a18b-35ba2bf28011>

## Summary

Every world model overrates the policies trained in it, so a policy has to be judged in a model it did not train in. Ranked on the rig, 13 policies are ordered almost exactly by the plain one-step 512 × 3 model (Spearman 0.99). The K = 8 rollout-trained model, chosen earlier on four policies, ranks them worst: it forecast +2.2 per frame for its own policy `ppo-k8`, which scores +1.03 on the rig, behind `ppo-wm3` (+1.33).

- **Scaling helps little.** Across 8 sizes (0.18M to 17M parameters) and 5 training lengths, short-horizon error improved about 15%; larger models overfit the 5 hours of data. No size fixed the long horizon near upright.
- **Rollout training is the big lever for open-loop accuracy.** Training on K-frame rollouts cut the 0.5 s error from upright from 1.59 (worse than copying the last frame) to 0.32 at K = 32.
- **Open-loop accuracy is not what policies need.** One-step models overrate a balancing policy; long-rollout models underrate it, because they learn to predict the average future.
- **Policies exploit the model they train in.** PPO finds the model's errors: K = 8 overrates its own policies by 1.01 per frame on average and the others by 0.06; wm3 its own by 0.60 and the others by 0.11. A second K = 8 seed, and K = 10–16, share the errors, so they are no independent check.
- **Rank policies with one-step models, K ≤ 6** (MMRV 0.02–0.07), and choose world models with the ranking (`imagination.ranking`), not with 1 − R² or a handful of policies.

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

**Closed-loop metrics**: each world model's predicted sum of cosines per frame against the rig's, with the policy in the loop: first for four policies from hanging (`imagination.sim2real`), then for 13 policies ranked on the rig (`imagination.ranking`).

**Infrastructure**: an L4 on-demand GPU, spot L4 and L40S GPUs when available, and the Mac, driven by a job queue (`cloud/queue.py`). Long runs checkpoint and resume on S3 (two spot interruptions were survived), and one constant-rate run per model size branches into cooldowns at five lengths (`world_model.scale`).

## Profiling and training settings

An L4 GPU with batch 4096, `torch.compile` and bf16 trains 2–11× faster than the Mac, more the larger the model (4.7× for 512 × 3), and batch 4096 at lr 2e-3 matches batch 1024 per window seen.

Milliseconds per training step (`world_model.profile`):

![Milliseconds per training step by model, batch and device](figures/profiling.svg)

- At batch 1024 every device is bound by kernel launches (about 2 ms); large batches and compilation are where the GPU pays.
- The T4 lacks bf16, which makes it slower; it was dropped from the instance list.
- Evaluation of one set (4,096 windows, 125 frames) takes 0.2 s on the Mac and 0.06 s on the L4.

The batch sweep (`sweeps/batch.txt`) trained the 256 × 3 model on 100M windows each (1 − R² on val_policy_wm3). The differences are small, except that batch 16,384 needs lr 4e-3:

![1 − R² at 16 and 64 frames by batch size, learning rate and precision](figures/batch.svg)

All later runs use batch 4096, lr 2e-3, compile and bf16 on the GPU.

## Phase 1: scaling model size and training length

With one-step training, the best model was 512 × 3 after 246M windows (0.0199 at 16 frames), about 15% better than the original 175k-parameter recipe (0.0233, batch 1024, 100M windows); nothing fixed the long horizon near upright.

Each of 8 sizes (widths 256–2048, depths 3 and 5) trained one 200k-step constant-rate run, with cooldown branches at 12.5k to 200k steps (`cloud/queues/scaling.txt`). A branch at step b cools down over another 0.2 b steps, so it trains on 1.2 b × 4096 windows, 61M to 983M: 40 finished models for about 1.4× the cost of the 200k-step runs.

1 − R² at 16 frames on val_policy_wm3 (lower is better), by training windows:

![1 − R² at 16 frames against training windows, one line per width, for depths 3 and 5](figures/scaling.svg)

- Small models improve with length to the end; models of 2M parameters and more are best at 61–246M windows, then get worse as they overfit.
- On val_random_walk the same pattern holds: the largest models go from about 0.036 to 0.056 at 16 frames over the longer runs.
- From upright, 0.5 s ahead, every one of the 40 models scores 1 − R² between 1.54 and 1.64: worse than predicting no change. Capacity and length do not fix it.

## Rollout training

Training on K-frame rollouts of the model's own predictions (`rollout_train`), with gradients through the whole rollout, cut the 0.5 s error from upright up to 5×; longer rollouts help the long horizon and cost some short-horizon accuracy.

All runs: 512 × 3, 50k steps (205M windows), batch 4096, lr 2e-3, unless noted. 1 − R² on val_policy_wm3, lower is better; "from upright" is its upright subset. K = 64 is the run with gradient clipping and lr 1e-3.

![1 − R² against the training rollout length K, at 16 frames and at 0.5 s and 1 s, overall and from upright](figures/rollout.svg)

- K = 12–16 give the best short horizon (about 2× better than one-step); K = 64 gives the best long horizon, and is the only model below 1 at 1 s from upright.
- K = 64 diverged at lr 2e-3; gradient clipping at norm 1.0 (`max_grad_norm`) and lr 1e-3 made it train. Clipping changes little at K = 32 (0.351 against 0.317 from upright at 64 frames).
- At K = 16, more steps (100k), width (1024) or history (32 frames) add nothing: all score within 0.06 of the base run from upright at 64 frames. Data and objective, not capacity, limit these models.

## Closed-loop fidelity on four policies

On the first closed-loop test the K = 8 model looked best: +1.19 for the balancing policy ppo-wm3 against +1.22 on the rig, and a mean error of 0.05 over four policies. The policy ranking below overturned this: the test held no policy trained in the model under test.

Predicted sum of cosines per frame for ppo-wm3 (first 10 s from hanging, mean of 32 rollouts), and the mean absolute error over the four policies (ppo-wm3, ppo-persistent, ppo-speed and ppo-wm2, each 10–30 s on the rig), by the rollout length K used in training:

![Predicted ppo-wm3 score and mean error over four policies against the training rollout length K](figures/sim2real.svg)

The prediction for ppo-wm3 falls steadily with K: one-step models overrate balancing (the original wm3: +2.14), because they learn a false stable point near upright; long-rollout models underrate it, because averaging the future blurs the small corrections that balancing needs. The three policies that do not balance were predicted within 0.15 by wm3, K = 8 and K = 16.

## A policy trained in the K = 8 model

`ppo-k8` was trained in the K = 8 model (`imagination/configs/wm-k8.toml`) with the recipe that balanced best before (lr 1e-3, 5,000 iterations, speed penalty), on a spot L40S. The K = 8 model scored it +2.30 per frame, a second K = 8 seed +2.15 and wm3 +1.06; I read the second seed's agreement as evidence that it did not exploit its model, and forecast about +2.2 on the rig. On the rig it scores +1.03: wm3, the one model of a different kind, was right. Two seeds of one recipe share its systematic errors, so they do not check each other.

## Policy ranking on the rig

Ranking 13 policies on the rig, after SIMPLER (arXiv:2405.05941), shows which world models can judge policies: the one-step models order them almost exactly, and K = 8–32 order them badly, because they overrate the policies trained in K = 8.

**Protocol** (`imagination.ranking`, `imagination/ranking/`): each policy ran greedily on the rig for 2 minutes, driving 10 s and resting 5 s, in two sessions with the order shuffled (52 minutes in all). Each drive after the first is an episode: its real score is the sum of the arms' cosines per frame over the drive, and each world model imagines 8 rollouts from the same recorded 16 frames before the drive, with the policy in the loop. 14 episodes per policy. The pool spreads over the range the K = 8 model predicts: `ppo-k8` and four of its checkpoints, six policies trained in wm3, and two older ones; 10 of the 13 had never run on the rig, so their data is in no world model's training set.

![Rig score of each policy, coloured by the world model it was trained in](figures/ranking_rig.svg)

- The three best policies were trained in wm3, and score within 0.15 of each other. `ppo-wm3-lr1e3-5k`'s three-arm balance in wm3 (+2.54 there) does not happen on the rig.
- The `ppo-k8` checkpoints improve steadily with training, from +0.44 at iteration 100 to +1.03 at 5,000.

Each world model's prediction against the rig; a perfect model puts every policy on the dotted line. Stars are the policies trained in that model:

![Predicted against real score for six world models, stars marking policies trained in the model](figures/ranking_scatter.svg)

- **Each model overrates its own policies most.** K = 8 puts its own four trained checkpoints at +1.6 to +2.3, against +0.54 to +1.03 real, while it predicts the wm3-trained policies well. wm3 puts its own balancers at +2.1 to +2.5, against +1.2 to +1.3, while it predicts `ppo-k8` exactly (+1.04 against +1.03).
- One-step models overrate balancing by about 0.5 across the board, but consistently, so the order holds. Long-rollout models (K = 64) underrate every balancer and flatten the order.

Each policy's overrating in each world model (predicted − real); stars are the model's own policies:

![Predicted minus real score of every policy in every world model, stars marking the policies trained in the model](figures/ranking_exploitation.svg)

- **The exploitation gap**, how much more a model overrates its own policies than the others, is +0.94 per frame for K = 8 and +0.49 for wm3. The gap grows with training: K = 8 is right about `ppo-k8` at iteration 100 (+0.08) and overrates it by 1.1–1.3 from iteration 250 on.
- **It spreads to the models closest to K = 8.** K = 8 seed 1, K = 10 and K = 12 overrate the K = 8 policies by up to 1.2 too, though none was trained in them; K = 1–6 overrate them by 0.1–0.4, as they do the rest.

Agreement over policies (MMRV, SIMPLER's mean maximum rank violation: 0 is the right order, and a swap costs the real gap between the swapped policies). For wm3 and K = 8, "held out" leaves out the policies trained in them; the other models trained none:

![MMRV, Spearman correlation and mean absolute error for each world model, over all policies, those new to the rig, and those not trained in the model](figures/ranking_agreement.svg)

- The one-step 512 × 3 model (K = 1) ranks best: MMRV 0.02, Spearman 0.99, and as well on the 10 policies new to the rig (MMRV 0.03).
- K = 4 and K = 6 rank nearly as well (MMRV 0.05–0.07) with the smallest absolute error (0.18–0.22).
- K = 8 to 32 rank badly (MMRV 0.31–0.41), from the `ppo-k8` checkpoints: exploitation of K = 8 carries over to the models closest to it. K = 64 is less affected (MMRV 0.25), since it underrates everything.
- **Held out, K = 8 is the most accurate model** (mean error 0.15, bias +0.06, against 0.27 and +0.27 for K = 1), but it still ranks worse (MMRV 0.18 against 0.02): it squeezes the three best policies into +1.10 to +1.34 and misorders them. wm3 held out does about as well as on all policies (MMRV 0.07), with a mean error of only 0.11.
- Both views matter: held out measures a world model as a judge of other models' policies; all policies, own included, measures it as the forecaster of what is trained in it, which is how `ppo-k8`'s +2.2 was forecast.

## Costs and next steps

The program used roughly 15–20 GPU instance-hours, an estimated $15 (Cost Explorer lags a day), and left 8.4 GB of runs on S3, about $0.20 a month.

1. **Train policies against exploitation**: an ensemble of world models of different kinds, with a penalty on their disagreement, pushes PPO away from the states where any one model is wrong. Choose policies with a model they did not train in, such as the one-step 512 × 3.
2. **Add the ranking recordings to the training data**: 52 minutes of 13 policies, most of them new to the rig, cover states no training set holds.
3. **Stochastic world models** (ensembles, or distributional outputs) could keep K = 32–64's long-horizon accuracy without the averaging that makes them pessimistic in closed loop.
4. **Extend the ranking** with each new policy and world model: `imagination.ranking imagine` pools every session's episodes, and `agreement` rescores without new rollouts.
5. **Faster policy training on GPUs**: on the L40S it ran at the Mac's speed, and single-rollout evaluation took 20% of the time.

The code is in `learning/` (main, as of commit ad2bc75), and every run is in W&B (projects double-pendulum-world-model and double-pendulum-imagination) and on S3 under `double_pendulum/runs/`.
