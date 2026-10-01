"""Open-loop rollouts and the metrics computed on them.

A rollout starts from a window of recorded steps and feeds the model its
own predictions for `horizon` frames, with the recorded actions. Each
predicted tag is fed back as seen, with its sin and cos put back on the
unit circle.

Rollouts of a stochastic model follow its mean. The metric at horizon h
is 1 - R² of the change since each tag was last seen in the starting window: the error of the predicted change over the
variance of the actual one. Copying the last observation predicts no
change, and scores about 1; a perfect model scores 0. The copy-last
score is logged alongside, computed on the same samples. A stochastic
model is also scored on its one-step negative log-likelihood per element
(nll) and calibration; a flow model on its flow-matching loss and the
negative log-likelihood of its yaw changes (flow_metrics). Rollouts of a
flow model follow its flow from x0 = 0 (tau 0).
"""
import torch
from torch.nn import functional as F

from common.data_lib import NUM_TAGS, Batch, Windows
from world_model.flow_lib import FlowWorldModel, flow_losses, yaw_change
from world_model.model_lib import WorldModel, gaussian_nll, last_seen, losses


@torch.no_grad()
def rollout(model: WorldModel, batch: Batch, horizon: int, tau: float = 0.0,
            generator: torch.Generator | None = None):
    """Predicted observations [B, horizon, NUM_TAGS, 2], and the starting
    window's references [B, NUM_TAGS, 2] and whether each tag had one. A
    stochastic model follows its mean, or with `tau` draws each frame."""
    w = model.window
    obs, present = batch.obs[:, :w], batch.present[:, :w]
    ref, has_ref = last_seen(obs, present)
    seen = torch.ones_like(present[:, 0])
    preds = []
    for h in range(horizon):
        nxt, _, _, _ = model.predict(obs, present, batch.action[:, h : h + w], tau, generator)
        nxt = F.normalize(nxt, dim=-1)
        preds.append(nxt)
        obs = torch.cat([obs[:, 1:], nxt[:, None]], dim=1)
        present = torch.cat([present[:, 1:], seen[:, None]], dim=1)
    return torch.stack(preds, dim=1), ref, has_ref


def ensemble_metrics(model: WorldModel, batch: Batch, horizons: list[int], members: int, seed: int = 0) -> dict:
    """A stochastic model's rollouts as an ensemble: `members` sampled
    rollouts from each window (tau 1), scored at each horizon on the change
    since each tag was last seen, over the tags seen:

    - ensemble/one_minus_r2: 1 - R² of the ensemble's mean;
    - ensemble/crps: the continuous ranked probability score of the
      ensemble, per element, over copying the last frame's (its mean
      absolute error): 1 is no better than copying, 0 is perfect. A proper
      score, it rewards spread only where the future is uncertain;
    - ensemble/spread_skill: the ensemble's spread over the error of its
      mean, corrected for its size; 1 if calibrated, below 1 overconfident.
    """
    w = model.window
    repeat = lambda t: t.repeat_interleave(members, 0)
    big = Batch(repeat(batch.obs), repeat(batch.present), repeat(batch.action))
    preds, ref, has_ref = rollout(model, big, max(horizons), 1.0, torch.Generator().manual_seed(seed))
    b = batch.obs.shape[0]
    preds = preds.reshape(b, members, *preds.shape[1:])  # [B, M, H, T, 2]
    ref, has_ref = ref[::members], has_ref[::members]
    out = {}
    for h in horizons:
        mask = batch.present[:, w + h - 1] & has_ref
        y = (batch.obs[:, w + h - 1] - ref)[mask]  # [N, 2]
        x = (preds[:, :, h - 1] - ref[:, None]).permute(0, 2, 1, 3)[mask]  # [N, M, 2]
        mean = x.mean(1)
        crps = (x - y[:, None]).abs().mean(1) - 0.5 * (x[:, :, None] - x[:, None]).abs().mean((1, 2))
        spread = x.var(1, unbiased=True).mean().sqrt() * ((members + 1) / members) ** 0.5
        out[f"ensemble/one_minus_r2/h{h:03d}"] = one_minus_r2(mean, y, torch.ones(len(y), dtype=torch.bool, device=y.device))
        out[f"ensemble/crps/h{h:03d}"] = (crps.mean() / y.abs().mean()).item()
        out[f"ensemble/spread_skill/h{h:03d}"] = (spread / (mean - y).pow(2).mean().sqrt()).item()
    return out


def one_minus_r2(pred_change, change, mask) -> float:
    """SSE / SST over the entries where `mask` holds (it has no last dim)."""
    y, p = change[mask], pred_change[mask]
    sse = (p - y).pow(2).sum()
    sst = (y - y.mean(dim=0)).pow(2).sum()
    return (sse / sst).item()


def calibration(model: WorldModel, batch: Batch) -> dict:
    """A stochastic model's one-step calibration: the share of normalised
    changes within 1 and 2 predicted standard deviations of the mean (0.683
    and 0.954 if calibrated), the median standard deviation, and the median
    NLL per element, which a few confident misses cannot move as they do
    the mean."""
    w = model.window
    obs, present = batch.obs[:, :w], batch.present[:, :w]
    delta, _, logvar = model(obs, present, batch.action[:, :w])
    ref, has_ref = last_seen(obs, present)
    mask = batch.present[:, w] & has_ref
    error = (delta - (batch.obs[:, w] - ref) / model.delta_scale)[mask]
    z = (error * (-0.5 * logvar[mask]).exp()).abs()
    return {
        "median_nll": gaussian_nll(error.float(), logvar[mask].float()).median().item(),
        "within_1sd": (z < 1).float().mean().item(),
        "within_2sd": (z < 2).float().mean().item(),
        "median_sd": (0.5 * logvar[mask]).exp().median().item(),
    }


def flow_metrics(model: FlowWorldModel, batch: Batch, nll_steps: int) -> dict:
    """A flow model's one-step metrics: its flow-matching loss (`fm`, over
    noise and times drawn from a fixed seed) and the cross-entropy of its
    missing logits; and over the windows whose next frame has every tag seen
    (with a reference), the negative log-likelihood of the yaw changes, in
    radians, per tag (`nll`, the mean, and `median_nll`), integrated in
    `nll_steps` RK4 steps."""
    w = model.window
    obs, present, action = batch.obs[:, :w], batch.present[:, :w], batch.action[:, :w]
    fm, bce = flow_losses(model, obs, present, action, batch.obs[:, w], batch.present[:, w],
                          torch.Generator().manual_seed(0))
    ref, has_ref = last_seen(obs, present)
    full = (batch.present[:, w] & has_ref).all(-1)
    change = yaw_change(ref[full], batch.obs[full, w])
    nll = -model.log_prob(obs[full], present[full], action[full], change, nll_steps) / NUM_TAGS
    return {"fm": fm.item(), "bce": bce.item(), "nll": nll.mean().item(), "median_nll": nll.median().item()}


@torch.no_grad()
def evaluate(model: WorldModel | FlowWorldModel, windows: Windows, index, horizons: list[int], members: int = 0,
             nll_steps: int = 32) -> dict:
    """Metrics on the windows at `index`, which hold `window + max(horizons)`
    steps each; for a stochastic model with `members`, also its
    ensemble_metrics."""
    model.eval()
    w, horizon = model.window, max(horizons)
    batch = windows.gather(index)
    preds, ref, has_ref = rollout(model, batch, horizon)
    target = batch.obs[:, w : w + horizon]
    mask = batch.present[:, w : w + horizon] & has_ref[:, None]
    change = target - ref[:, None]
    pred_change = preds - ref[:, None]

    if isinstance(model, FlowWorldModel):
        metrics = flow_metrics(model, batch, nll_steps)
    else:
        fit, bce, mse = losses(
            model,
            batch.obs[:, :w],
            batch.present[:, :w],
            batch.action[:, :w],
            batch.obs[:, w],
            batch.present[:, w],
        )
        metrics = {"mse": mse.item(), "bce": bce.item()}
        if model.stochastic:
            metrics |= calibration(model, batch)
            metrics["nll"] = fit.item()
    if model.stochastic and members:
        metrics |= ensemble_metrics(model, batch, horizons, members)
    for h in horizons:
        metrics[f"one_minus_r2/h{h:03d}"] = one_minus_r2(
            pred_change[:, h - 1], change[:, h - 1], mask[:, h - 1]
        )
        metrics[f"copy_last/h{h:03d}"] = one_minus_r2(
            torch.zeros_like(change[:, h - 1]), change[:, h - 1], mask[:, h - 1]
        )
    model.train()
    return metrics
