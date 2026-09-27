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
(nll) and calibration.
"""
import torch
from torch.nn import functional as F

from common.data_lib import Batch, Windows
from world_model.model_lib import WorldModel, last_seen, losses


@torch.no_grad()
def rollout(model: WorldModel, batch: Batch, horizon: int):
    """Predicted observations [B, horizon, NUM_TAGS, 2], and the starting
    window's references [B, NUM_TAGS, 2] and whether each tag had one."""
    w = model.window
    obs, present = batch.obs[:, :w], batch.present[:, :w]
    ref, has_ref = last_seen(obs, present)
    seen = torch.ones_like(present[:, 0])
    preds = []
    for h in range(horizon):
        nxt, _, _, _ = model.predict(obs, present, batch.action[:, h : h + w])
        nxt = F.normalize(nxt, dim=-1)
        preds.append(nxt)
        obs = torch.cat([obs[:, 1:], nxt[:, None]], dim=1)
        present = torch.cat([present[:, 1:], seen[:, None]], dim=1)
    return torch.stack(preds, dim=1), ref, has_ref


def one_minus_r2(pred_change, change, mask) -> float:
    """SSE / SST over the entries where `mask` holds (it has no last dim)."""
    y, p = change[mask], pred_change[mask]
    sse = (p - y).pow(2).sum()
    sst = (y - y.mean(dim=0)).pow(2).sum()
    return (sse / sst).item()


def calibration(model: WorldModel, batch: Batch) -> dict:
    """A stochastic model's one-step calibration: the share of normalised
    changes within 1 and 2 predicted standard deviations of the mean (0.683
    and 0.954 if calibrated), and the median standard deviation."""
    w = model.window
    obs, present = batch.obs[:, :w], batch.present[:, :w]
    delta, _, logvar = model(obs, present, batch.action[:, :w])
    ref, has_ref = last_seen(obs, present)
    mask = batch.present[:, w] & has_ref
    z = ((delta - (batch.obs[:, w] - ref) / model.delta_scale) * (-0.5 * logvar).exp())[mask].abs()
    return {
        "within_1sd": (z < 1).float().mean().item(),
        "within_2sd": (z < 2).float().mean().item(),
        "median_sd": (0.5 * logvar[mask]).exp().median().item(),
    }


@torch.no_grad()
def evaluate(model: WorldModel, windows: Windows, index, horizons: list[int]) -> dict:
    """Metrics on the windows at `index`, which hold `window + max(horizons)`
    steps each."""
    model.eval()
    w, horizon = model.window, max(horizons)
    batch = windows.gather(index)
    preds, ref, has_ref = rollout(model, batch, horizon)
    target = batch.obs[:, w : w + horizon]
    mask = batch.present[:, w : w + horizon] & has_ref[:, None]
    change = target - ref[:, None]
    pred_change = preds - ref[:, None]

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
    for h in horizons:
        metrics[f"one_minus_r2/h{h:03d}"] = one_minus_r2(
            pred_change[:, h - 1], change[:, h - 1], mask[:, h - 1]
        )
        metrics[f"copy_last/h{h:03d}"] = one_minus_r2(
            torch.zeros_like(change[:, h - 1]), change[:, h - 1], mask[:, h - 1]
        )
    model.train()
    return metrics
