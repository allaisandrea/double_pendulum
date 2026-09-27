"""The world model: an MLP from a window of steps to the next observation.

Input is `window` steps of (sin and cos of each tag's yaw, whether each tag
was seen, action). An unseen tag's sin and cos are zeroed. Output is, per
tag, the change in sin and cos since the tag was last seen in the window,
and the logit of the tag going unseen in the next frame.

Predicting the change rather than the value keeps the target on the scale
of one frame's motion. The change is divided by `delta_scale`, its typical
size in the training data, so the network works in units of about one.

A stochastic model also predicts the log-variance of each normalised
change: the change is a normal distribution with a diagonal covariance,
trained on its negative log-likelihood (beta-NLL, Seitzer et al. 2022,
arXiv:2203.09168, which weights each term by its variance to the power
beta so that hard transitions do not buy a loose fit with a large
variance: beta 0 is the plain likelihood, 1 weighs errors as MSE does).
`predict` draws the next observation from it, its spread scaled by `tau`
(0: the mean).
"""
import math

import torch
from torch import nn

from common.data_lib import NUM_TAGS

STEP_FEATURES = NUM_TAGS * 2 + NUM_TAGS + 1
# Bounds on a stochastic model's log-variance, in normalised units, applied
# softly so that the gradient never vanishes: a standard deviation from
# about 0.007 to 7.4 times a typical frame's change.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 4.0


def last_seen(obs: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each tag's most recent observation in the window, and whether it has one.

    obs [B, L, NUM_TAGS, 2], present [B, L, NUM_TAGS] -> [B, NUM_TAGS, 2], [B, NUM_TAGS].
    A tag never seen in the window gets zeros and False.
    """
    steps = torch.arange(obs.shape[1], device=obs.device)[None, :, None]
    latest = torch.where(present, steps, -1).amax(dim=1)  # [B, NUM_TAGS]
    index = latest.clamp(min=0)[:, None, :, None].expand(-1, 1, -1, 2)
    ref = obs.gather(1, index).squeeze(1)
    seen = latest >= 0
    return ref * seen[..., None], seen


def step_features(obs, present, action) -> torch.Tensor:
    """[B, L, STEP_FEATURES]: each step's sin and cos, zeroed for unseen
    tags, whether each tag was seen, and the action."""
    seen = present.to(obs.dtype)
    return torch.cat([(obs * seen[..., None]).flatten(2), seen, action[..., None]], dim=-1)


def soft_clamp(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    x = low + nn.functional.softplus(x - low)
    return high - nn.functional.softplus(high - x)


class WorldModel(nn.Module):
    def __init__(self, window: int, hidden: int, layers: int, delta_scale: float, stochastic: bool = False):
        super().__init__()
        self.window = window
        self.stochastic = stochastic
        self.register_buffer("delta_scale", torch.tensor(float(delta_scale)))
        dims = [window * STEP_FEATURES] + [hidden] * layers
        blocks = []
        for a, b in zip(dims, dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], NUM_TAGS * (5 if stochastic else 3))

    def forward(self, obs, present, action):
        """Normalised change [B, NUM_TAGS, 2] (the mean, for a stochastic
        model), missing logit [B, NUM_TAGS], and the change's log-variance
        [B, NUM_TAGS, 2], or None for a deterministic model.

        obs [B, window, NUM_TAGS, 2], present [B, window, NUM_TAGS], action [B, window].
        """
        b = obs.shape[0]
        x = step_features(obs, present, action).reshape(b, -1)
        out = self.head(self.body(x)).reshape(b, NUM_TAGS, -1)
        logvar = soft_clamp(out[..., 3:5], LOGVAR_MIN, LOGVAR_MAX) if self.stochastic else None
        return out[..., :2], out[..., 2], logvar

    def predict(self, obs, present, action, tau: float = 0.0, generator: torch.Generator | None = None):
        """The next observation [B, NUM_TAGS, 2], the missing logit, and the
        reference it was predicted from, with whether each tag had one. A
        stochastic model draws the change, its standard deviation scaled by
        `tau`, from `generator` (a CPU generator) if given."""
        delta, logit, logvar = self(obs, present, action)
        if logvar is not None and tau > 0:
            if generator is None:
                noise = torch.randn_like(delta)
            else:
                noise = torch.randn(delta.shape, generator=generator).to(delta.device)
            delta = delta + tau * (0.5 * logvar).exp() * noise
        ref, has_ref = last_seen(obs, present)
        return ref + delta * self.delta_scale, logit, ref, has_ref


def load_world_model(path, device) -> WorldModel:
    """A checkpoint train.py saved, frozen, in eval mode."""
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = WorldModel(cfg["window"], cfg["hidden"], cfg["layers"], ck["delta_scale"], cfg.get("stochastic", False))
    model.load_state_dict(ck["model"])
    model.requires_grad_(False)
    return model.to(device).eval()


def gaussian_nll(error: torch.Tensor, logvar: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Each element's negative log-likelihood under a normal distribution
    with log-variance `logvar`, of `error` from its mean, weighted by the
    detached variance to the power `beta` (beta-NLL; 0 is the plain NLL)."""
    nll = 0.5 * (logvar + error.pow(2) * (-logvar).exp() + math.log(2 * math.pi))
    return nll * (beta * logvar.detach()).exp() if beta else nll


def step_losses(delta, logit, logvar, ref, has_ref, target, target_present, delta_scale, beta: float):
    """The fit of the normalised change (its MSE, or for a stochastic model
    its beta-NLL), the cross-entropy of the missing logits, and the MSE, over
    the tags seen in the target that have a reference; in float32."""
    mask = target_present & has_ref
    error = (delta.float() - (target - ref) / delta_scale)[mask]
    mse = error.pow(2).mean()
    fit = mse if logvar is None else gaussian_nll(error, logvar.float()[mask], beta).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(
        logit.float(), (~target_present).to(torch.float32)
    )
    return fit, bce, mse


def losses(model: WorldModel, obs, present, action, target, target_present, beta: float = 0.0):
    """`step_losses` of predicting target [B, NUM_TAGS, 2], target_present
    [B, NUM_TAGS] from the window."""
    delta, logit, logvar = model(obs, present, action)
    ref, has_ref = last_seen(obs, present)
    return step_losses(delta, logit, logvar, ref, has_ref, target, target_present, model.delta_scale, beta)


def rollout_losses(model: WorldModel, obs, present, action, steps: int, beta: float = 0.0):
    """`losses` averaged over `steps` frames of rollout: the model predicts
    frame window from the recorded window, then each next frame from its
    own predictions, fed back seen and on the unit circle, with the
    recorded actions, and gradients flow through the whole rollout. Each
    step's error is against the recorded frame. obs [B, window + steps,
    NUM_TAGS, 2], present likewise, action [B, window + steps - 1]. With
    steps 1, it is `losses`."""
    w = model.window
    o, p = obs[:, :w], present[:, :w]
    seen = torch.ones_like(p[:, 0])
    parts = []
    for k in range(steps):
        target, target_present = obs[:, w + k], present[:, w + k]
        delta, logit, logvar = model(o, p, action[:, k : k + w])
        ref, has_ref = last_seen(o, p)
        parts.append(step_losses(delta, logit, logvar, ref, has_ref, target, target_present,
                                 model.delta_scale, beta))
        if k + 1 < steps:
            # A stochastic model's rollout feeds back its mean.
            nxt = nn.functional.normalize(ref + delta * model.delta_scale, dim=-1)
            o = torch.cat([o[:, 1:], nxt[:, None]], dim=1)
            p = torch.cat([p[:, 1:], seen[:, None]], dim=1)
    return tuple(torch.stack(x).mean() for x in zip(*parts))
