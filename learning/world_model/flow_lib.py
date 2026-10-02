"""A flow-matching world model: the next frame drawn by integrating a learned
velocity field, rather than from a normal distribution.

It sees the same window as WorldModel and gives the same missing logits.
The next observation is the change in each tag's yaw since it was last seen
in the window, divided by that tag's typical one-frame change
(`delta_scale`, [NUM_TAGS]), and it is drawn by carrying a standard normal
draw x0 at t = 0 to the change at t = 1 along dx/dt = v(x, t | window):
conditional flow matching with the straight path (Lipman et al. 2023,
arXiv:2210.02747; Liu et al. 2023, arXiv:2209.03003). The body encodes the
window once; a small head gives the velocity from that encoding, x and t,
at every step of the ODE. The tags' changes are drawn jointly, so the
model can say how the arms move together, and their distribution has no
set shape: several modes, skew, heavy tails.

The target is the change in yaw, not the change in sin and cos WorldModel
predicts: given the reference, the next sin and cos lie on a circle, a
curve in the plane, where a density over the plane is degenerate, and a
flow would chase an unbounded likelihood. The yaw is one number per tag,
and the predicted frame lies on the unit circle by construction.

Training regresses v(x_t, t) onto x1 − x0 at x_t = (1 − t) x0 + t x1, over
the tags seen in the next frame that have a reference. An unseen tag's
coordinate stays at its noise x0 all along the path, and its velocity is
not trained: the others' see that it carries no information.

Sampling integrates the ODE in `flow_steps` midpoint steps from noise
scaled by `tau`; tau 0 starts from x0 = 0, a deterministic model that
stands in for a mean. The log-likelihood of a change is the instantaneous
change of variables (Chen et al. 2018, arXiv:1806.07366): integrating back
from t = 1 to 0, log p1(x1) = log N(x0; 0, I) − ∫ div v dt, with the
divergence exact (one backward pass per tag) and RK4 steps.
"""
import math

import torch
from torch import nn

from common.data_lib import NUM_TAGS
from world_model.model_lib import LOGVAR_MAX, LOGVAR_MIN, STEP_FEATURES, gaussian_nll, last_seen, soft_clamp, step_features

# Sine and cosine features of t, at frequencies π, 2π, ... TIME_FREQS π.
TIME_FREQS = 8


def yaw_change(ref: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """The angle [...] that turns ref [..., 2] into obs [..., 2], each a
    (sin, cos), in (−π, π]."""
    s0, c0 = ref.unbind(-1)
    s1, c1 = obs.unbind(-1)
    return torch.atan2(s1 * c0 - c1 * s0, c1 * c0 + s1 * s0)


def rotate(ref: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """ref [..., 2], a (sin, cos), turned by angle [...]."""
    s, c = ref.unbind(-1)
    sa, ca = angle.sin(), angle.cos()
    return torch.stack([s * ca + c * sa, c * ca - s * sa], dim=-1)


def time_features(t: torch.Tensor) -> torch.Tensor:
    """[B, 2 * TIME_FREQS] for t [B]."""
    angle = t[:, None] * math.pi * torch.arange(1, TIME_FREQS + 1, device=t.device, dtype=t.dtype)
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class FlowWorldModel(nn.Module):
    stochastic = True

    def __init__(self, window: int, hidden: int, layers: int, delta_scale, flow_hidden: int, flow_layers: int,
                 flow_steps: int = 8):
        super().__init__()
        self.window = window
        self.flow_steps = flow_steps
        self.register_buffer("delta_scale", torch.as_tensor(delta_scale, dtype=torch.float32).reshape(NUM_TAGS))
        dims = [window * STEP_FEATURES] + [hidden] * layers
        blocks = []
        for a, b in zip(dims, dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.logit_head = nn.Linear(hidden, NUM_TAGS)
        # The head's first layer is a sum: the window's part is computed
        # once per window, the part of x and t at each step of the ODE.
        self.condition = nn.Linear(hidden, flow_hidden)
        self.point = nn.Linear(NUM_TAGS + 2 * TIME_FREQS, flow_hidden, bias=False)
        head = [nn.GELU()]
        for _ in range(flow_layers - 1):
            head += [nn.Linear(flow_hidden, flow_hidden), nn.GELU()]
        self.head = nn.Sequential(*head, nn.Linear(flow_hidden, NUM_TAGS))

    @classmethod
    def from_config(cls, cfg: dict, delta_scale) -> "FlowWorldModel":
        """The head is `flow_hidden` (the body's `hidden`) wide and
        `flow_layers` (2) deep; sampling takes `flow_steps` (8)."""
        return cls(cfg["window"], cfg["hidden"], cfg["layers"], delta_scale, cfg.get("flow_hidden", cfg["hidden"]),
                   cfg.get("flow_layers", 2), cfg.get("flow_steps", 8))

    def encode(self, obs, present, action):
        """The window's condition for the velocity [B, flow_hidden], and the
        missing logit [B, NUM_TAGS]."""
        b = obs.shape[0]
        h = self.body(step_features(obs, present, action).reshape(b, -1))
        return self.condition(h), self.logit_head(h)

    def velocity(self, cond, x, t):
        """v [B, NUM_TAGS] at x [B, NUM_TAGS] and t [B]."""
        return self.head(cond + self.point(torch.cat([x, time_features(t).to(x.dtype)], dim=-1)))

    def forward(self, obs, present, action, x, t):
        """The velocity at x, t, and the missing logit: what training needs,
        in one call to compile."""
        cond, logit = self.encode(obs, present, action)
        return self.velocity(cond, x, t), logit

    def integrate(self, cond, x0):
        """x1 [B, NUM_TAGS]: x0 carried from t = 0 to 1, in `flow_steps`
        midpoint steps."""
        x, dt = x0, 1.0 / self.flow_steps
        t = torch.zeros(len(x0), device=x0.device, dtype=x0.dtype)
        for _ in range(self.flow_steps):
            mid = x + 0.5 * dt * self.velocity(cond, x, t)
            x = x + dt * self.velocity(cond, mid, t + 0.5 * dt)
            t = t + dt
        return x

    def predict(self, obs, present, action, tau: float = 0.0, generator: torch.Generator | None = None):
        """As WorldModel.predict: the next observation [B, NUM_TAGS, 2], the
        missing logit, and the reference it was predicted from, with whether
        each tag had one. The change is drawn from noise scaled by `tau`
        (0: from x0 = 0), from `generator` (a CPU generator) if given."""
        cond, logit = self.encode(obs, present, action)
        shape = (obs.shape[0], NUM_TAGS)
        if tau > 0:
            noise = torch.randn(shape) if generator is None else torch.randn(shape, generator=generator)
            x0 = tau * noise.to(obs.device)
        else:
            x0 = torch.zeros(shape, device=obs.device)
        x1 = self.integrate(cond, x0)
        ref, has_ref = last_seen(obs, present)
        return rotate(ref, x1 * self.delta_scale), logit, ref, has_ref

    def log_prob(self, obs, present, action, change, steps: int = 32):
        """The log-density [B] of the yaw changes change [B, NUM_TAGS], in
        radians, all tags jointly, by integrating the ODE back from them in
        `steps` RK4 steps."""
        with torch.no_grad():
            cond, _ = self.encode(obs, present, action)

        def f(x, t):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = self.velocity(cond, x, t)
                div = sum(torch.autograd.grad(v[:, i].sum(), x, retain_graph=i + 1 < NUM_TAGS)[0][:, i]
                          for i in range(NUM_TAGS))
            return v.detach(), div.detach()

        # Back from t = 1 to 0: da/dt = div v, so a(0) = −∫ div v dt.
        x = change / self.delta_scale
        a = torch.zeros(len(x), device=x.device)
        h = -1.0 / steps
        t = torch.ones(len(x), device=x.device)
        for _ in range(steps):
            v1, d1 = f(x, t)
            v2, d2 = f(x + 0.5 * h * v1, t + 0.5 * h)
            v3, d3 = f(x + 0.5 * h * v2, t + 0.5 * h)
            v4, d4 = f(x + h * v3, t + h)
            x = x + h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
            a = a + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
            t = t + h
        log_p0 = -0.5 * (x.pow(2) + math.log(2 * math.pi)).sum(-1)
        # Normalised units to radians.
        return log_p0 + a - self.delta_scale.log().sum()


def flow_losses(model: FlowWorldModel, obs, present, action, target, target_present,
                generator: torch.Generator | None = None):
    """The flow-matching loss (the velocity's squared error, per element),
    and the cross-entropy of the missing logits, of predicting target
    [B, NUM_TAGS, 2], target_present [B, NUM_TAGS] from the window; in
    float32. The noise and times are drawn from `generator` (CPU) if given,
    else from torch's generator on the device."""
    ref, has_ref = last_seen(obs, present)
    mask = target_present & has_ref
    x1 = yaw_change(ref, target) / model.delta_scale
    if generator is None:
        x0, t = torch.randn_like(x1), torch.rand(len(x1), device=x1.device)
    else:
        x0 = torch.randn(x1.shape, generator=generator).to(x1.device)
        t = torch.rand(len(x1), generator=generator).to(x1.device)
    # An unseen tag stays at its noise.
    x1 = torch.where(mask, x1, x0)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    v, logit = model(obs, present, action, xt, t)
    fm = (v.float() - (x1 - x0))[mask].pow(2).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logit.float(), (~target_present).to(torch.float32))
    return fm, bce



class YawGaussianWorldModel(nn.Module):
    """The flow model's Gaussian twin: the same window, body, missing logits
    and target (each tag's yaw change over its `delta_scale`), drawn from a
    normal distribution with a diagonal covariance, as WorldModel's change
    in sin and cos is. It tells what predicting in yaw buys apart from the
    flow's free-form distribution, and its NLL, in radians per tag, compares
    with the flow model's."""
    stochastic = True

    def __init__(self, window: int, hidden: int, layers: int, delta_scale, logvar_min: float = LOGVAR_MIN):
        super().__init__()
        self.window = window
        self.logvar_min = logvar_min
        self.register_buffer("delta_scale", torch.as_tensor(delta_scale, dtype=torch.float32).reshape(NUM_TAGS))
        dims = [window * STEP_FEATURES] + [hidden] * layers
        blocks = []
        for a, b in zip(dims, dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        # Per tag: the mean and log-variance of its normalised yaw change,
        # and its missing logit.
        self.head = nn.Linear(hidden, NUM_TAGS * 3)

    @classmethod
    def from_config(cls, cfg: dict, delta_scale) -> "YawGaussianWorldModel":
        return cls(cfg["window"], cfg["hidden"], cfg["layers"], delta_scale, cfg.get("logvar_min", LOGVAR_MIN))

    def forward(self, obs, present, action):
        """The normalised yaw change's mean [B, NUM_TAGS], the missing logit
        [B, NUM_TAGS], and the change's log-variance [B, NUM_TAGS]."""
        b = obs.shape[0]
        out = self.head(self.body(step_features(obs, present, action).reshape(b, -1))).reshape(b, NUM_TAGS, 3)
        return out[..., 0], out[..., 1], soft_clamp(out[..., 2], self.logvar_min, LOGVAR_MAX)

    def predict(self, obs, present, action, tau: float = 0.0, generator: torch.Generator | None = None):
        """As WorldModel.predict: the next observation, its reference turned
        by a yaw change drawn with its standard deviation scaled by `tau`
        (0: the mean)."""
        mean, logit, logvar = self(obs, present, action)
        x = mean
        if tau > 0:
            noise = torch.randn(mean.shape) if generator is None else torch.randn(mean.shape, generator=generator)
            x = mean + tau * (0.5 * logvar).exp() * noise.to(mean.device)
        ref, has_ref = last_seen(obs, present)
        return rotate(ref, x * self.delta_scale), logit, ref, has_ref

    def log_prob(self, obs, present, action, change, steps: int = 0):
        """The log-density [B] of the yaw changes change [B, NUM_TAGS], in
        radians, all tags jointly, as FlowWorldModel.log_prob (`steps` is
        unused)."""
        mean, _, logvar = self(obs, present, action)
        x = change / self.delta_scale
        return -gaussian_nll(x - mean, logvar).sum(-1) - self.delta_scale.log().sum()


def yaw_gaussian_losses(model: YawGaussianWorldModel, obs, present, action, target, target_present,
                        beta: float = 0.0):
    """The beta-NLL of the normalised yaw changes, per element, over the tags
    seen in the target that have a reference, and the cross-entropy of the
    missing logits; in float32."""
    mean, logit, logvar = model(obs, present, action)
    ref, has_ref = last_seen(obs, present)
    mask = target_present & has_ref
    error = (mean.float() - yaw_change(ref, target) / model.delta_scale)[mask]
    fit = gaussian_nll(error, logvar.float()[mask], beta).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logit.float(), (~target_present).to(torch.float32))
    return fit, bce


@torch.no_grad()
def change_scale(obs, present, w: int) -> list[float]:
    """Each tag's RMS one-frame yaw change, over windows obs [B, w + 1,
    NUM_TAGS, 2], present likewise."""
    ref, has_ref = last_seen(obs[:, :w], present[:, :w])
    mask = (present[:, w] & has_ref).float()
    sq = yaw_change(ref, obs[:, w]).pow(2) * mask
    return (sq.sum(0) / mask.sum(0)).sqrt().tolist()
