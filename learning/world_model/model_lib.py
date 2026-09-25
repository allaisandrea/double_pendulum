"""The world model: an MLP from a window of steps to the next observation.

Input is `window` steps of (sin and cos of each tag's yaw, whether each tag
was seen, action). An unseen tag's sin and cos are zeroed. Output is, per
tag, the change in sin and cos since the tag was last seen in the window,
and the logit of the tag going unseen in the next frame.

Predicting the change rather than the value keeps the target on the scale
of one frame's motion. The change is divided by `delta_scale`, its typical
size in the training data, so the network works in units of about one.
"""
import torch
from torch import nn

from common.data_lib import NUM_TAGS

STEP_FEATURES = NUM_TAGS * 2 + NUM_TAGS + 1


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


class WorldModel(nn.Module):
    def __init__(self, window: int, hidden: int, layers: int, delta_scale: float):
        super().__init__()
        self.window = window
        self.register_buffer("delta_scale", torch.tensor(float(delta_scale)))
        dims = [window * STEP_FEATURES] + [hidden] * layers
        blocks = []
        for a, b in zip(dims, dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], NUM_TAGS * 3)

    def forward(self, obs, present, action):
        """Normalised change [B, NUM_TAGS, 2] and missing logit [B, NUM_TAGS].

        obs [B, window, NUM_TAGS, 2], present [B, window, NUM_TAGS], action [B, window].
        """
        b = obs.shape[0]
        seen = present.to(obs.dtype)
        x = torch.cat(
            [
                (obs * seen[..., None]).flatten(2),
                seen,
                action[..., None],
            ],
            dim=-1,
        ).reshape(b, -1)
        out = self.head(self.body(x)).reshape(b, NUM_TAGS, 3)
        return out[..., :2], out[..., 2]

    def predict(self, obs, present, action):
        """The next observation [B, NUM_TAGS, 2], the missing logit, and the
        reference it was predicted from, with whether each tag had one."""
        delta, logit = self(obs, present, action)
        ref, has_ref = last_seen(obs, present)
        return ref + delta * self.delta_scale, logit, ref, has_ref


def losses(model: WorldModel, obs, present, action, target, target_present):
    """Mean squared error of the normalised change, over the tags seen in
    the target that have a reference, and the cross-entropy of the missing
    logits. target [B, NUM_TAGS, 2], target_present [B, NUM_TAGS]."""
    delta, logit = model(obs, present, action)
    ref, has_ref = last_seen(obs, present)
    mask = target_present & has_ref
    change = (target - ref) / model.delta_scale
    mse = (delta - change)[mask].pow(2).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(
        logit, (~target_present).to(logit.dtype)
    )
    return mse, bce
