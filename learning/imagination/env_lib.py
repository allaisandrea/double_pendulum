"""The pendulum in imagination: a batch of environments stepped by the world model.

Each environment keeps the last `history` steps, laid out as in the
recordings: the observation at each frame and the action in effect after
it. An episode starts from a window of real steps. Each step, the policy
sees the latest `policy_window` observations with the action before each
of them, and picks the action for the newest frame; the world model then
predicts the next frame from its own window, with that action in place.

A tag is dropped from the next frame with the probability the model
predicts, when `sample_missing` is set, so the policy learns to cope with
misses as it must on the rig. The reward comes from the prediction itself,
seen or not: the sum of the cosines of the yaws, corrected so that each
tag's hanging yaw reads 180°. A hanging arm scores -1, an upright one +1.

The world model works in the camera's raw yaws; only the reward, and what
is drawn or measured as upright, use the corrected ones.
"""
import torch
from torch.nn import functional as F

from common.data_lib import ACTION_SCALE, NUM_TAGS, Batch, Windows, correct_yaws
from world_model.model_lib import STEP_FEATURES, WorldModel, step_features

INT8_MIN, INT8_MAX = -128, 127


def action_levels(step: int, bins: int) -> torch.Tensor:
    """The int8 action for each bin: `bins` levels `step` apart, centred on 0."""
    if bins % 2 == 0:
        raise ValueError(f"{bins} bins have no middle one at 0")
    levels = (torch.arange(bins) - bins // 2) * step
    if levels.min() < INT8_MIN or levels.max() > INT8_MAX:
        raise ValueError(f"{bins} bins {step} apart leave the int8 range")
    return levels


class ImaginedEnv:
    def __init__(
        self,
        model: WorldModel,
        levels: torch.Tensor,
        policy_window: int,
        sample_missing: bool,
        hanging: torch.Tensor,
    ):
        """`hanging` [NUM_TAGS, 2] is the sin and cos of each tag's yaw with
        the pendulum hanging still, from `hanging_yaws`."""
        self.model = model
        self.hanging = hanging.to(model.delta_scale.device)
        self.policy_window = policy_window
        self.history = max(model.window, policy_window + 1)
        self.levels = levels.to(model.delta_scale.device)
        self.actions = self.levels / ACTION_SCALE
        self.sample_missing = sample_missing
        # Draws the misses when set, for repeatable runs; otherwise torch's
        # global generator does.
        self.generator: torch.Generator | None = None

    @property
    def features(self) -> int:
        """Size of the policy's input."""
        return self.policy_window * STEP_FEATURES

    def reset(self, starts: Windows, index: torch.Tensor):
        """Starts one environment from each window at `index`, of `history` steps."""
        assert starts.length == self.history
        self.reset_to(starts.gather(index))

    def restart(self, which: torch.Tensor, starts: Windows, generator: torch.Generator):
        """Starts the environments where `which` [N], on the CPU, holds
        again, each from a window of `starts` drawn with `generator`.
        Keeping `which` on the CPU spares waiting for the device to count."""
        n = int(which.sum())
        if n == 0:
            return
        batch = starts.gather(starts.sample(n, generator))
        index = which.nonzero().squeeze(1).to(self.hanging.device)
        self.obs.index_copy_(0, index, batch.obs)
        self.present.index_copy_(0, index, batch.present)
        self.action.index_copy_(0, index, batch.action)

    def state_dict(self) -> dict:
        return {"obs": self.obs.cpu(), "present": self.present.cpu(), "action": self.action.cpu()}

    def load_state_dict(self, state: dict):
        device = self.hanging.device
        self.reset_to(Batch(*(state[k].to(device) for k in ("obs", "present", "action"))))

    def reset_hanging(self, envs: int):
        """Starts `envs` environments from the pendulum hanging still: every
        frame of the history at the hanging yaws, every tag seen, action 0."""
        h, device = self.history, self.hanging.device
        self.reset_to(
            Batch(
                self.hanging.expand(envs, h, NUM_TAGS, 2).clone(),
                torch.ones(envs, h, NUM_TAGS, dtype=torch.bool, device=device),
                torch.zeros(envs, h, device=device),
            )
        )

    def reset_to(self, batch: Batch):
        """Starts one environment from each window of `history` steps in `batch`."""
        assert batch.obs.shape[1] == self.history
        self.obs = batch.obs.clone()
        self.present = batch.present.clone()
        self.action = batch.action.clone()

    def observe(self) -> torch.Tensor:
        """The policy's input [N, features]: the latest `policy_window`
        observations, each with the action that was in effect before it."""
        p = self.policy_window
        x = step_features(self.obs[:, -p:], self.present[:, -p:], self.action[:, -p - 1 : -1])
        return x.flatten(1)

    @torch.no_grad()
    def step(self, bin_index: torch.Tensor) -> torch.Tensor:
        """Acts on the newest frame with the action in each bin [N]; returns
        the reward [N] of the frame that follows."""
        self.action[:, -1] = self.actions[bin_index]
        w = self.model.window
        nxt, logit, _, _ = self.model.predict(
            self.obs[:, -w:], self.present[:, -w:], self.action[:, -w:]
        )
        nxt = F.normalize(nxt, dim=-1)
        # The predicted frame, including tags then dropped as unseen.
        self.prediction = nxt
        if self.sample_missing:
            if self.generator is None:
                u = torch.rand_like(logit)
            else:
                u = torch.rand(logit.shape, generator=self.generator).to(logit.device)
            seen = u >= torch.sigmoid(logit)
        else:
            seen = torch.ones_like(logit, dtype=torch.bool)
        self.obs = torch.cat([self.obs[:, 1:], (nxt * seen[..., None])[:, None]], dim=1)
        self.present = torch.cat([self.present[:, 1:], seen[:, None]], dim=1)
        # The new frame's action is the policy's next choice.
        self.action = torch.cat([self.action[:, 1:], torch.zeros_like(self.action[:, :1])], dim=1)
        return reward(correct_yaws(nxt, self.hanging))


def reward(obs: torch.Tensor) -> torch.Tensor:
    """The sum over tags of cos(yaw), for corrected obs [..., NUM_TAGS, 2]."""
    return obs[..., 1].sum(-1)


# A tag counts as upright within 30° of vertical.
UPRIGHT_COS = 0.8660254


def upright(obs: torch.Tensor) -> torch.Tensor:
    """Whether every tag is upright, for corrected obs [..., NUM_TAGS, 2]."""
    return (obs[..., 1] > UPRIGHT_COS).all(-1)
