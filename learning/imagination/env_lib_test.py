import pytest
import torch
from torch.nn import functional as F

from common.data_lib import ACTION_SCALE, Recording, Windows
from imagination.env_lib import ImaginedEnv, action_levels, reward
from world_model.model_lib import STEP_FEATURES, WorldModel


def test_action_levels_are_centred_int8_steps():
    assert action_levels(16, 9).tolist() == [-64, -48, -32, -16, 0, 16, 32, 48, 64]
    with pytest.raises(ValueError):
        action_levels(16, 8)
    with pytest.raises(ValueError):
        action_levels(16, 17)


def test_reward_sums_the_cosines():
    hanging = torch.tensor([[0.0, -1.0]] * 3)
    upright = torch.tensor([[0.0, 1.0]] * 3)
    assert reward(hanging).item() == -3
    assert reward(upright).item() == 3


class Recorder(WorldModel):
    """A world model that records its inputs and predicts a fixed frame
    with a fixed missing logit."""

    def __init__(self, window, frame, logit):
        super().__init__(window, 4, 1, 1.0)
        self.frame, self.logit, self.inputs = frame, logit, []

    def predict(self, obs, present, action):
        self.inputs.append((obs.clone(), present.clone(), action.clone()))
        b = obs.shape[0]
        return self.frame.expand(b, -1, -1), torch.full((b, 3), self.logit), None, None


def make_env(window=3, policy_window=2, logit=-20.0, sample_missing=True):
    frame = F.normalize(torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), dim=-1)
    model = Recorder(window, frame, logit)
    env = ImaginedEnv(model, action_levels(16, 9), policy_window, sample_missing)
    steps = 10
    rec = Recording(
        "r",
        obs=torch.randn(steps, 3, 2).numpy(),
        present=torch.ones(steps, 3, dtype=torch.bool).numpy(),
        action=(torch.arange(steps, dtype=torch.float32) / ACTION_SCALE).numpy(),
    )
    starts = Windows([rec], env.history, "cpu")
    env.reset(starts, torch.tensor([0, 2]))
    return env, model, frame


def test_the_policy_sees_each_frame_with_the_action_before_it():
    env, _, _ = make_env(window=3, policy_window=2)
    assert env.history == 3
    x = env.observe().reshape(2, 2, STEP_FEATURES)
    # Starts at frames 0 and 2: the window is frames 0-2, the policy sees
    # frames 1 and 2 with the actions after frames 0 and 1.
    assert (x[:, :, -1] * ACTION_SCALE).tolist() == [[0, 1], [2, 3]]


def test_a_step_acts_on_the_newest_frame_and_appends_the_prediction():
    env, model, frame = make_env()
    r = env.step(torch.tensor([8, 0]))
    obs, present, action = model.inputs[0]
    assert (action * ACTION_SCALE).tolist() == [[0, 1, 64], [2, 3, -64]]
    assert torch.allclose(env.obs[:, -1], frame.expand(2, -1, -1))
    assert env.present[:, -1].all()
    assert (env.action * ACTION_SCALE).tolist() == [[1, 64, 0], [3, -64, 0]]
    assert r.tolist() == pytest.approx([reward(frame).item()] * 2)


def test_a_likely_miss_drops_the_tag_but_not_the_reward():
    env, _, frame = make_env(logit=20.0)
    r = env.step(torch.tensor([4, 4]))
    assert not env.present[:, -1].any()
    assert (env.obs[:, -1] == 0).all()
    assert r.tolist() == pytest.approx([reward(frame).item()] * 2)
    env, _, _ = make_env(logit=20.0, sample_missing=False)
    env.step(torch.tensor([4, 4]))
    assert env.present[:, -1].all()
