import numpy as np
import pytest
import torch
from torch.nn import functional as F

from common.data_lib import ACTION_SCALE, Batch, Recording, Windows
from imagination.env_lib import ImaginedEnv, Starts, action_levels, reward

STRAIGHT = torch.tensor([[0.0, -1.0]] * 3)
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


def make_env(window=3, policy_window=2, logit=-20.0, sample_missing=True, hanging=STRAIGHT):
    frame = F.normalize(torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), dim=-1)
    model = Recorder(window, frame, logit)
    env = ImaginedEnv(model, action_levels(16, 9), policy_window, sample_missing, hanging)
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


def test_the_reward_counts_each_arm_from_its_own_hanging_yaw():
    # Tag 0 hangs at 170°: a prediction at 350° is its upright, and scores 1.
    tilt = torch.tensor(np.radians(170.0), dtype=torch.float32)
    hanging = STRAIGHT.clone()
    hanging[0] = torch.stack([torch.sin(tilt), torch.cos(tilt)])
    env, model, _ = make_env(hanging=hanging)
    up = torch.tensor(np.radians(350.0), dtype=torch.float32)
    model.frame = torch.stack([torch.stack([torch.sin(up), torch.cos(up)]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])])
    assert env.step(torch.tensor([4, 4])).tolist() == pytest.approx([3.0, 3.0], abs=1e-5)


def test_a_hanging_start_is_still_and_seen():
    env, _, _ = make_env()
    env.reset_hanging(4)
    assert env.obs.shape == (4, env.history, 3, 2)
    assert torch.equal(env.obs, STRAIGHT.expand_as(env.obs))
    assert env.present.all() and (env.action == 0).all()


def test_restarting_replaces_only_the_chosen_environments():
    env, _, _ = make_env()
    before = env.obs.clone()
    starts = Windows(
        [Recording("s", torch.ones(10, 3, 2).numpy(), torch.ones(10, 3, dtype=torch.bool).numpy(), torch.zeros(10).numpy())],
        env.history,
        "cpu",
    )
    env.restart(torch.tensor([False, True]), starts, torch.Generator().manual_seed(0))
    assert torch.equal(env.obs[0], before[0])
    assert (env.obs[1] == 1).all()


def test_an_environment_state_survives_a_save():
    env, _, _ = make_env()
    env.step(torch.tensor([8, 0]))
    state = env.state_dict()
    other, _, _ = make_env()
    other.load_state_dict(state)
    assert torch.equal(other.obs, env.obs) and torch.equal(other.action, env.action)


def test_an_arm_over_its_speed_limit_costs_the_square_of_the_excess():
    # Every frame of the start has the arms at 0°; the model predicts 72°,
    # 0° and 180° next: a fifth of a turn in 8 ms is 25 rev/s.
    angles = torch.tensor([0.2, 0.0, 0.5]) * 2 * np.pi
    frame = torch.stack([angles.sin(), angles.cos()], -1)
    model = Recorder(3, frame, -20.0)
    env = ImaginedEnv(model, action_levels(16, 9), 2, False, STRAIGHT, [20.0, 1.0, 30.0], 0.5)
    zero = torch.tensor([[0.0, 1.0]] * 3)
    env.reset_to(Batch(zero.expand(2, env.history, 3, 2).clone(), torch.ones(2, env.history, 3, dtype=torch.bool), torch.zeros(2, env.history)))
    r = env.step(torch.tensor([4, 4]))
    assert env.speed[0].tolist() == pytest.approx([25.0, 0.0, 62.5], rel=1e-4)
    # Arm 0 is 5 over its limit, arm 2 32.5: 0.5 * (25 + 1056.25).
    assert env.penalty.tolist() == pytest.approx([540.625] * 2, rel=1e-4)
    assert (r + env.penalty).tolist() == pytest.approx(env.cos_sum.tolist(), abs=1e-3)


def test_an_arm_never_seen_is_not_penalised():
    angles = torch.tensor([0.2, 0.2, 0.2]) * 2 * np.pi
    model = Recorder(3, torch.stack([angles.sin(), angles.cos()], -1), -20.0)
    env = ImaginedEnv(model, action_levels(16, 9), 2, False, STRAIGHT, [1.0] * 3, 1.0)
    present = torch.ones(1, env.history, 3, dtype=torch.bool)
    present[..., 1] = False
    env.reset_to(Batch(torch.tensor([[0.0, 1.0]] * 3).expand(1, env.history, 3, 2).clone(), present, torch.zeros(1, env.history)))
    env.step(torch.tensor([4]))
    assert env.speed[0, 1].item() == 0.0 and env.speed[0, 0].item() > 1.0


def test_upright_starts_draw_their_share_from_upright_windows():
    up, down = [[0.0, 1.0]] * 3, [[0.0, -1.0]] * 3
    obs = torch.tensor([down] * 10 + [up] + [down] * 10)
    rec = Recording("s", obs.numpy(), torch.ones(21, 3, dtype=torch.bool).numpy(), torch.zeros(21).numpy())
    windows = Windows([rec], 3, "cpu")
    starts = Starts(windows, STRAIGHT, upright_fraction=0.5)
    assert starts.upright.tolist() == [8]  # the window ending at frame 10
    index = starts.sample(4000, torch.Generator().manual_seed(0))
    share = (index == 8).float().mean().item()
    assert 0.47 < share < 0.54  # half on purpose, and 1 in 19 of the rest
    assert (Starts(windows, STRAIGHT).sample(4000, torch.Generator().manual_seed(0)) == 8).float().mean() < 0.08
