import torch

from imagination.agent_lib import Agent
from imagination.env_lib import ImaginedEnv, action_levels
from imagination.rollout_lib import closed_loop, from_hanging, write_rollout_video
from world_model.model_lib import WorldModel


def make(window=4, policy_window=3):
    torch.manual_seed(0)
    hanging = torch.tensor([[0.0, -1.0]] * 3)
    env = ImaginedEnv(WorldModel(window, 8, 1, 0.1), action_levels(16, 9), policy_window, True, hanging)
    agent = Agent(env.features, 8, 1, 9)
    return agent, env


def test_a_closed_loop_rollout_takes_actions_from_the_levels():
    agent, env = make()
    env.reset_hanging(5)
    run = closed_loop(agent, env, 7, greedy=False)
    assert run.prediction.shape == (7, 5, 3, 2)
    assert run.seen.shape == (7, 5, 3) and run.reward.shape == (7, 5)
    assert set(run.action.unique().tolist()) <= set(action_levels(16, 9).tolist())


def test_a_rollout_from_hanging_repeats_exactly():
    agent, env = make()
    a = from_hanging(agent, env, 50, seed=3)
    torch.rand(100)  # disturbing the global generator changes nothing
    b = from_hanging(agent, env, 50, seed=3)
    assert a.action.shape == (50, 1)
    assert torch.equal(a.action, b.action) and torch.equal(a.seen, b.seen)
    assert env.generator is None


def test_the_video_shows_the_start_then_each_step(tmp_path):
    agent, env = make()
    run = from_hanging(agent, env, 10)
    path = tmp_path / "v.mp4"
    write_rollout_video(run, env, path)
    assert path.stat().st_size > 0
