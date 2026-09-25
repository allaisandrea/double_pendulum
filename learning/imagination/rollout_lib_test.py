import torch

from common.data_lib import Batch
from imagination.agent_lib import Agent
from imagination.env_lib import ImaginedEnv, action_levels
from imagination.rollout_lib import closed_loop, video_from_rest
from world_model.model_lib import WorldModel


def make(window=4, policy_window=3):
    torch.manual_seed(0)
    env = ImaginedEnv(WorldModel(window, 8, 1, 0.1), action_levels(16, 9), policy_window, True)
    agent = Agent(env.features, 8, 1, 9)
    h = env.history
    rest = Batch(
        torch.tensor([[0.0, -1.0]] * 3).expand(1, h, 3, 2).clone(),
        torch.ones(1, h, 3, dtype=torch.bool),
        torch.zeros(1, h),
    )
    return agent, env, rest


def test_a_closed_loop_rollout_takes_actions_from_the_levels():
    agent, env, rest = make()
    env.reset_to(Batch(*(t.expand(5, *t.shape[1:]) for t in (rest.obs, rest.present, rest.action))))
    run = closed_loop(agent, env, 7, greedy=False)
    assert run.prediction.shape == (7, 5, 3, 2)
    assert run.seen.shape == (7, 5, 3) and run.reward.shape == (7, 5)
    assert set(run.action.unique().tolist()) <= set(action_levels(16, 9).tolist())


def test_the_video_from_rest_shows_the_rest_then_each_step(tmp_path):
    agent, env, rest = make()
    path = tmp_path / "v.mp4"
    run = video_from_rest(agent, env, rest, path, seconds=0.08)
    assert run.prediction.shape[0] == 10
    assert path.stat().st_size > 0
