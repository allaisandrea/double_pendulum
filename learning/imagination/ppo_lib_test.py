import pytest
import torch

from imagination.agent_lib import Agent
from imagination.ppo_lib import discounted_returns, gae, ppo_loss


def after(values, last_value):
    """The value of the state each step led to, with no resets."""
    return torch.cat([values[1:], last_value[None]])


def test_gae_with_lambda_one_is_the_bootstrapped_discounted_return():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.25], [0.0]])
    no_reset = torch.zeros(3, 1, dtype=torch.bool)
    adv, ret = gae(rewards, values, after(values, torch.tensor([10.0])), no_reset, gamma=0.5, lam=1.0)
    assert ret[:, 0].tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 10, 2 + 0.5 * 3 + 0.25 * 10, 3 + 0.5 * 10])
    assert torch.allclose(adv, ret - values)


def test_gae_with_lambda_zero_is_the_one_step_td_error():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[4.0], [6.0]])
    no_reset = torch.zeros(2, 1, dtype=torch.bool)
    adv, _ = gae(rewards, values, after(values, torch.tensor([8.0])), no_reset, gamma=0.5, lam=0.0)
    assert adv[:, 0].tolist() == pytest.approx([1 + 0.5 * 6 - 4, 2 + 0.5 * 8 - 6])


def test_a_reset_bootstraps_from_where_the_episode_was_going_and_stops_there():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.25], [0.0]])
    # Reset after step 0: its episode would have gone on to a state worth 7.
    next_values = torch.tensor([[7.0], [0.0], [10.0]])
    reset = torch.tensor([[True], [False], [False]])
    _, ret = gae(rewards, values, next_values, reset, gamma=0.5, lam=1.0)
    assert ret[:, 0].tolist() == pytest.approx([1 + 0.5 * 7, 2 + 0.5 * 3 + 0.25 * 10, 3 + 0.5 * 10])


def test_a_fresh_policy_is_near_uniform_and_the_loss_has_no_ratio_change():
    torch.manual_seed(0)
    agent = Agent(features=5, hidden=8, layers=2, bins=9)
    x = torch.randn(64, 5)
    dist = agent.policy(x)
    assert dist.entropy().mean().item() == pytest.approx(torch.log(torch.tensor(9.0)).item(), rel=1e-3)
    bins = dist.sample()
    cfg = {"clip": 0.2, "ent_coef": 0.01, "vf_coef": 0.5}
    _, stats = ppo_loss(agent, x, bins, dist.log_prob(bins).detach(), torch.randn(64), torch.randn(64), cfg)
    assert stats["approx_kl"].item() == pytest.approx(0.0, abs=1e-6)
    assert stats["clipfrac"].item() == 0.0


def test_discounted_returns_keep_only_frames_the_run_measures():
    rewards = torch.ones(10, 2)
    # At gamma 0.5 the discount falls to 1/8 in 3 frames, so the first 7 count.
    ret = discounted_returns(rewards, gamma=0.5, tail=0.125)
    assert ret.shape == (7, 2)
    assert ret[-1, 0].item() == pytest.approx(sum(0.5**k for k in range(4)))
    assert ret[0, 0].item() == pytest.approx(sum(0.5**k for k in range(10)))
    assert discounted_returns(rewards, gamma=0.99).shape == (0, 2)
