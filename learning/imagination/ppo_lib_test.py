import pytest
import torch

from imagination.agent_lib import Agent
from imagination.ppo_lib import gae, ppo_loss


def test_gae_with_lambda_one_is_the_bootstrapped_discounted_return():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.25], [0.0]])
    adv, ret = gae(rewards, values, torch.tensor([10.0]), gamma=0.5, lam=1.0)
    assert ret[:, 0].tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 10, 2 + 0.5 * 3 + 0.25 * 10, 3 + 0.5 * 10])
    assert torch.allclose(adv, ret - values)


def test_gae_with_lambda_zero_is_the_one_step_td_error():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[4.0], [6.0]])
    adv, _ = gae(rewards, values, torch.tensor([8.0]), gamma=0.5, lam=0.0)
    assert adv[:, 0].tolist() == pytest.approx([1 + 0.5 * 6 - 4, 2 + 0.5 * 8 - 6])


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
