"""PPO's advantage estimate and loss, after CleanRL's ppo_continuous_action_isaacgym.py.

Episodes here only end by truncation, so the return is always bootstrapped
from the critic's value of the state after the last step, never from 0.
"""
import math

import torch
from torch.nn import functional as F

from imagination.agent_lib import Agent


def gae(rewards, values, last_value, gamma: float, lam: float):
    """Advantages and returns [T, N] from rewards and values [T, N] and the
    value [N] of the state after the last step."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def discounted_returns(rewards, gamma: float, tail: float = 0.01):
    """The discounted return [T', N] from each of the first T' frames of
    rewards [T, N], for the frames with at least as many ahead as it takes
    the discount to fall to `tail`: those whose return a finite run
    measures to within that share of its weight."""
    ahead = math.ceil(math.log(tail) / math.log(gamma))
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns[: max(len(rewards) - ahead, 0)]


def ppo_loss(agent: Agent, x, bins, old_logp, advantages, returns, cfg: dict):
    """The clipped PPO loss on a minibatch, and statistics to log."""
    dist = agent.policy(x)
    logp = dist.log_prob(bins)
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clip = cfg["clip"]
    pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)).mean()
    v_loss = 0.5 * F.mse_loss(agent.value(x), returns)
    entropy = dist.entropy().mean()
    loss = pg_loss - cfg["ent_coef"] * entropy + cfg["vf_coef"] * v_loss
    with torch.no_grad():
        stats = {
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "clipfrac": ((ratio - 1).abs() > clip).float().mean(),
        }
    return loss, stats
