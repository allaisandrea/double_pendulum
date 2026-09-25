"""The policy and its critic, as in CleanRL's PPO: two separate tanh MLPs,
orthogonally initialised, the policy's last layer small so it starts close
to uniform over the action bins."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer(a: int, b: int, std: float = np.sqrt(2)) -> nn.Linear:
    linear = nn.Linear(a, b)
    nn.init.orthogonal_(linear.weight, std)
    nn.init.zeros_(linear.bias)
    return linear


def mlp(features: int, hidden: int, layers: int, out: int, out_std: float) -> nn.Sequential:
    dims = [features] + [hidden] * layers
    blocks = []
    for a, b in zip(dims, dims[1:]):
        blocks += [layer(a, b), nn.Tanh()]
    return nn.Sequential(*blocks, layer(dims[-1], out, out_std))


class Agent(nn.Module):
    def __init__(self, features: int, hidden: int, layers: int, bins: int):
        super().__init__()
        self.actor = mlp(features, hidden, layers, bins, 0.01)
        self.critic = mlp(features, hidden, layers, 1, 1.0)

    def policy(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)
