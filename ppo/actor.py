import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Policy network: state -> distribution over continuous force."""

    def __init__(self, state_dim=6, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # outputs mu
        )
        self.log_std = nn.Parameter(torch.zeros(1))  # state-independent, learned

    def forward(self, state):
        mu = self.net(state)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
