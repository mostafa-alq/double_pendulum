import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from ppo.actor import Actor

torch.manual_seed(0)

actor = Actor()

# A batch of 5 fake states -- shape (5, 6), matching [x, v, theta1, z1, theta2, z2].
# Values don't need to be physically real here, just the right shape.
states = torch.randn(5, 6)

dist = actor(states)
print('mu shape:', dist.mean.shape)
print('mu values:', dist.mean.squeeze(-1).tolist())
print('std (shared across all states):', torch.exp(actor.log_std).item())

action, log_prob = actor.act(states)
print('\nsampled action shape:', action.shape)
print('sampled actions:', action.squeeze(-1).tolist())
print('log_prob shape:', log_prob.shape)
print('log_probs:', log_prob.tolist())
