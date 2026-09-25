import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_env import CartDoublePendulumEnv
from train import MLP, actor_loss_and_grads, critic_loss_and_grads, gaussian_logp

EPS = 1e-6


# The same nudge-up/nudge-down check as neural.py's check_gradients()
def check(label, net, loss_and_grads):
    _, grads = loss_and_grads()
    for name, p in net.params.items():
        worst = 0.0
        for idx in np.ndindex(p.shape):
            original = p[idx]
            p[idx] = original + EPS
            loss_plus = loss_and_grads()[0]
            p[idx] = original - EPS
            loss_minus = loss_and_grads()[0]
            p[idx] = original
            worst = max(worst, abs((loss_plus - loss_minus) / (2 * EPS) - grads[name][idx]))
        print(f'{label} {name}: worst gap {worst:.2e}')
        assert worst < 1e-6


rng = np.random.default_rng(1)

# Check 1: the network's backprop on a plain squared-error loss
net = MLP(8, 16, 3, rng)
x = rng.normal(size=(20, 8))
targets = rng.normal(size=(20, 3))


def mse_loss_and_grads():
    out, cache = net.forward(x)
    return np.mean(np.sum((out - targets) ** 2, axis=1)), net.backward(cache, 2 * (out - targets) / len(x))


check('mlp', net, mse_loss_and_grads)

# Check 2: both PPO losses
n = 64
obs = rng.normal(size=(n, 8))
actor = MLP(8, 16, 1, rng)
actor.params['log_std'] = np.array([np.log(0.4)])
critic = MLP(8, 16, 1, rng)
mu = actor.forward(obs)[0][:, 0]
act = mu + 0.4 * rng.normal(size=n)
logp_old = gaussian_logp(act, mu, np.log(0.4)) + rng.normal(0, 0.15, n)
adv = rng.normal(size=n)
returns = rng.normal(size=n)

check('actor', actor, lambda: actor_loss_and_grads(actor, obs, act, logp_old, adv))
check('critic', critic, lambda: critic_loss_and_grads(critic, obs, returns))

# Check 3: many pendulums at once behave exactly like the same pendulums run one at a time
n_env = 8
batch = CartDoublePendulumEnv(n=n_env, rng=rng)
batch.reset()
singles = []
for i in range(n_env):
    e = CartDoublePendulumEnv()
    e.state = batch.state[:, i].copy()
    singles.append(e)
actions = rng.uniform(-45, 45, (300, n_env))
actions[::7] = 0.0

worst = 0.0
alive = np.ones(n_env, dtype=bool)
for t in range(300):
    sb, rb, db, ib = batch.step(actions[t])
    for i in range(n_env):
        if not alive[i]:
            continue
        s, r, d, info = singles[i].step(actions[t, i])
        worst = max(worst, np.abs(s - sb[:, i]).max(), abs(r - rb[i]))
        assert d == db[i] and all(info[k] == ib[k][i] for k in info)
        if d:
            alive[i] = False
print(f'batched env vs single env: worst gap {worst:.2e}')
assert worst < 1e-9
print('All training checks passed.')
