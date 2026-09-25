import csv
import os
import shutil
import sys
import time

import numpy as np

from cart_env import CartDoublePendulumEnv

ROOT = os.path.dirname(os.path.abspath(__file__))

# Game settings, keep these the same as game.py
GAME = dict(dt=1 / 60, x_max=4.0, max_force=30.0, damping=15.0,
            g=8.0665, M=1.0, m1=1.0, m2=1.0, l1=1.0, l2=1.5)

# Settings for each task
TASKS = {
    # Start upright and keep it there
    'balance': dict(
        env=dict(wall_crash=True), top_start_fraction=0.0,
        iterations=300, checkpoint_every=10, hidden=64, gamma=0.99, init_std=0.3,
        eval_seconds=60, eval_hold=30,
        render_start=[0.0, 0.0, np.pi + 0.1, 0.0, np.pi - 0.05, 0.0], render_seconds=8,
        render_iterations=[0, 30, 60, 90, 140, 230],
    ),
    # Start at the bottom, swing up and balance
    'swingup': dict(
        env=dict(start_angle=0.0, perturb=0.1, fall_angle=np.inf, max_steps=1200, wall_crash=True),
        # Some start near the top so it learns balancing beats spinning
        top_start_fraction=0.25,
        iterations=1500, checkpoint_every=25, hidden=256, gamma=0.995, init_std=0.5,
        eval_seconds=20, eval_hold=5,
        render_start=[0.0, 0.0, 0.05, 0.0, -0.05, 0.0], render_seconds=15,
        render_iterations=[0, 250, 500, 750, 1000, 1500],
    ),
}

# Consts
N_ENVS = 128
ROLLOUT = 128
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 10
MINIBATCH = 2048
LR = 3e-4
REWARD_SCALE = 0.1
MAX_GRAD_NORM = 0.5
N_OBS = 8
UPRIGHT_ANGLE = 0.3
LOG_2PI = np.log(2 * np.pi)


# Output folders for a task
def paths(task):
    base = os.path.join(ROOT, 'outputs', task)
    return {
        'policy': os.path.join(base, 'policy.npz'),
        'checkpoints': os.path.join(base, 'checkpoints'),
        'progress': os.path.join(base, 'progress'),
        'log': os.path.join(base, 'training_log.csv'),
    }


# Network, same as neural.py but with numpy matrices
class MLP:
    def __init__(self, n_in, n_hidden, n_out, rng, out_scale=1.0):
        # Weights are (inputs, neurons), flipped from neural.py
        self.params = {
            'w1': rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
            'b1': np.zeros(n_hidden),
            'w2': rng.normal(0, out_scale / np.sqrt(n_hidden), (n_hidden, n_out)),
            'b2': np.zeros(n_out),
        }

    def forward(self, x):
        p = self.params
        pre = x @ p['w1'] + p['b1']
        hidden = np.maximum(pre, 0)
        out = hidden @ p['w2'] + p['b2']
        return out, (x, pre, hidden)

    # Same maths as derivatives() in neural.py
    def backward(self, cache, d_out):
        x, pre, hidden = cache
        p = self.params
        d_pre = (d_out @ p['w2'].T) * (pre > 0)
        return {
            'w2': hidden.T @ d_out,
            'b2': d_out.sum(0),
            'w1': x.T @ d_pre,
            'b1': d_pre.sum(0),
        }


# Adam optimiser, gradient descent with a step size per weight
class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            # Running averages of the gradient and its square
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Shrink the gradients if they get too big
def clip_grad_norm(grads, max_norm):
    norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / norm)


# Environment helpers
def make_env(task, n, rng, **overrides):
    return CartDoublePendulumEnv(n=n, rng=rng, **{**GAME, **TASKS[task]['env'], **overrides})


# What the policy sees, angles as sin and cos so there's no wrap around
def observe(state):
    x, v, the1, z1, the2, z2 = state
    return np.stack([x / GAME['x_max'], v / 5, np.sin(the1), np.cos(the1), z1 / 10,
                     np.sin(the2), np.cos(the2), z2 / 10], axis=-1)


# Reset pendulums, putting some of them near the top
def reset_envs(env, idx, rng, top_fraction):
    env.reset(idx)
    top = idx[rng.random(len(idx)) < top_fraction]
    env.state[2, top] = np.pi + rng.uniform(-0.15, 0.15, len(top))
    env.state[4, top] = np.pi + rng.uniform(-0.15, 0.15, len(top))
    return env.state.copy()


# True when both links are close to upright
def upright(state):
    dev1 = (state[2] % (2 * np.pi)) - np.pi
    dev2 = (state[4] % (2 * np.pi)) - np.pi
    return (np.abs(dev1) < UPRIGHT_ANGLE) & (np.abs(dev2) < UPRIGHT_ANGLE)


# PPO
def gaussian_logp(a, mu, log_std):
    z = (a - mu) / np.exp(log_std)
    return -0.5 * z ** 2 - log_std - 0.5 * LOG_2PI


# Actor loss, favours actions that did better than expected
def actor_loss_and_grads(actor, obs, act, logp_old, adv):
    out, cache = actor.forward(obs)
    mu = out[:, 0]
    log_std = actor.params['log_std'][0]
    std = np.exp(log_std)
    logp = gaussian_logp(act, mu, log_std)
    ratio = np.exp(logp - logp_old)
    s1 = ratio * adv
    s2 = np.clip(ratio, 1 - CLIP, 1 + CLIP) * adv
    loss = -np.mean(np.minimum(s1, s2))

    # Gradients
    d_logp = -(adv * ratio * (s1 <= s2)) / len(adv)
    d_mu = d_logp * (act - mu) / std ** 2
    grads = actor.backward(cache, d_mu[:, None])
    grads['log_std'] = np.array([np.sum(d_logp * (((act - mu) / std) ** 2 - 1))])
    return loss, grads


# Critic loss, squared error against the actual returns
def critic_loss_and_grads(critic, obs, returns):
    out, cache = critic.forward(obs)
    err = out[:, 0] - returns
    loss = 0.5 * np.mean(err ** 2)
    grads = critic.backward(cache, (err / len(err))[:, None])
    return loss, grads


# Critic's guess of how good a state is
def value(critic, obs):
    return critic.forward(obs)[0][:, 0]


# Force the trained policy applies, no noise
def policy_force(actor, state):
    mu = actor.forward(observe(state))[0][..., 0]
    return GAME['max_force'] * np.clip(mu, -1, 1)


# Saving and loading
def checkpoint_path(task, iteration):
    return os.path.join(paths(task)['checkpoints'], f'iter_{iteration:04d}.npz')


def save_policy(actor, path):
    np.savez(path, **actor.params)


def load_policy(path):
    data = np.load(path)
    actor = MLP(N_OBS, data['w1'].shape[1], 1, np.random.default_rng(0))
    actor.params = {k: data[k] for k in data.files}
    return actor


# Success rate from the task's normal start
def evaluate(task, actor, n=200, seed=123):
    return run_eval(task, actor, n, seed)[0]


# Returns success rate, seconds upright and how many touched the wall. Success means upright at the end and never touching the wall
def run_eval(task, actor, n, seed):
    cfg = TASKS[task]
    env = make_env(task, n, np.random.default_rng(seed), max_steps=10 ** 9, wall_crash=False)
    state = env.reset()
    steps = int(cfg['eval_seconds'] / env.dt)
    hold_from = steps - int(cfg['eval_hold'] / env.dt)
    ok = np.ones(n, dtype=bool)
    touched = np.zeros(n, dtype=bool)
    upright_steps = np.zeros(n)
    for t in range(steps):
        state, _, _, info = env.step(policy_force(actor, state))
        up = upright(state)
        upright_steps += up
        touched |= info['hit_wall']
        if t >= hold_from:
            ok &= up
    return (ok & ~touched).mean(), upright_steps.mean() * env.dt, touched.mean()


# Keep the best checkpoint since PPO can get worse near the end
def select_best(task, n=100, seed=321):
    out = paths(task)
    files = sorted(f for f in os.listdir(out['checkpoints']) if f.endswith('.npz'))
    best_score, best_file = None, None
    for f in files:
        success, up, wall = run_eval(task, load_policy(os.path.join(out['checkpoints'], f)), n, seed)
        print(f'{f}: success {success:4.0%}  upright {up:5.1f}s  touched wall {wall:4.0%}', flush=True)
        if best_score is None or (success, up) >= best_score:
            best_score, best_file = (success, up), f
    shutil.copyfile(os.path.join(out['checkpoints'], best_file), out['policy'])
    print(f'best: {best_file} -> {out["policy"]}')
    return best_file


# Training loop
def train(task, seed=0):
    # Setup
    cfg = TASKS[task]
    out = paths(task)
    gamma = cfg['gamma']
    rng = np.random.default_rng(seed)
    env = make_env(task, N_ENVS, rng)
    actor = MLP(N_OBS, cfg['hidden'], 1, rng, out_scale=0.01)
    actor.params['log_std'] = np.array([np.log(cfg['init_std'])])
    critic = MLP(N_OBS, cfg['hidden'], 1, rng)
    actor_opt = Adam(actor.params, LR)
    critic_opt = Adam(critic.params, LR)

    # Log file
    os.makedirs(out['checkpoints'], exist_ok=True)
    save_policy(actor, checkpoint_path(task, 0))
    log_file = open(out['log'], 'w', newline='')
    log = csv.writer(log_file)
    log.writerow(['iteration', 'env_steps', 'episode_return', 'episode_seconds', 'upright_seconds',
                  'eval_success', 'eval_upright_seconds', 'noise_std', 'wall_seconds'])

    top_fraction = cfg['top_start_fraction']
    obs = observe(reset_envs(env, np.arange(N_ENVS), rng, top_fraction))
    ep_return = np.zeros(N_ENVS)
    ep_len = np.zeros(N_ENVS)
    ep_upright = np.zeros(N_ENVS)
    recent = []
    start_time = time.time()

    for it in range(1, cfg['iterations'] + 1):
        b_obs = np.zeros((ROLLOUT, N_ENVS, N_OBS))
        b_act, b_logp, b_rew, b_done, b_val = (np.zeros((ROLLOUT, N_ENVS)) for _ in range(5))

        # Collect a rollout
        for t in range(ROLLOUT):
            # Noisy force from the policy
            mu = actor.forward(obs)[0][:, 0]
            log_std = actor.params['log_std'][0]
            act = mu + np.exp(log_std) * rng.standard_normal(N_ENVS)
            b_obs[t], b_act[t] = obs, act
            b_logp[t] = gaussian_logp(act, mu, log_std)
            b_val[t] = value(critic, obs)

            state, reward, done, info = env.step(GAME['max_force'] * np.clip(act, -1, 1))
            next_obs = observe(state)
            ep_return += reward
            ep_len += 1
            ep_upright += upright(state)

            # Timed out episodes get the critic's guess of what came next
            r = reward * REWARD_SCALE
            truncated = info['timeout'] & ~info['fell'] & ~info['crashed']
            if truncated.any():
                r[truncated] += gamma * value(critic, next_obs[truncated])
            b_rew[t], b_done[t] = r, done

            # Reset finished pendulums
            if done.any():
                idx = np.flatnonzero(done)
                recent.extend(zip(ep_return[idx], ep_len[idx], ep_upright[idx]))
                ep_return[idx] = 0
                ep_len[idx] = 0
                ep_upright[idx] = 0
                next_obs[idx] = observe(reset_envs(env, idx, rng, top_fraction)[:, idx])
            obs = next_obs

        # Advantages, how much better each action did than the critic expected
        adv = np.zeros_like(b_rew)
        gae = 0.0
        next_val = value(critic, obs)
        for t in reversed(range(ROLLOUT)):
            nonterminal = 1.0 - b_done[t]
            delta = b_rew[t] + gamma * next_val * nonterminal - b_val[t]
            gae = delta + gamma * LAMBDA * nonterminal * gae
            adv[t] = gae
            next_val = b_val[t]
        returns = adv + b_val

        # Flatten and normalise
        f_obs = b_obs.reshape(-1, N_OBS)
        f_act, f_logp, f_ret = b_act.ravel(), b_logp.ravel(), returns.ravel()
        f_adv = adv.ravel()
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        # Update actor and critic
        for _ in range(EPOCHS):
            perm = rng.permutation(len(f_adv))
            for s in range(0, len(perm), MINIBATCH):
                mb = perm[s:s + MINIBATCH]
                _, g = actor_loss_and_grads(actor, f_obs[mb], f_act[mb], f_logp[mb], f_adv[mb])
                clip_grad_norm(g, MAX_GRAD_NORM)
                actor_opt.step(g)
                _, g = critic_loss_and_grads(critic, f_obs[mb], f_ret[mb])
                clip_grad_norm(g, MAX_GRAD_NORM)
                critic_opt.step(g)

        # Checkpoint, log and print
        if it % cfg['checkpoint_every'] == 0:
            recent = recent[-200:]
            if recent:
                mean_ret, mean_len, mean_up = np.mean(recent, axis=0)
            else:
                mean_ret = mean_len = mean_up = float('nan')
            std = np.exp(actor.params['log_std'][0])
            eval_success, eval_up, _ = run_eval(task, actor, n=32, seed=it)
            elapsed = time.time() - start_time
            steps_done = it * ROLLOUT * N_ENVS
            log.writerow([it, steps_done, f'{mean_ret:.2f}', f'{mean_len * env.dt:.3f}',
                          f'{mean_up * env.dt:.3f}', f'{eval_success:.3f}', f'{eval_up:.3f}',
                          f'{std:.4f}', f'{elapsed:.0f}'])
            log_file.flush()
            save_policy(actor, checkpoint_path(task, it))
            save_policy(actor, out['policy'])
            print(f'iter {it:4d}  steps {steps_done:8d}  episode return {mean_ret:8.1f}  '
                  f'upright {mean_up * env.dt:5.2f}s  test: upright {eval_up:5.2f}s, '
                  f'success {eval_success:4.0%}  noise std {std:.3f}  {elapsed:5.0f}s', flush=True)

    log_file.close()
    return actor


# One GIF per checkpoint plus the learning curve
def render(task, iterations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualize import animate_trajectory

    cfg = TASKS[task]
    out = paths(task)
    os.makedirs(out['progress'], exist_ok=True)
    rows = list(csv.DictReader(open(out['log'])))
    steps_at = {int(r['iteration']): int(r['env_steps']) for r in rows}
    steps_at[0] = 0

    for it in iterations:
        # Same starting position every time so the GIFs compare fairly
        actor = load_policy(checkpoint_path(task, it))
        env = make_env(task, None, np.random.default_rng(0), max_steps=10 ** 9)
        env.reset()
        env.state = np.array(cfg['render_start'])
        states, forces = [env.state.copy()], []
        for _ in range(int(cfg['render_seconds'] / env.dt)):
            f = float(policy_force(actor, env.state))
            env.step(f)
            states.append(env.state.copy())
            forces.append(f)
        path = os.path.join(out['progress'], f'iter_{it:04d}.gif')
        animate_trajectory(np.array(states), env.dt, GAME['l1'], GAME['l2'],
                           series={'force (N)': np.array(forces)}, x_max=GAME['x_max'],
                           fixed_camera=True, save_path=path,
                           title=f'{task}: iteration {it} ({steps_at.get(it, 0):,} simulation steps)')
        print(f'saved {path}')

    # Learning curve
    column = next(c for c in ('eval_upright_seconds', 'upright_seconds', 'episode_seconds') if c in rows[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([int(r['env_steps']) for r in rows], [float(r[column]) for r in rows])
    ax.set_xlabel('simulation steps')
    ax.set_ylabel(f'seconds upright (out of {cfg["eval_seconds"]})')
    ax.set_title(f'Training progress ({task})')
    fig.tight_layout()
    fig.savefig(os.path.join(out['progress'], 'learning_curve.png'), dpi=120)
    print('saved learning curve')


# Command line outputs
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == 'render' and args[1] in TASKS:
        chosen = [int(a) for a in args[2:]] or TASKS[args[1]]['render_iterations']
        render(args[1], chosen)
    elif len(args) == 2 and args[0] == 'select' and args[1] in TASKS:
        select_best(args[1])
    elif len(args) == 1 and args[0] in TASKS:
        train(args[0])
        select_best(args[0])
        cfg = TASKS[args[0]]
        print(f'best policy: never touched the wall and upright for the last {cfg["eval_hold"]}s '
              f'of a {cfg["eval_seconds"]}s run: {evaluate(args[0], load_policy(paths(args[0])["policy"])):.0%}')
    else:
        print('usage: python train.py balance|swingup\n'
              '       python train.py select balance|swingup\n'
              '       python train.py render balance|swingup [iterations...]')
