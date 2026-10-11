import numpy as np

from physics import step as physics_step, Params, total_energy


# Training environment, runs one pendulum or n of them at once
class CartDoublePendulumEnv:
    def __init__(self, n=None, dt=0.01, max_steps=1000, x_max=2.4, max_force=30.0,
                 fall_angle=np.pi / 2, perturb=0.15, start_angle=np.pi, damping=15.0, wall_crash=False,
                 wall_cost=0.0, energy_cost=0.0, center_cost=0.1, fall_after_catch=False, alive_bonus=0.0,
                 g=9.80665, M=1.0, m1=1.0, m2=1.0, l1=1, l2=1.0, rng=None):
        self.n = n
        self.shape = () if n is None else (n,)
        self.dt = dt
        self.max_steps = max_steps
        self.x_max = x_max
        self.max_force = max_force
        self.fall_angle = fall_angle
        self.perturb = perturb
        self.start_angle = start_angle
        self.damping = damping
        self.wall_crash = wall_crash
        self.wall_cost = wall_cost
        self.energy_cost = energy_cost
        self.center_cost = center_cost
        self.fall_after_catch = fall_after_catch
        self.alive_bonus = alive_bonus
        self.caught = np.zeros(self.shape, dtype=bool)
        self.p = Params(g=g, M=M, m1=m1, m2=m2, l1=l1, l2=l2)
        self.top_energy = (m1 * l1 + m2 * (l1 + l2)) * g
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = None
        self.steps = np.zeros(self.shape, dtype=int)

    # idx picks which pendulums to reset (all of them by default)
    def reset(self, idx=None, rng=None):
        rng = rng if rng is not None else self.rng
        if idx is None or self.state is None:
            self.state = np.zeros((6,) + self.shape)
            idx = ...
        size = self.state[0, idx].shape
        self.state[:, idx] = 0.0
        self.state[2, idx] = self.start_angle + rng.uniform(-self.perturb, self.perturb, size)
        self.state[4, idx] = self.start_angle + rng.uniform(-self.perturb, self.perturb, size)
        self.steps[idx] = 0
        self.caught[idx] = False
        return self.state.copy()

    # Push the cart, move the physics on one step and score it
    def step(self, action):
        action = np.asarray(action, dtype=float)
        F = np.where(action == 0, -self.damping * self.state[1],
                     np.clip(action, -self.max_force, self.max_force))
        self.state = physics_step(self.state, F, self.dt, self.p)
        self.steps += 1

        # Stop the cart dead at the end of the track
        hit_wall = np.abs(self.state[0]) > self.x_max
        self.state[0] = np.where(hit_wall, np.sign(self.state[0]) * self.x_max, self.state[0])
        self.state[1] = np.where(hit_wall, 0.0, self.state[1])

        # Points for being upright and near the middle
        x, v, the1, z1, the2, z2 = self.state
        reward =self.alive_bonus - np.cos(the1) - np.cos(the2) - self.center_cost * (x / self.x_max) ** 2
        if self.energy_cost:
            p = self.p
            energy = total_energy(self.state.T, p.g, p.M, p.m1, p.m2, p.l1, p.l2)
            # Lose points for having the wrong energy to stand at the top
            reward = reward - self.energy_cost * np.minimum(np.abs(energy - self.top_energy) / self.top_energy, 2.0)

        # Check if it fell or crashed
        dev1 = (the1 % (2 * np.pi)) - np.pi
        dev2 = (the2 % (2 * np.pi)) - np.pi
        fell = (np.abs(dev1) > self.fall_angle) | (np.abs(dev2) > self.fall_angle)
        if self.fall_after_catch:
            self.caught |= (np.abs(dev1) < 0.3) & (np.abs(dev2) < 0.3)
            fell = fell & self.caught
        timeout = self.steps >= self.max_steps
        crashed = hit_wall & self.wall_crash
        reward = reward - 50.0 * (fell | crashed) - self.wall_cost * hit_wall
        done = fell | crashed | timeout

        if self.n is None:
            reward, fell, hit_wall, crashed, timeout, done = (float(reward), bool(fell), bool(hit_wall),
                                                              bool(crashed), bool(timeout), bool(done))
        info = {'fell': fell, 'hit_wall': hit_wall, 'crashed': crashed, 'timeout': timeout}
        return self.state.copy(), reward, done, info
