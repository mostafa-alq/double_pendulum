import numpy as np

from physics import step as physics_step, Params


# n=None runs one pendulum (state shape (6,)); n=k runs k pendulums at once (state shape (6, k)).
# start_angle=np.pi starts upright (balancing); start_angle=0 starts hanging down (swing-up).
# wall_crash=True treats hitting the track limit like falling (-50, episode over). Training needs it:
# the wall stops the cart instantly without affecting the pendulum, a free brake a policy learns to abuse.
class CartDoublePendulumEnv:
    def __init__(self, n=None, dt=0.01, max_steps=1000, x_max=2.4, max_force=30.0,
                 fall_angle=np.pi / 2, perturb=0.15, start_angle=np.pi, damping=15.0, wall_crash=False,
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
        self.p = Params(g=g, M=M, m1=m1, m2=m2, l1=l1, l2=l2)
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
        return self.state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        F = np.where(action == 0, -self.damping * self.state[1],
                     np.clip(action, -self.max_force, self.max_force))
        self.state = physics_step(self.state, F, self.dt, self.p)
        self.steps += 1

        hit_wall = np.abs(self.state[0]) > self.x_max
        self.state[0] = np.where(hit_wall, np.sign(self.state[0]) * self.x_max, self.state[0])
        self.state[1] = np.where(hit_wall, 0.0, self.state[1])

        x, v, the1, z1, the2, z2 = self.state
        reward = -np.cos(the1) - np.cos(the2) - 0.1 * (x / self.x_max) ** 2

        dev1 = (the1 % (2 * np.pi)) - np.pi
        dev2 = (the2 % (2 * np.pi)) - np.pi
        fell = (np.abs(dev1) > self.fall_angle) | (np.abs(dev2) > self.fall_angle)
        timeout = self.steps >= self.max_steps
        crashed = hit_wall & self.wall_crash
        reward = reward - 50.0 * (fell | crashed)
        done = fell | crashed | timeout

        if self.n is None:
            reward, fell, hit_wall, crashed, timeout, done = (float(reward), bool(fell), bool(hit_wall),
                                                              bool(crashed), bool(timeout), bool(done))
        info = {'fell': fell, 'hit_wall': hit_wall, 'crashed': crashed, 'timeout': timeout}
        return self.state.copy(), reward, done, info
