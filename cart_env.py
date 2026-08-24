import numpy as np

from physics import step as physics_step, Params


class CartDoublePendulumEnv:
    """Gym-style env: force on the cart, goal is to balance near upright."""

    def __init__(self, dt=0.01, max_steps=1000, x_max=2.4, max_force=30.0,
                 fall_angle=np.pi / 2, perturb=0.15,
                 g=9.80665, M=1.0, m1=1.0, m2=1.0, l1=1.5, l2=1.0):
        self.dt = dt
        self.max_steps = max_steps
        self.x_max = x_max
        self.max_force = max_force
        self.fall_angle = fall_angle
        self.perturb = perturb
        self.p = Params(g=g, M=M, m1=m1, m2=m2, l1=l1, l2=l2)
        self.state = None
        self.steps = 0

    def reset(self, rng=None):
        rng = rng if rng is not None else np.random
        the1 = np.pi + rng.uniform(-self.perturb, self.perturb)
        the2 = np.pi + rng.uniform(-self.perturb, self.perturb)
        self.state = np.array([0.0, 0.0, the1, 0.0, the2, 0.0])
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        F = float(np.clip(action, -self.max_force, self.max_force))
        self.state = physics_step(self.state, F, self.dt, self.p)
        self.steps += 1

        x, v, the1, z1, the2, z2 = self.state
        reward = -np.cos(the1) - np.cos(the2) - 0.1 * (x / self.x_max) ** 2

        dev1 = (the1 % (2 * np.pi)) - np.pi
        dev2 = (the2 % (2 * np.pi)) - np.pi
        fell = abs(dev1) > self.fall_angle or abs(dev2) > self.fall_angle
        off_track = abs(x) > self.x_max
        timeout = self.steps >= self.max_steps

        if fell or off_track:
            reward -= 50.0

        done = fell or off_track or timeout
        info = {'fell': fell, 'off_track': off_track, 'timeout': timeout}
        return self.state.copy(), reward, done, info
