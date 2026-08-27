import os
import sys

import numpy as np

# See verify_physics.py for why this path fix is needed -- this script
# moved into verify/, but cart_env.py and visualize.py live in the root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
OUTPUT_DIR = os.path.join(_ROOT, 'outputs')

from cart_env import CartDoublePendulumEnv
from visualize import animate_trajectory


def rollout(env, action_fn, seed=None):
    rng = np.random.default_rng(seed)
    state = env.reset(rng=rng)
    states = [state]
    forces = []
    for _ in range(env.max_steps):
        F = action_fn(state)
        state, reward, done, info = env.step(F)
        states.append(state)
        forces.append(F)
        if done:
            break
    return np.array(states), np.array(forces), info


if __name__ == '__main__':
    env = CartDoublePendulumEnv()

    # Check 1: no force at all -- upright is unstable, so this should end
    # in a fall, at roughly the same timescale Phase 3 saw (~1.7s).
    states_zero, forces_zero, info_zero = rollout(env, lambda s: 0.0, seed=0)
    print(f'F=0 rollout: ended after {len(forces_zero)} steps '
          f'({len(forces_zero) * env.dt:.2f}s), info={info_zero}')

    # Check 2: constant max force -- should push the cart off the track
    # quickly, before the pendulum has much time to fall.
    states_push, forces_push, info_push = rollout(env, lambda s: env.max_force, seed=0)
    print(f'F=max rollout: ended after {len(forces_push)} steps '
          f'({len(forces_push) * env.dt:.2f}s), info={info_push}')

    animate_trajectory(
        states_push, env.dt, env.p.l1, env.p.l2,
        series={'force (N)': forces_push},
        x_max=env.x_max,
        fixed_camera=True,
        save_path=os.path.join(OUTPUT_DIR, 'cart_env_push.gif'),
    )

    # Check 3: the Phase 3 P-controller, now bounded by the track limit --
    # a longer, more watchable run than either check above.
    k = 50.0
    p_control = lambda s: -k * (s[2] - np.pi)
    states_p, forces_p, info_p = rollout(env, p_control, seed=0)
    print(f'P-controlled rollout: ended after {len(forces_p)} steps '
          f'({len(forces_p) * env.dt:.2f}s), info={info_p}')

    animate_trajectory(
        states_p, env.dt, env.p.l1, env.p.l2,
        series={
            'force (N)': forces_p,
            'theta1 dev (deg)': np.degrees(states_p[:, 2] - np.pi),
            'theta2 dev (deg)': np.degrees(states_p[:, 4] - np.pi),
        },
        x_max=env.x_max,
        fixed_camera=True,
        save_path=os.path.join(OUTPUT_DIR, 'cart_env_pcontrol.gif'),
    )
