import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# See verify_physics.py for why this path fix is needed -- this script
# moved into verify/, but physics.py and visualize.py live in the root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
OUTPUT_DIR = os.path.join(_ROOT, 'outputs')

from physics import step, Params

# Phase 3: Run step() with a trivial controller and check the simulator behaves sanely.

g_val = 9.80665
M_val = 1.0
m1_val, m2_val = 1.0, 1.0
l1_val, l2_val = 1.5, 1.0
p = Params(g=g_val, M=M_val, m1=m1_val, m2=m2_val, l1=l1_val, l2=l2_val)

dt = 0.01
n_steps = 500  # 5 seconds

# Small perturbation from upright: 0.1 rad (~5.7 degrees) off on each link.
S0 = np.array([0, 0, np.pi + 0.1, 0, np.pi + 0.1, 0])

def run(controller, S0=S0, n_steps=n_steps, dt=dt):
    S = S0.copy()
    states = np.zeros((n_steps + 1, 6))
    forces = np.zeros(n_steps)
    states[0] = S
    for i in range(n_steps):
        F_val = controller(S)
        forces[i] = F_val
        S = step(S, F_val, dt, p)
        states[i + 1] = S
    return states, forces

def no_control(S):
    return 0.0

k = 50.0

def p_control(S):
    the1 = S[2]
    return -k * (the1 - np.pi)

if __name__ == '__main__':
    t_arr = np.arange(n_steps + 1) * dt

    states_u, forces_u = run(no_control)
    states_c, forces_c = run(p_control)

    for label, states in [('Uncontrolled', states_u), ('P-controlled', states_c)]:
        the1_deg = np.degrees(states[:, 2] - np.pi)
        x_range = states[:, 0].min(), states[:, 0].max()
        print(f'{label}: theta1 deviation from upright ranges '
              f'{the1_deg.min():.1f} to {the1_deg.max():.1f} degrees, '
              f'cart x ranges {x_range[0]:.2f} to {x_range[1]:.2f} m')

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t_arr, np.degrees(states_u[:, 2] - np.pi), label='uncontrolled')
    axes[0].plot(t_arr, np.degrees(states_c[:, 2] - np.pi), label='P-controlled')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel('theta1 deviation (deg)')
    axes[0].legend()

    axes[1].plot(t_arr, states_u[:, 0], label='uncontrolled')
    axes[1].plot(t_arr, states_c[:, 0], label='P-controlled')
    axes[1].set_ylabel('cart position x (m)')

    axes[2].plot(t_arr[:-1], forces_c, label='P-controlled force')
    axes[2].set_ylabel('force F (N)')
    axes[2].set_xlabel('time (s)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'verify_controller_force.png'), dpi=120)
    print("\nSaved plot to outputs/verify_controller_force.png")

    from visualize import animate_trajectory

    animate_trajectory(
        states_c, dt, l1_val, l2_val,
        series={'force (N)': forces_c, 'theta1 dev (deg)': np.degrees(states_c[:, 2] - np.pi)},
        x_max=2.4,
        save_path=os.path.join(OUTPUT_DIR, 'verify_controller.gif'),
    )
