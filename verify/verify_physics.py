import os
import sys

import numpy as np
from scipy.integrate import odeint

# This script lives in verify/, but physics.py and visualize.py live one
# level up in the project root -- Python only looks in this file's own
# directory by default, so we have to add the root to sys.path ourselves.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
OUTPUT_DIR = os.path.join(_ROOT, 'outputs')

from physics import dSdt, total_energy, Params, step

# Phase 1 check: with F=0, total energy should stay ~constant.
# Phase 2 check: the RK4 step() loop should reproduce the odeint trajectory at short horizons, and any long-horizon divergence should be explained by the double pendulum's chaos, not a bug in step().

g_val = 9.80665
M_val = 1.0
m1_val, m2_val = 1.0, 1.0
l1_val, l2_val = 1.5, 1.0
F_val = 0.0

t_arr = np.linspace(0, 20, 2001)
S0 = [0, 0, 1, -3, -1, 5]
ans = odeint(dSdt, y0=S0, t=t_arr, args=(g_val, M_val, m1_val, m2_val, l1_val, l2_val, F_val))

E = total_energy(ans, g_val, M_val, m1_val, m2_val, l1_val, l2_val)
print(f'Energy at t=0:  {E[0]:.6f}')
print(f'Energy at t=20: {E[-1]:.6f}')
print(f'Max abs drift:  {np.max(np.abs(E - E[0])):.6e}')

# Loop the RK4 step() function over the same dt/duration as the odeint
p = Params(g=g_val, M=M_val, m1=m1_val, m2=m2_val, l1=l1_val, l2=l2_val)
dt = t_arr[1] - t_arr[0]
n_steps = len(t_arr) - 1

S = np.array(S0, dtype=float)
ans_rk4 = np.zeros((len(t_arr), 6))
ans_rk4[0] = S
for i in range(n_steps):
    S = step(S, F_val, dt, p)
    ans_rk4[i + 1] = S

diff = np.abs(ans_rk4 - ans)
print(f'\nPhase 2 check: RK4 step() loop vs. single-call odeint (dt={dt:.4f}, {n_steps} steps)')
for idx in (50, 100, 200, 500, 1000, 2000):  # t = 0.5s, 1s, 2s, 5s, 10s, 20s
    print(f'  t={t_arr[idx]:5.1f}s  max abs diff so far: {np.max(diff[:idx + 1]):.6e}')

E_rk4 = total_energy(ans_rk4, g_val, M_val, m1_val, m2_val, l1_val, l2_val)
print(f'RK4 energy drift over run: {np.max(np.abs(E_rk4 - E_rk4[0])):.6e}')

# The double pendulum is inherently chaotic: nudge the odeint initial condition by a tiny epsilon and see if it diverges from the original odeint run by a similar order of magnitude

eps = 1e-8
S0_nudged = np.array(S0, dtype=float)
S0_nudged[2] += eps  # nudge theta1 by a tiny amount
ans_nudged = odeint(dSdt, y0=S0_nudged, t=t_arr,
                     args=(g_val, M_val, m1_val, m2_val, l1_val, l2_val, F_val))
chaos_diff = np.abs(ans_nudged - ans)
print(f'\nChaos check: odeint with theta1 nudged by {eps:.0e} vs. original odeint')
print(f'Max abs difference by t=20s: {np.max(chaos_diff):.6e}')
print('(similar order of magnitude to the RK4-vs-odeint diff above => '
      'divergence is chaos, not a step() bug)')

from visualize import animate_trajectory

animate_trajectory(
    ans, dt, l1_val, l2_val,
    series={'total energy': E},
    save_path=os.path.join(OUTPUT_DIR, 'free_swing.gif'),
)
