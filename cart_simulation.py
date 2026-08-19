from collections import namedtuple

import numpy as np
import sympy as smp

# Part 1 - Set up my equations

t, g = smp.symbols('t g')
M, m1, m2 = smp.symbols('M m1 m2')
l1, l2 = smp.symbols('L1 L2')
F = smp.symbols('F')

# x, theta1, theta2 are all functions of time now
x, the1, the2 = smp.symbols('x, \\theta_1, \\theta_2', cls=smp.Function)
x = x(t)
the1 = the1(t)
the2 = the2(t)

x_d = smp.diff(x, t)
x_dd = smp.diff(x_d, t)
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

# Cartesian coordinates of the two bobs, now offset by the cart position x
x1 = x + l1 * smp.sin(the1)
y1 = -l1 * smp.cos(the1)
x2 = x + l1 * smp.sin(the1) + l2 * smp.sin(the2)
y2 = -l1 * smp.cos(the1) - l2 * smp.cos(the2)

# Kinetic energy: cart + bob1 + bob2
Tc = smp.Rational(1, 2) * M * x_d ** 2
T1 = smp.Rational(1, 2) * m1 * (smp.diff(x1, t) ** 2 + smp.diff(y1, t) ** 2)
T2 = smp.Rational(1, 2) * m2 * (smp.diff(x2, t) ** 2 + smp.diff(y2, t) ** 2)
T = Tc + T1 + T2

# Potential energy
V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2

L = T - V

# Euler-Lagrange equations:
# le = dL/dq - d/dt(dL/dq_dot)
lex = smp.diff(L, x) - smp.diff(smp.diff(L, x_d), t).simplify()
le1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
le2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

sols = smp.solve([le1, le2, lex + F], (the1_dd, the2_dd, x_dd), simplify=False, rational=False)

# Sanity check: x itself should not appear in any solved acceleration - only x_dot should, since no potential energy depends on cart position
for name, expr in [('the1_dd', sols[the1_dd]), ('the2_dd', sols[the2_dd]), ('x_dd', sols[x_dd])]:
    assert x not in expr.free_symbols, f'{name} unexpectedly depends on x itself: {expr}'
print('Sanity check passed: none of the solved accelerations depend on cart position x.')

# Lambdifyyy
_args = (t, g, M, m1, m2, l1, l2, the1, the2, the1_d, the2_d, x_d, F)
dz1dt_f = smp.lambdify(_args, sols[the1_dd])
dz2dt_f = smp.lambdify(_args, sols[the2_dd])
dvdt_f = smp.lambdify(_args, sols[x_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)
dxdt_f = smp.lambdify(x_d, x_d)

# Create my vector S
def dSdt(S, t, g, M, m1, m2, l1, l2, F_val):
    x_, v, the1_, z1, the2_, z2 = S
    return [
        dxdt_f(v),
        dvdt_f(t, g, M, m1, m2, l1, l2, the1_, the2_, z1, z2, v, F_val),
        dthe1dt_f(z1),
        dz1dt_f(t, g, M, m1, m2, l1, l2, the1_, the2_, z1, z2, v, F_val),
        dthe2dt_f(z2),
        dz2dt_f(t, g, M, m1, m2, l1, l2, the1_, the2_, z1, z2, v, F_val),
    ]

def total_energy(S, g, M, m1, m2, l1, l2):
    x_, v, the1_, z1, the2_, z2 = S.T
    y1_ = -l1 * np.cos(the1_)
    y2_ = -l1 * np.cos(the1_) - l2 * np.cos(the2_)
    x1_ = x_ + l1 * np.sin(the1_)
    x2_ = x_ + l1 * np.sin(the1_) + l2 * np.sin(the2_)
    x1d = v + l1 * np.cos(the1_) * z1
    y1d = l1 * np.sin(the1_) * z1
    x2d = v + l1 * np.cos(the1_) * z1 + l2 * np.cos(the2_) * z2
    y2d = l1 * np.sin(the1_) * z1 + l2 * np.sin(the2_) * z2
    ke = 0.5 * M * v ** 2 + 0.5 * m1 * (x1d ** 2 + y1d ** 2) + 0.5 * m2 * (x2d ** 2 + y2d ** 2)
    pe = m1 * g * y1_ + m2 * g * y2_
    return ke + pe


# Part 2: step function
Params = namedtuple('Params', ['g', 'M', 'm1', 'm2', 'l1', 'l2'])

def derivs(S, F_val, p):
    return np.array(dSdt(S, 0.0, p.g, p.M, p.m1, p.m2, p.l1, p.l2, F_val))

def step(S, F_val, dt, p):
    S = np.asarray(S, dtype=float)
    k1 = derivs(S, F_val, p)
    k2 = derivs(S + dt / 2 * k1, F_val, p)
    k3 = derivs(S + dt / 2 * k2, F_val, p)
    k4 = derivs(S + dt * k3, F_val, p)
    return S + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
