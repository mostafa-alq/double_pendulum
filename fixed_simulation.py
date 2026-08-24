import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

# Lagrangian mechanics approach
# I am using sympy for deriving equations of motion symbolically, odeint for numerical integration and matplotlib for rendering

t, g = smp.symbols('t g')
m1, m2 = smp.symbols ('m1 m2')
l1, l2 = smp.symbols('L1, L2')

# Theta 1 and 2 are functions of time which we will eventually need to solve for
the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)

the1 = the1(t)
the2 = the2(t)

# Define derivatives and second derivatives
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

# Define cartestian coordinates of bobs using trigonometry
x1 = l1 * smp.sin(the1)
y1 = -l1 * smp.cos(the1)
x2 = (l1 * smp.sin(the1)) + (l2 * smp.sin(the2))
y2 = (-l1 * smp.cos(the1) - (l2 * smp.cos(the2)))

# Kinetic energy calculations
t1 = 1/2 * m1 * (smp.diff(x1, t) ** 2 + smp.diff(y1, t) ** 2)
t2 = 1/2 * m2 * (smp.diff(x2, t) ** 2 + smp.diff(y2, t) ** 2)
T = t1 + t2

# Potential energy calculations
v1 = m1 * g * y1
v2 = m2 * g * y2
v = v1 + v2

# Lagrangian (Literally just kinetic minus potential energy)
l = T - v

# Get Lagrange's equations and solve (assuming le1 and le2 is 0)
le1 = smp.diff(l, the1) - smp.diff(smp.diff(l, the1_d), t).simplify()
le2 = smp.diff(l, the2) - smp.diff(smp.diff(l, the2_d), t).simplify()

sols = smp.solve([le1, le2], (the1_dd, the2_dd), simplify = False, rational = False)

# Convert two second order ODEs into four first order ODE. I will define z1 = dtheta1/dt and z2 = dtheta2/dt. Therefore, dz1/dt = d^2theta1/dt^2 and dz2/dt = d^2theta2/dt^2
dz1dt_f = smp.lambdify((t,g,m1,m2,l1,l2,the1,the2,the1_d,the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify((t,g,m1,m2,l1,l2,the1,the2,the1_d,the2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

# Define my vector S which will contain theta1, z1, theta2 and z2. This function takes in my vector S and t and returns the derivative of S in respect to time.
def dSdt(S, t, g, m1, m2, l1, l2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(t, g, m1, m2, l1, l2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, g, m1, m2, l1, l2, the1, the2, z1, z2),
    ]

# Values for simulation
t = np.linspace(0, 40, 1001)
g = 9.80665
m1 = 1
m2 = 1
l1 = 1.5
l2 = 1
ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t, args=(g, m1, m2, l1, l2))
the1 = ans.T[0]
the2 = ans.T[2]

# Function to return location of the two masses based on theta1 and theta2
def get_x1y1x2y2(t, the1, the2, l1, l2):
    return (l1 * np.sin(the1),
            -l1*np.cos(the1),
            l1*np.sin(the1) + l2*np.sin(the2),
            -l1*np.cos(the1) - l2*np.cos(the2))

x1, y1, x2, y2 = get_x1y1x2y2(t, ans.T[0], ans.T[2], l1, l2)

def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ln1, = plt.plot([], [], 'ro--', lw = 3, markersize = 8)
ax.set_ylim(-4,4)
ax.set_xlim(-4,4)
ani = animation.FuncAnimation(fig, animate, frames = 1000, interval = 50)
ani.save('outputs/simulation.gif', writer = 'pillow', fps = 25)