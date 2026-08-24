import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_trajectory(states, dt, l1, l2, series=None, x_max=None,
                        fps=25, save_path='animation.gif', title=None):
    n = len(states)
    stride = max(1, round(1 / (dt * fps)))  # ~real-time playback
    frame_idx = np.arange(0, n, stride)
    t_arr = np.arange(n) * dt

    x = states[:, 0]
    the1 = states[:, 2]
    the2 = states[:, 4]

    x1 = x + l1 * np.sin(the1)
    y1 = -l1 * np.cos(the1)
    x2 = x1 + l2 * np.sin(the2)
    y2 = y1 - l2 * np.cos(the2)

    n_series = len(series) if series else 0
    fig = plt.figure(figsize=(8, 8 + 2 * n_series))
    gs = fig.add_gridspec(1 + n_series, 1, height_ratios=[4] + [1] * n_series, hspace=0.4)

    ax_main = fig.add_subplot(gs[0])
    ax_main.set_facecolor('k')
    ax_main.set_aspect('equal')
    ax_main.get_xaxis().set_ticks([])
    ax_main.get_yaxis().set_ticks([])
    if title:
        ax_main.set_title(title)

    reach = l1 + l2
    ax_main.set_xlim(x[0] - reach - 1, x[0] + reach + 1)  # camera follows the cart -- see update()
    ax_main.set_ylim(-reach - 0.5, reach + 0.5)

    ax_main.axhline(0, color='w', lw=1)  # rail
    if x_max is not None:
        ax_main.axvline(-x_max, color='r', lw=1, ls='--')
        ax_main.axvline(x_max, color='r', lw=1, ls='--')

    cart_w, cart_h = 0.3, 0.2
    cart_patch = plt.Rectangle((x[0] - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                                fc='dimgray', ec='w')
    ax_main.add_patch(cart_patch)
    link_line, = ax_main.plot([], [], 'ro-', lw=3, markersize=8)

    cursor_lines = []
    if series:
        axes_series = []
        for i, (label, arr) in enumerate(series.items()):
            ax_s = fig.add_subplot(gs[1 + i])
            t_s = np.arange(len(arr)) * dt
            ax_s.plot(t_s, arr)
            ax_s.set_ylabel(label)
            ax_s.set_xlim(t_arr[0], t_arr[-1])
            cursor_lines.append(ax_s.axvline(0, color='r', lw=1))
            axes_series.append(ax_s)
        axes_series[-1].set_xlabel('time (s)')

    def update(i):
        ax_main.set_xlim(x[i] - reach - 1, x[i] + reach + 1)  # camera follows the cart
        cart_patch.set_xy((x[i] - cart_w / 2, -cart_h / 2))
        link_line.set_data([x[i], x1[i], x2[i]], [0, y1[i], y2[i]])
        for cursor in cursor_lines:
            cursor.set_xdata([t_arr[i], t_arr[i]])
        return [cart_patch, link_line] + cursor_lines

    ani = animation.FuncAnimation(fig, update, frames=frame_idx)
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f'Saved animation to {save_path} ({len(frame_idx)} frames, ~{n * dt:.1f}s real-time)')
