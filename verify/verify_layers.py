import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from ppo.layers import Linear

np.random.seed(0)


def numerical_gradient(f, param, eps=1e-5):
    """Finite-difference approximation of d(f())/d(param), elementwise.
    f re-reads `param` by reference each call, so nudging param in place
    and calling f() again tells us how much the output moved."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        f_plus = f()

        param[idx] = orig - eps
        f_minus = f()

        param[idx] = orig  # restore
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


if __name__ == '__main__':
    layer = Linear(in_dim=4, out_dim=3)
    x = np.random.randn(5, 4)  # batch of 5 examples, 4 features each

    # A stand-in "loss": sum of the layer's output. Simple enough that its
    # gradient w.r.t. the output is just all-ones -- easy to check by hand.
    def loss():
        return layer.forward(x).sum()

    loss()  # populate layer._x for backward()
    grad_output = np.ones((5, 3))  # d(loss)/d(output), since loss = sum(output)
    grad_x_analytical, grad_W_analytical, grad_b_analytical = layer.backward(grad_output)

    grad_W_numerical = numerical_gradient(loss, layer.W)
    grad_b_numerical = numerical_gradient(loss, layer.b)
    grad_x_numerical = numerical_gradient(lambda: layer.forward(x).sum(), x)

    for name, analytical, numerical in [
        ('grad_W', grad_W_analytical, grad_W_numerical),
        ('grad_b', grad_b_analytical, grad_b_numerical),
        ('grad_x', grad_x_analytical, grad_x_numerical),
    ]:
        max_diff = np.max(np.abs(analytical - numerical))
        print(f'{name}: max abs difference between analytical and numerical = {max_diff:.2e}')
