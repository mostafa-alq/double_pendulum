import numpy as np


class Linear:
    """y = x @ W + b, with a hand-written backward pass."""

    def __init__(self, in_dim, out_dim):
        # Same small-random init idea PyTorch uses -- if weights all started
        # at exactly the same value, every hidden unit would learn the same
        # thing (no symmetry breaking).
        limit = np.sqrt(1 / in_dim)
        self.W = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self._x = None  # cached input, needed by backward()

    def forward(self, x):
        self._x = x  # backward() needs this later
        return x @ self.W + self.b

    def backward(self, grad_output):
        # grad_output: gradient of the loss w.r.t. this layer's OUTPUT,
        # shape (batch, out_dim). We need three things from it:
        grad_W = self._x.T @ grad_output          # d(loss)/d(W)
        grad_b = grad_output.sum(axis=0)           # d(loss)/d(b)
        grad_x = grad_output @ self.W.T             # d(loss)/d(x), passed upstream
        return grad_x, grad_W, grad_b
