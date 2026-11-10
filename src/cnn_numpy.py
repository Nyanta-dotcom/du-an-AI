"""Small CNN written in pure NumPy (week 4).

Used because PyTorch could not be installed in the environment where the
week-4 experiments were run. Same idea as src/train_cnn.py (PyTorch):

  input 28x28x1
  conv 3x3, 16 filters, pad 1 + ReLU -> max-pool 2x2   (14x14x16)
  conv 3x3, 32 filters, pad 1 + ReLU -> max-pool 2x2   (7x7x32)
  flatten 1568 -> dense 128 + ReLU -> dropout -> dense 24 -> softmax

Layout is NHWC. Convolution = im2col + matrix multiply.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_forward(X, W, b):
    """X (N,H,W,C), W (C*9, F), b (F,) -> Y (N,H,W,F), cache"""
    N, H, Wd, C = X.shape
    Xp = np.pad(X, ((0, 0), (1, 1), (1, 1), (0, 0)))
    win = sliding_window_view(Xp, (3, 3), axis=(1, 2))          # (N,H,W,C,3,3)
    cols = win.reshape(N * H * Wd, C * 9)                          # copies
    Y = (cols @ W + b).reshape(N, H, Wd, -1)
    return Y, (X.shape, cols)


def conv_backward(dY, W, cache):
    (N, H, Wd, C), cols = cache
    dY2 = dY.reshape(N * H * Wd, -1)
    dW = cols.T @ dY2
    db = dY2.sum(0)
    dcols = (dY2 @ W.T).reshape(N, H, Wd, C, 3, 3)
    dXp = np.zeros((N, H + 2, Wd + 2, C), dtype=dY.dtype)
    for i in range(3):
        for j in range(3):
            dXp[:, i:i + H, j:j + Wd, :] += dcols[..., i, j]
    return dXp[:, 1:-1, 1:-1, :], dW, db


def pool_forward(X):
    N, H, W, C = X.shape
    R = X.reshape(N, H // 2, 2, W // 2, 2, C)
    Y = R.max(axis=(2, 4))
    mask = (R == Y[:, :, None, :, None, :])
    mask = mask / mask.sum(axis=(2, 4), keepdims=True)           # split ties
    return Y, (X.shape, mask)


def pool_backward(dY, cache):
    shape, mask = cache
    return (mask * dY[:, :, None, :, None, :]).reshape(shape)


class CNN:
    def __init__(self, n_classes=24, f1=16, f2=32, hidden=128, dropout=0.3, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        he = lambda fan_in, shape: (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(dtype)
        self.p = {
            "W1": he(9, (9, f1)), "b1": np.zeros(f1, dtype),
            "W2": he(9 * f1, (9 * f1, f2)), "b2": np.zeros(f2, dtype),
            "W3": he(7 * 7 * f2, (7 * 7 * f2, hidden)), "b3": np.zeros(hidden, dtype),
            "W4": he(hidden, (hidden, n_classes)), "b4": np.zeros(n_classes, dtype),
        }
        self.dropout = dropout
        self.rng = rng
        self.dtype = dtype

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def forward(self, X, train=False):
        p = self.p
        X = X.reshape(len(X), 28, 28, 1).astype(self.dtype)
        z1, c1 = conv_forward(X, p["W1"], p["b1"]); a1 = np.maximum(z1, 0)
        q1, k1 = pool_forward(a1)
        z2, c2 = conv_forward(q1, p["W2"], p["b2"]); a2 = np.maximum(z2, 0)
        q2, k2 = pool_forward(a2)
        f = q2.reshape(len(X), -1)
        z3 = f @ p["W3"] + p["b3"]; a3 = np.maximum(z3, 0)
        if train and self.dropout > 0:
            m = (self.rng.random(a3.shape) >= self.dropout).astype(self.dtype) / (1 - self.dropout)
            a3 = a3 * m
        else:
            m = None
        logits = a3 @ p["W4"] + p["b4"]
        self._cache = (c1, z1, k1, c2, z2, k2, q2.shape, f, z3, a3, m)
        return logits

    def backward(self, dlogits):
        p = self.p
        c1, z1, k1, c2, z2, k2, q2s, f, z3, a3, m = self._cache
        g = {}
        g["W4"] = a3.T @ dlogits; g["b4"] = dlogits.sum(0)
        da3 = dlogits @ p["W4"].T
        if m is not None:
            da3 = da3 * m
        dz3 = da3 * (z3 > 0)
        g["W3"] = f.T @ dz3; g["b3"] = dz3.sum(0)
        dq2 = (dz3 @ p["W3"].T).reshape(q2s)
        da2 = pool_backward(dq2, k2); dz2 = da2 * (z2 > 0)
        dq1, g["W2"], g["b2"] = conv_backward(dz2, p["W2"], c2)
        da1 = pool_backward(dq1, k1); dz1 = da1 * (z1 > 0)
        _, g["W1"], g["b1"] = conv_backward(dz1, p["W1"], c1)
        return g

    def predict_proba(self, X, batch=1024):
        out = []
        for s in range(0, len(X), batch):
            z = self.forward(X[s:s + batch])
            z = z - z.max(1, keepdims=True); e = np.exp(z)
            out.append(e / e.sum(1, keepdims=True))
        return np.concatenate(out)

    def state(self):
        return {k: v.copy() for k, v in self.p.items()}

    def load(self, st):
        self.p = {k: v.copy() for k, v in st.items()}


def softmax_ce(logits, y, smoothing=0.0):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z); P = e / e.sum(1, keepdims=True)
    n, k = P.shape
    T = np.full_like(P, smoothing / k); T[np.arange(n), y] += 1 - smoothing
    loss = -(T * np.log(P + 1e-12)).sum(1).mean()
    return loss, (P - T) / n


class Adam:
    def __init__(self, params, lr=1e-3, wd=1e-4, b1=0.9, b2=0.999):
        self.lr, self.wd, self.b1, self.b2, self.t = lr, wd, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k] + (self.wd * params[k] if k[0] == "W" else 0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(params[k].dtype)
