"""
The model is written in plain NumPy -- forward and backward passes are
coded by hand instead of using an autograd library like PyTorch. That's
what lets it plug directly into flwr.client.NumPyClient.

There are two interchangeable "bases": SharedBase, an MLP over
statistical features, and the CNNBase classes, a 1D CNN over the raw
signal. Both expose the same interface, so PersonalHead and FedPerModel
can work with either one without any special-casing.
"""
import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cross_entropy(probs, y, sample_weight=None):
    n = len(y)
    p = np.clip(probs[np.arange(n), y], 1e-12, 1.0)
    nll = -np.log(p)
    if sample_weight is None:
        return np.mean(nll)
    w = sample_weight
    return float(np.sum(nll * w) / np.sum(w))


# Base #1: MLP over statistical features

class SharedBase:
    """One hidden layer: n_features -> hidden, ReLU. This is what gets
    federated (averaged) in fedavg / fedper / fedper_adaptive."""

    def __init__(self, n_features=30, hidden=32, seed=0, momentum=0.0):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.momentum = momentum
        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._cache = None

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        self._cache = (X, z1)
        return a1

    def backward(self, d_a1, lr):
        X, z1 = self._cache
        n = X.shape[0]
        dz1 = d_a1 * (z1 > 0)
        dW1 = X.T @ dz1 / n
        db1 = dz1.mean(axis=0)
        # momentum=0 -> plain SGD (W -= lr*grad)
        self._vW1 = self.momentum * self._vW1 - lr * dW1
        self._vb1 = self.momentum * self._vb1 - lr * db1
        self.W1 += self._vW1
        self.b1 += self._vb1

    def get_parameters(self):
        return [self.W1.copy(), self.b1.copy()]

    def set_parameters(self, params):
        self.W1, self.b1 = [p.copy() for p in params]


# Base #2: real 1D CNN over the raw signal

class Conv1DLayer:
    """1D convolution: (N, C_in, T) -> (N, C_out, L), via im2col
    (sliding_window_view) for speed."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, seed=0, momentum=0.0):
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(out_channels, in_channels, kernel_size))
        self.b = np.zeros(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.momentum = momentum
        self._vW = np.zeros_like(self.W)
        self._vb = np.zeros_like(self.b)
        self._cache = None

    def forward(self, X):
        n, c_in, t = X.shape
        k, s = self.kernel_size, self.stride
        windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=k, axis=2)  # (N,C,T-k+1,k)
        windows = windows[:, :, ::s, :]  # (N, C_in, L, k)
        L = windows.shape[2]
        cols = windows.transpose(0, 2, 1, 3).reshape(n, L, c_in * k)  # (N, L, C_in*k)
        w_flat = self.W.reshape(self.out_channels, c_in * k)  # (C_out, C_in*k)
        out = cols @ w_flat.T + self.b  # (N, L, C_out)
        self._cache = (cols, X.shape)
        return out.transpose(0, 2, 1)  # (N, C_out, L)

    def backward(self, dout, lr):
        cols, x_shape = self._cache
        n, c_in, t = x_shape
        k, s = self.kernel_size, self.stride
        L = dout.shape[2]
        dout_t = dout.transpose(0, 2, 1)  # (N, L, C_out)

        w_flat = self.W.reshape(self.out_channels, c_in * k)
        dW_flat = np.einsum("nlk,nlc->ck", cols, dout_t) / n  # (C_out, C_in*k)
        db = dout_t.sum(axis=(0, 1)) / n

        dcols = dout_t @ w_flat  # (N, L, C_in*k)
        dcols = dcols.reshape(n, L, c_in, k)

        dX = np.zeros(x_shape, dtype=cols.dtype)
        for l in range(L):
            start = l * s
            dX[:, :, start:start + k] += dcols[:, l, :, :]

        self._vW = self.momentum * self._vW - lr * dW_flat.reshape(self.W.shape)
        self._vb = self.momentum * self._vb - lr * db
        self.W += self._vW
        self.b += self._vb
        return dX

    def get_parameters(self):
        return [self.W.copy(), self.b.copy()]

    def set_parameters(self, params):
        self.W, self.b = [p.copy() for p in params]


class CNNBaseDeep:
    """Two conv layers + ReLU + global average pool over time.
    Input: (N, 5, T'). Output (embedding): (N, hidden_dim).
    Federates BOTH conv layers (arch='cnn_deep')."""

    def __init__(self, in_channels=5, seed=0, momentum=0.0):
        self.conv1 = Conv1DLayer(in_channels, 8, kernel_size=15, stride=5, seed=seed, momentum=momentum)
        self.conv2 = Conv1DLayer(8, 16, kernel_size=9, stride=3, seed=seed + 1, momentum=momentum)
        self.hidden_dim = 16
        self._cache = None

    def forward(self, X):
        z1 = self.conv1.forward(X)
        a1 = np.maximum(0, z1)
        z2 = self.conv2.forward(a1)
        a2 = np.maximum(0, z2)
        emb = a2.mean(axis=2)  # global average pool -> (N, 16)
        self._cache = (z1, a1, z2, a2)
        return emb

    def backward(self, d_emb, lr):
        z1, a1, z2, a2 = self._cache
        n, c2, L2 = a2.shape
        d_a2 = np.repeat(d_emb[:, :, None], L2, axis=2) / L2  # avg-pool gradient
        d_z2 = d_a2 * (z2 > 0)
        d_a1 = self.conv2.backward(d_z2, lr)
        d_z1 = d_a1 * (z1 > 0)
        self.conv1.backward(d_z1, lr)

    def get_parameters(self):
        return self.conv1.get_parameters() + self.conv2.get_parameters()

    def set_parameters(self, params):
        self.conv1.set_parameters(params[0:2])
        self.conv2.set_parameters(params[2:4])


class CNNBaseShallow:
    """Only the first conv layer here is federated. The second layer,
    which picks up more frequency-specific detail, lives in
    PersonalHeadCNN and stays local to each client instead.

    The reason: averaging both layers (like CNNBaseDeep does) washes out
    the client-specific frequency filters that second layer learns. This
    is what arch='cnn' uses. The output here isn't pooled, since the
    second conv layer needs the full time sequence to work with."""

    def __init__(self, in_channels=5, seed=0, momentum=0.0):
        self.conv1 = Conv1DLayer(in_channels, 8, kernel_size=15, stride=5, seed=seed, momentum=momentum)
        self.hidden_dim = 8
        self._cache = None

    def forward(self, X):
        z1 = self.conv1.forward(X)
        a1 = np.maximum(0, z1)
        self._cache = z1
        return a1

    def backward(self, d_a1, lr):
        z1 = self._cache
        d_z1 = d_a1 * (z1 > 0)
        self.conv1.backward(d_z1, lr)

    def get_parameters(self):
        return self.conv1.get_parameters()

    def set_parameters(self, params):
        self.conv1.set_parameters(params)


# Personal (local, never federated) head

class PersonalHead:
    """hidden -> mid -> 2 classes, ReLU + softmax."""

    def __init__(self, hidden=32, mid=16, n_classes=2, seed=1, momentum=0.0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, mid))
        self.b1 = np.zeros(mid)
        self.W2 = rng.normal(0, np.sqrt(2.0 / mid), size=(mid, n_classes))
        self.b2 = np.zeros(n_classes)
        self.momentum = momentum
        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)
        self._cache = None

    def forward(self, a0):
        z1 = a0 @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        probs = _softmax(z2)
        self._cache = (a0, z1, a1)
        return probs

    def backward(self, probs, y, lr, sample_weight=None):
        a0, z1, a1 = self._cache
        n = len(y)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1
        if sample_weight is None:
            dz2 = (probs - onehot) / n  # (N, n_classes)
        else:
            # class-balanced weighted cross-entropy: bigger gradient for
            # the rare class in this client's batch, normalized by the
            # sum of weights (unbiased gradient estimate)
            w = sample_weight[:, None]
            dz2 = (probs - onehot) * w / np.sum(sample_weight)

        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = a0.T @ dz1
        db1 = dz1.sum(axis=0)
        d_a0 = dz1 @ self.W1.T

        self._vW2 = self.momentum * self._vW2 - lr * dW2
        self._vb2 = self.momentum * self._vb2 - lr * db2
        self._vW1 = self.momentum * self._vW1 - lr * dW1
        self._vb1 = self.momentum * self._vb1 - lr * db1
        self.W2 += self._vW2
        self.b2 += self._vb2
        self.W1 += self._vW1
        self.b1 += self._vb1
        return d_a0

    def get_parameters(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_parameters(self, params):
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]

    def state_dict(self):
        return self.get_parameters()

    def load_state_dict(self, state):
        self.set_parameters(state)


class PersonalHeadCNN:
    """The personal head that goes with CNNBaseShallow. It holds the
    second conv layer (never federated), plus a global average pool and
    an FC layer on top. It exposes the same methods as PersonalHead
    (forward/backward/get_parameters/state_dict), so FedPerModel doesn't
    need any special-casing to work with CNN instead of MLP."""

    def __init__(self, in_channels=8, mid_channels=16, n_classes=2, seed=1, momentum=0.0):
        self.conv2 = Conv1DLayer(in_channels, mid_channels, kernel_size=9, stride=3, seed=seed, momentum=momentum)
        rng = np.random.default_rng(seed + 100)
        self.W2 = rng.normal(0, np.sqrt(2.0 / mid_channels), size=(mid_channels, n_classes))
        self.b2 = np.zeros(n_classes)
        self.momentum = momentum
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)
        self._cache = None

    def forward(self, a1):
        z2 = self.conv2.forward(a1)  # (N, 16, L2)
        a2 = np.maximum(0, z2)
        pooled = a2.mean(axis=2)  # GAP -> (N, 16)
        logits = pooled @ self.W2 + self.b2
        probs = _softmax(logits)
        self._cache = (z2, a2, pooled)
        return probs

    def backward(self, probs, y, lr, sample_weight=None):
        z2, a2, pooled = self._cache
        n, c2, L2 = a2.shape
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1
        if sample_weight is None:
            dlogits = (probs - onehot) / n
        else:
            w = sample_weight[:, None]
            dlogits = (probs - onehot) * w / np.sum(sample_weight)

        dW2 = pooled.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dpooled = dlogits @ self.W2.T  # (N, 16)

        d_a2 = np.repeat(dpooled[:, :, None], L2, axis=2) / L2
        d_z2 = d_a2 * (z2 > 0)
        d_a1 = self.conv2.backward(d_z2, lr)  # gradient back into CNNBaseShallow

        self._vW2 = self.momentum * self._vW2 - lr * dW2
        self._vb2 = self.momentum * self._vb2 - lr * db2
        self.W2 += self._vW2
        self.b2 += self._vb2
        return d_a1

    def get_parameters(self):
        return self.conv2.get_parameters() + [self.W2.copy(), self.b2.copy()]

    def set_parameters(self, params):
        self.conv2.set_parameters(params[0:2])
        self.W2, self.b2 = [p.copy() for p in params[2:]]

    def state_dict(self):
        return self.get_parameters()

    def load_state_dict(self, state):
        self.set_parameters(state)


# Full model: federated base + local personal head

class FedPerModel:
    def __init__(self, arch="mlp", n_features=30, seed=0, momentum=0.0, n_channels=5):
        self.arch = arch
        if arch == "cnn":
            # only the first conv layer is federated; the second lives
            # in the personal head (see CNNBaseShallow).
            self.base = CNNBaseShallow(in_channels=n_channels, seed=seed, momentum=momentum)
            self.personal = PersonalHeadCNN(in_channels=8, mid_channels=16, n_classes=2, seed=seed + 1, momentum=momentum)
        elif arch == "cnn_deep":
            # older arch: both conv layers federated, personal part is
            # just the FC head. Kept for ablation/comparison.
            self.base = CNNBaseDeep(in_channels=n_channels, seed=seed, momentum=momentum)
            self.personal = PersonalHead(hidden=self.base.hidden_dim, mid=16, n_classes=2, seed=seed + 1, momentum=momentum)
        else:
            self.base = SharedBase(n_features=n_features, hidden=32, seed=seed, momentum=momentum)
            self.personal = PersonalHead(hidden=self.base.hidden_dim, mid=16, n_classes=2, seed=seed + 1, momentum=momentum)
        self._n_base_params = len(self.base.get_parameters())

    def forward(self, X):
        emb = self.base.forward(X)
        return self.personal.forward(emb)

    def train_step(self, X, y, lr, update_base=True, sample_weight=None):
        emb = self.base.forward(X)
        probs = self.personal.forward(emb)
        d_emb = self.personal.backward(probs, y, lr, sample_weight=sample_weight)
        if update_base:
            self.base.backward(d_emb, lr)

    def predict(self, X):
        probs = self.forward(X)
        return probs.argmax(axis=1), probs

    def loss(self, X, y, sample_weight=None):
        probs = self.forward(X)
        return _cross_entropy(probs, y, sample_weight=sample_weight)

    def get_base_parameters(self):
        return self.base.get_parameters()

    def set_base_parameters(self, params):
        self.base.set_parameters(params)

    def get_all_parameters(self):
        return self.base.get_parameters() + self.personal.get_parameters()

    def set_all_parameters(self, params):
        self.base.set_parameters(params[: self._n_base_params])
        self.personal.set_parameters(params[self._n_base_params:])


def build_model(arch="mlp", n_features=30, seed=0, momentum=0.0, n_channels=5):
    return FedPerModel(arch=arch, n_features=n_features, seed=seed, momentum=momentum, n_channels=n_channels)
