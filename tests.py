"""
Sanity tests, no pytest needed. Run with:

    python3 tests.py
"""
import numpy as np

from model import Conv1DLayer, build_model
from multi_seed_eval import sign_test_pvalue
from data import FeatureWindowDataset, RawWindowDataset, train_val_split, N_CHANNELS, N_FEATURES, WINDOW_LEN
from privacy import clip_and_add_noise, quantize_dequantize
from client import _stable_seed


def _numerical_grad(f, x, idx, eps=1e-4):
    orig = x[idx]
    x[idx] = orig + eps
    f_plus = f()
    x[idx] = orig - eps
    f_minus = f()
    x[idx] = orig
    return (f_plus - f_minus) / (2 * eps)


def test_conv1d_input_gradient():
    rng = np.random.default_rng(0)
    layer = Conv1DLayer(in_channels=2, out_channels=3, kernel_size=3, stride=1, seed=0)
    X = rng.normal(size=(2, 2, 10))

    def loss_fn():
        out = layer.forward(X)
        return 0.5 * float(np.sum(out ** 2))

    out = layer.forward(X)
    dout = out  # d(0.5*sum(out^2))/d(out) = out
    dX = layer.backward(dout, lr=0.0)  # lr=0 -> weights untouched, safe to reuse layer

    for idx in [(0, 0, 0), (1, 1, 5), (0, 1, 9)]:
        analytic = dX[idx]
        numeric = _numerical_grad(loss_fn, X, idx)
        assert abs(analytic - numeric) < 1e-3, f"input grad mismatch at {idx}: {analytic} vs {numeric}"
    print("[OK] test_conv1d_input_gradient")


def test_conv1d_weight_gradient():
    # Conv1DLayer's contract: dout is treated as the gradient of a SUM-over-
    # batch loss (matches how dX is returned, unscaled, to keep propagating
    # upstream); but the WEIGHT gradient it applies is the usual mean-over-
    # batch SGD gradient, i.e. dW = (dL_sum/dW) / n. So the numeric check
    # here divides by n to match that.
    rng = np.random.default_rng(1)
    n = 2
    X = rng.normal(size=(n, 2, 10))
    lr = 1e-3

    layer = Conv1DLayer(in_channels=2, out_channels=3, kernel_size=3, stride=1, seed=0)
    W_before = layer.W.copy()
    out = layer.forward(X)
    dout = out
    layer.backward(dout, lr=lr)  # updates layer.W in place (momentum=0 -> plain SGD)
    recovered_dW = (W_before - layer.W) / lr  # W_new = W_old - lr*dW

    probe = Conv1DLayer(in_channels=2, out_channels=3, kernel_size=3, stride=1, seed=0)
    probe.W = W_before.copy()

    def loss_fn():
        out = probe.forward(X)
        return 0.5 * float(np.sum(out ** 2))

    for idx in [(0, 0, 0), (2, 1, 2)]:
        analytic = recovered_dW[idx]
        numeric = _numerical_grad(loss_fn, probe.W, idx) / n
        assert abs(analytic - numeric) < 1e-3, f"weight grad mismatch at {idx}: {analytic} vs {numeric}"
    print("[OK] test_conv1d_weight_gradient")


def test_train_step_reduces_loss():
    """Not a gradient check, just a smoke test: a few SGD steps on random
    data should not make the loss worse on average."""
    rng = np.random.default_rng(2)
    n, n_features = 40, 10
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, 2, size=n)

    model = build_model(arch="mlp", n_features=n_features, seed=0)
    loss_before = model.loss(X, y)
    for _ in range(50):
        model.train_step(X, y, lr=0.1, update_base=True)
    loss_after = model.loss(X, y)
    assert loss_after < loss_before, f"loss did not decrease: {loss_before} -> {loss_after}"
    print("[OK] test_train_step_reduces_loss")


def test_sign_test_10_of_10():
    # matches the real 10/10, p=0.0020 results reported in README.md
    p, n_pos, n_neg = sign_test_pvalue([1.0] * 10)
    assert n_pos == 10 and n_neg == 0
    assert abs(p - 0.001953125) < 1e-9, p
    print("[OK] test_sign_test_10_of_10")


def test_sign_test_9_of_10():
    # matches the real CNN-vs-FedPer result on WESAD: 9/10, p=0.0215
    p, n_pos, n_neg = sign_test_pvalue([1.0] * 9 + [-1.0])
    assert n_pos == 9 and n_neg == 1
    assert abs(p - 0.021484375) < 1e-9, p
    print("[OK] test_sign_test_9_of_10")


def test_sign_test_even_split_not_significant():
    p, n_pos, n_neg = sign_test_pvalue([1.0] * 5 + [-1.0] * 5)
    assert n_pos == 5 and n_neg == 5
    assert p == 1.0
    print("[OK] test_sign_test_even_split_not_significant")


def test_sign_test_all_ties():
    p, n_pos, n_neg = sign_test_pvalue([0.0] * 10)
    assert n_pos == 0 and n_neg == 0 and p == 1.0
    print("[OK] test_sign_test_all_ties")


def test_dataset_shapes():
    rng = np.random.default_rng(3)
    windows = [rng.normal(size=(N_CHANNELS, WINDOW_LEN)) for _ in range(20)]
    labels = np.array([i % 2 for i in range(20)])

    feat_ds = FeatureWindowDataset(windows, labels)
    assert feat_ds.X.shape == (20, N_FEATURES), feat_ds.X.shape
    assert abs(feat_ds.X.mean()) < 0.5 and abs(feat_ds.X.std() - 1.0) < 0.5, "features don't look standardized"

    raw_ds = RawWindowDataset(windows, labels)
    assert raw_ds.X.shape[0] == 20 and raw_ds.X.shape[1] == N_CHANNELS, raw_ds.X.shape
    print("[OK] test_dataset_shapes")


def test_train_val_split_no_leakage():
    rng = np.random.default_rng(4)
    windows = [rng.normal(size=(2, 5)) for _ in range(40)]
    labels = np.array([i % 2 for i in range(40)])
    (tr_w, tr_y), (va_w, va_y) = train_val_split(windows, labels, val_frac=0.25, seed=1)
    assert len(tr_w) + len(va_w) == 40
    assert len(va_w) == 10
    assert {id(w) for w in tr_w}.isdisjoint({id(w) for w in va_w}), "same window in both train and val"
    print("[OK] test_train_val_split_no_leakage")


def test_dp_clipping_bounds_norm():
    rng = np.random.default_rng(5)
    delta = [rng.normal(0, 10, size=(4, 4)) for _ in range(3)]  # well above clip_norm
    clip_norm = 1.0
    clipped = clip_and_add_noise(delta, clip_norm, noise_multiplier=0.0, seed=0)
    total_norm = np.sqrt(sum(np.sum(d.astype(np.float64) ** 2) for d in clipped))
    assert total_norm <= clip_norm + 1e-6, total_norm
    print("[OK] test_dp_clipping_bounds_norm")


def test_quantize_dequantize():
    rng = np.random.default_rng(6)
    params = [rng.normal(size=(5, 5)) for _ in range(2)]

    unquantized = quantize_dequantize(params, None)
    for a, b in zip(params, unquantized):
        assert np.allclose(a, b), "bits=None should be a no-op copy"

    err_8 = sum(np.sum((a - b) ** 2) for a, b in zip(params, quantize_dequantize(params, 8)))
    err_2 = sum(np.sum((a - b) ** 2) for a, b in zip(params, quantize_dequantize(params, 2)))
    assert err_8 < err_2, "more bits should mean less reconstruction error"
    print("[OK] test_quantize_dequantize")


def test_stable_seed_determinism():
    a = _stable_seed("client_0", run_seed=1)
    b = _stable_seed("client_0", run_seed=1)
    c = _stable_seed("client_0", run_seed=2)
    assert a == b, "same (cid, run_seed) should give the same seed every time"
    assert a != c, "different run_seed should give a different seed"
    print("[OK] test_stable_seed_determinism")


if __name__ == "__main__":
    test_conv1d_input_gradient()
    test_conv1d_weight_gradient()
    test_train_step_reduces_loss()
    test_sign_test_10_of_10()
    test_sign_test_9_of_10()
    test_sign_test_even_split_not_significant()
    test_sign_test_all_ties()
    test_dataset_shapes()
    test_train_val_split_no_leakage()
    test_dp_clipping_bounds_norm()
    test_quantize_dequantize()
    test_stable_seed_determinism()
    print("\nAll tests passed.")
