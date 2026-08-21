"""
Adds client-level differential privacy (DP-FedAvg, McMahan et al., 2018)
and a quantization simulation to what a client uploads.

For DP: the local update gets its L2 norm clipped to `clip_norm`, then
Gaussian noise is added, scaled by noise_multiplier. The personal head
is never uploaded at all, so none of this applies to it.

approx_epsilon() only estimates the privacy cost of a single round. A
real multi-round privacy budget needs a proper accountant, like the one
in Opacus, which isn't implemented here.
"""
import math
import numpy as np


def clip_and_add_noise(delta, clip_norm, noise_multiplier, seed=0):
    """delta -- list of numpy arrays (local update = trained - received).
    Returns the clipped, noised delta."""
    total_sq = sum(float(np.sum(d.astype(np.float64) ** 2)) for d in delta)
    total_norm = math.sqrt(total_sq)
    scale = min(1.0, clip_norm / (total_norm + 1e-12))
    clipped = [d * scale for d in delta]
    if noise_multiplier <= 0:
        return clipped
    rng = np.random.default_rng(seed)
    noised = [d + rng.normal(0, noise_multiplier * clip_norm, size=d.shape) for d in clipped]
    return noised


def approx_epsilon(noise_multiplier, delta=1e-5):
    """Single-round epsilon of the Gaussian mechanism (not a cumulative
    budget, see the module docstring): epsilon = sqrt(2*ln(1.25/delta)) / noise_multiplier."""
    if noise_multiplier <= 0:
        return float("inf")
    return math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier


def quantize_dequantize(params, bits):
    """Simulates quantization: rounds each tensor to 2**bits levels within
    its own [min, max] range, then converts straight back to float. This
    measures the accuracy impact of lower-precision transmission without
    writing an actual bit-packing/serialization protocol."""
    if bits is None or bits >= 32:
        return [p.copy() for p in params]
    levels = 2 ** bits - 1
    out = []
    for p in params:
        pmin, pmax = float(p.min()), float(p.max())
        if pmax - pmin < 1e-12:
            out.append(p.copy())
            continue
        q = np.round((p - pmin) / (pmax - pmin) * levels)
        dq = (q / levels) * (pmax - pmin) + pmin
        out.append(dq.astype(p.dtype))
    return out


def compression_ratio(bits):
    """How many times fewer bits per parameter vs float32."""
    if bits is None or bits >= 32:
        return 1.0
    return 32.0 / bits
