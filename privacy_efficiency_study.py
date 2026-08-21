"""
Checks how much accuracy AdaptiveFedPer loses when we add DP noise to
the base update, or quantize it before sending.

The epsilon printed here is only a single-round estimate. See privacy.py
for why that's not the same as a full multi-round privacy budget.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import generate_synthetic_federation, load_wesad_all
from train import run_federation
from privacy import approx_epsilon, compression_ratio


def final_acc(clients_data, arch, n_rounds, seed, **client_kwargs):
    history = run_federation(clients_data, "fedper_adaptive", arch=arch, n_rounds=n_rounds, seed=seed, **client_kwargs)
    last = [h["accuracy"] for h in history if h["round"] == n_rounds]
    return float(np.mean(last))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--n-clients", type=int, default=15)
    p.add_argument("--n-windows", type=int, default=140)
    p.add_argument("--n-rounds", type=int, default=15)
    p.add_argument("--arch", choices=["mlp", "cnn", "cnn_deep"], default="mlp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reversal-prob", type=float, default=0.25)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--dp-clip-norm", type=float, default=1.0)
    args = p.parse_args()

    if args.data_dir:
        clients_data = load_wesad_all(args.data_dir)
    else:
        clients_data = generate_synthetic_federation(
            n_clients=args.n_clients, n_windows=args.n_windows,
            reversal_prob=args.reversal_prob, seed=args.seed,
        )

    base_kwargs = dict(local_epochs=args.local_epochs, lr=args.lr,
                        adaptive_max_epochs=8, improvement_threshold=0.002, mu=0.1)

    os.makedirs("results", exist_ok=True)

    # accuracy vs. DP noise strength
    noise_grid = [0.0, 0.2, 0.5, 1.0, 2.0]
    dp_rows = []
    for nm in noise_grid:
        kwargs = dict(base_kwargs)
        if nm > 0:
            kwargs["dp_clip_norm"] = args.dp_clip_norm
            kwargs["dp_noise_multiplier"] = nm
        acc = final_acc(clients_data, args.arch, args.n_rounds, args.seed, **kwargs)
        eps = approx_epsilon(nm) if nm > 0 else float("inf")
        dp_rows.append({"noise_multiplier": nm, "accuracy": acc, "epsilon_1round": eps})
        eps_str = f"{eps:.2f}" if eps != float("inf") else "inf (DP off)"
        print(f"[DP] noise_multiplier={nm:<4} acc={acc:.3f}  epsilon(1 round, illustrative)={eps_str}")

    # accuracy vs. how many bits we quantize to
    bits_grid = [32, 8, 4, 2]
    quant_rows = []
    for bits in bits_grid:
        kwargs = dict(base_kwargs)
        if bits < 32:
            kwargs["quant_bits"] = bits
        acc = final_acc(clients_data, args.arch, args.n_rounds, args.seed, **kwargs)
        ratio = compression_ratio(bits if bits < 32 else None)
        quant_rows.append({"bits": bits, "accuracy": acc, "compression_ratio": ratio})
        print(f"[Quant] bits={bits:<3} acc={acc:.3f}  compression={ratio:.1f}x")

    with open(os.path.join("results", "privacy_dp.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["noise_multiplier", "accuracy", "epsilon_1round"])
        w.writeheader()
        w.writerows(dp_rows)
    with open(os.path.join("results", "privacy_quant.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bits", "accuracy", "compression_ratio"])
        w.writeheader()
        w.writerows(quant_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["noise_multiplier"] for r in dp_rows], [r["accuracy"] for r in dp_rows], "-o")
    axes[0].set_xlabel("DP noise_multiplier (0 = no DP)")
    axes[0].set_ylabel("Accuracy (AdaptiveFedPer)")
    axes[0].set_title("Privacy vs accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].plot([r["compression_ratio"] for r in quant_rows], [r["accuracy"] for r in quant_rows], "-o", color="tab:orange")
    axes[1].set_xlabel("Compression ratio (x)")
    axes[1].set_ylabel("Accuracy (AdaptiveFedPer)")
    axes[1].set_title("Communication efficiency vs accuracy")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "plot_privacy_efficiency.png"), dpi=150)
    plt.close()

    print("\nSaved: results/privacy_dp.csv, results/privacy_quant.csv, results/plot_privacy_efficiency.png")


if __name__ == "__main__":
    main()
