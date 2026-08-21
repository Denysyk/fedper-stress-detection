"""
Checks how sensitive AdaptiveFedPer is to three hyperparameters:
local_epochs, lr, and client_fraction. For each value tried, it trains
both fedper (no adaptation) and fedper_adaptive and compares them.
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


def run_pair(clients_data, arch, n_rounds, seed, **client_kwargs):
    out = {}
    for mode in ("fedper", "fedper_adaptive"):
        history = run_federation(clients_data, mode, arch=arch, n_rounds=n_rounds, seed=seed, **client_kwargs)
        by_round = {}
        for h in history:
            by_round.setdefault(h["round"], []).append(h["accuracy"])
        out[mode] = {r: float(np.mean(v)) for r, v in by_round.items()}
    return out


def rounds_to_threshold(round_means, threshold=0.95, n_rounds=15):
    for r in range(1, n_rounds + 1):
        if round_means.get(r, 0) >= threshold:
            return r
    return n_rounds + 1  # not reached


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--n-clients", type=int, default=15)
    p.add_argument("--n-windows", type=int, default=140)
    p.add_argument("--n-rounds", type=int, default=15)
    p.add_argument("--arch", choices=["mlp", "cnn", "cnn_deep"], default="mlp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reversal-prob", type=float, default=0.25)
    args = p.parse_args()

    if args.data_dir:
        clients_data = load_wesad_all(args.data_dir)
    else:
        clients_data = generate_synthetic_federation(
            n_clients=args.n_clients, n_windows=args.n_windows,
            reversal_prob=args.reversal_prob, seed=args.seed,
        )

    os.makedirs("results", exist_ok=True)
    rows = []

    grids = {
        "local_epochs": [1, 2, 3, 5],
        "lr": [0.01, 0.02, 0.03, 0.05],
        "client_fraction": [1.0, 0.7, 0.5, 0.3],
    }
    base_kwargs = dict(local_epochs=1, lr=0.05, client_fraction=1.0,
                        adaptive_max_epochs=8, improvement_threshold=0.002, mu=0.1)

    for dim, values in grids.items():
        for v in values:
            kwargs = dict(base_kwargs)
            kwargs[dim] = v
            out = run_pair(clients_data, args.arch, args.n_rounds, args.seed, **kwargs)
            r1_gap = out["fedper_adaptive"].get(1, 0) - out["fedper"].get(1, 0)
            r_fedper = rounds_to_threshold(out["fedper"], n_rounds=args.n_rounds)
            r_adaptive = rounds_to_threshold(out["fedper_adaptive"], n_rounds=args.n_rounds)
            speedup = r_fedper / r_adaptive if r_adaptive else float("nan")
            rows.append({
                "dimension": dim, "value": v,
                "round1_gap": r1_gap, "rounds_fedper": r_fedper,
                "rounds_adaptive": r_adaptive, "speedup": speedup,
            })
            print(f"[{dim}={v}] round1_delta={r1_gap:+.3f}  fedper->{r_fedper}  adaptive->{r_adaptive}  speedup={speedup:.2f}x")

    with open(os.path.join("results", "ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "value", "round1_gap", "rounds_fedper", "rounds_adaptive", "speedup"])
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (dim, values) in zip(axes, grids.items()):
        sub = [r for r in rows if r["dimension"] == dim]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, [r["rounds_fedper"] for r in sub], width=0.4, label="FedPer")
        ax.bar(x + 0.2, [r["rounds_adaptive"] for r in sub], width=0.4, label="AdaptiveFedPer")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["value"]) for r in sub])
        ax.set_title(dim)
        ax.set_ylabel("rounds to acc>=0.95")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "plot_ablation.png"), dpi=150)
    plt.close()
    print("\nSaved results/ablation.csv, results/plot_ablation.png")


if __name__ == "__main__":
    main()
