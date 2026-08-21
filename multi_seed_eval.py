"""
Runs the same experiment across several independent seeds instead of
just one, so results can be reported as a mean +/- std rather than a
single lucky (or unlucky) number.

It also runs a paired sign test to check whether AdaptiveFedPer is
actually significantly better than the baselines, not just better on
this one run by chance. The sign test is implemented from scratch with
math.comb, so it doesn't need scipy.
"""
import argparse
import csv
import math
import os

import numpy as np

from data import generate_synthetic_federation, load_wesad_all, load_stress_wearable_all
from train import run_federation

DEFAULT_MODES = ["fedavg", "fedper", "ditto", "fedper_adaptive"]


def sign_test_pvalue(diffs):
    """Two-sided sign test: diffs = a_i - b_i across seeds.
    Zeros (a_i == b_i) are excluded (standard practice)."""
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 1.0, 0, 0
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = n - n_pos
    k = max(n_pos, n_neg)
    # P(X >= k) for X~Binomial(n, 0.5), doubled for a two-sided test
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)
    return p_value, n_pos, n_neg


def final_round_means(history, n_rounds):
    last = [h for h in history if h["round"] == n_rounds]
    acc = float(np.mean([h["accuracy"] for h in last]))
    f1 = float(np.mean([h["f1"] for h in last]))
    return acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--dataset", choices=["wesad", "stress_wearable"], default="wesad")
    p.add_argument("--n-clients", type=int, default=15)
    p.add_argument("--n-windows", type=int, default=140)
    p.add_argument("--n-rounds", type=int, default=15)
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--arch", choices=["mlp", "cnn", "cnn_deep"], default="mlp")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 13, 42, 99, 123, 2024, 555, 777, 31415])
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--adaptive-max-epochs", type=int, default=8)
    p.add_argument("--improvement-threshold", type=float, default=0.002)
    p.add_argument("--client-fraction", type=float, default=1.0)
    p.add_argument("--reversal-prob", type=float, default=0.25)
    p.add_argument("--mu", type=float, default=0.1)
    p.add_argument("--balanced-loss", action="store_true")
    p.add_argument("--ema-alpha", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    client_kwargs = dict(
        local_epochs=args.local_epochs, lr=args.lr,
        balanced_loss=args.balanced_loss, ema_alpha=args.ema_alpha, momentum=args.momentum,
        adaptive_max_epochs=args.adaptive_max_epochs,
        improvement_threshold=args.improvement_threshold, mu=args.mu,
    )

    # results[mode][seed] = (acc, f1)
    results = {mode: {} for mode in args.modes}

    for seed in args.seeds:
        if args.data_dir:
            if args.dataset == "stress_wearable":
                clients_data = load_stress_wearable_all(args.data_dir)
            else:
                clients_data = load_wesad_all(args.data_dir)
        else:
            clients_data = generate_synthetic_federation(
                n_clients=args.n_clients, n_windows=args.n_windows,
                reversal_prob=args.reversal_prob, seed=seed,
            )
        for mode in args.modes:
            history = run_federation(
                clients_data, mode, arch=args.arch, n_rounds=args.n_rounds,
                seed=seed, client_fraction=args.client_fraction, **client_kwargs,
            )
            acc, f1 = final_round_means(history, args.n_rounds)
            results[mode][seed] = (acc, f1)
            print(f"[seed={seed}] {mode}: acc={acc:.3f} f1={f1:.3f}")

    os.makedirs(args.results_dir, exist_ok=True)

    # mean +/- std across seeds
    summary_path = os.path.join(args.results_dir, "multi_seed_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "n_seeds", "acc_mean", "acc_std", "f1_mean", "f1_std"])
        for mode in args.modes:
            accs = [results[mode][s][0] for s in args.seeds]
            f1s = [results[mode][s][1] for s in args.seeds]
            writer.writerow([mode, len(args.seeds), np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)])

    print(f"\n=== Descriptive stats (mean +/- std, n={len(args.seeds)} seeds) ===")
    for mode in args.modes:
        accs = [results[mode][s][0] for s in args.seeds]
        f1s = [results[mode][s][1] for s in args.seeds]
        print(f"{mode:16s}  acc = {np.mean(accs):.3f} +/- {np.std(accs):.3f}   f1 = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    # paired sign test: fedper_adaptive vs every other mode
    target = "fedper_adaptive"
    sign_rows = []
    if target in results:
        print(f"\n=== Sign test: {target} vs others (accuracy, n={len(args.seeds)} seeds) ===")
        for mode in args.modes:
            if mode == target:
                continue
            diffs = [results[target][s][0] - results[mode][s][0] for s in args.seeds]
            p_value, n_pos, n_neg = sign_test_pvalue(diffs)
            verdict = "significant (p<0.05)" if p_value < 0.05 else "NOT significant"
            print(f"  {target} vs {mode:16s}: wins {n_pos}/{len(args.seeds)}, losses {n_neg}/{len(args.seeds)}, p={p_value:.4f} -> {verdict}")
            sign_rows.append([target, mode, n_pos, n_neg, len(args.seeds), p_value, verdict])

    sign_path = os.path.join(args.results_dir, "sign_test.csv")
    with open(sign_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method_a", "method_b", "a_wins", "b_wins", "n_seeds", "p_value", "verdict"])
        writer.writerows(sign_rows)

    print(f"\nSaved: {summary_path}, {sign_path}")


if __name__ == "__main__":
    main()
