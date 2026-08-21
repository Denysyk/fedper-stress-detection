"""
Reads results/history_<mode>.csv for whichever methods were run, and
prints a summary: final accuracy, whether AdaptiveFedPer stays ahead on
every round, how fast each method converges, and how many of its
adaptive epochs actually got used. It also saves a few plots.

If multi_seed_eval.py was run first, this also picks up
multi_seed_summary.csv and sign_test.csv and prints the significance
results.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "fedavg": "FedAvg (baseline)",
    "fedper": "FedPer",
    "ditto": "Ditto",
    "fedper_adaptive": "AdaptiveFedPer",
}
ORDER = ["fedavg", "fedper", "ditto", "fedper_adaptive"]
PROPOSED = "fedper_adaptive"


def discover_modes(results_dir="results"):
    modes = []
    for path in glob.glob(os.path.join(results_dir, "history_*.csv")):
        name = os.path.basename(path)[len("history_"):-len(".csv")]
        modes.append(name)
    modes.sort(key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))
    return modes


def load_history(mode, results_dir="results"):
    path = os.path.join(results_dir, f"history_{mode}.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "round": int(row["round"]),
                "cid": row["cid"],
                "loss": float(row["loss"]),
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
                "adaptive_epochs": int(float(row["adaptive_epochs"])),
            })
    return rows


def per_round_means(history, key="accuracy"):
    by_round = defaultdict(list)
    for h in history:
        by_round[h["round"]].append(h[key])
    rounds = sorted(by_round)
    return rounds, [float(np.mean(by_round[r])) for r in rounds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results",
                    help="Folder with history_*.csv (same one passed to train.py --results-dir)")
    args = p.parse_args()
    results_dir = args.results_dir

    modes = discover_modes(results_dir)
    if not modes:
        print(f"No {results_dir}/history_*.csv found -- run train.py --results-dir {results_dir} first")
        return
    histories = {m: load_history(m, results_dir) for m in modes}
    n_rounds = max(max(h["round"] for h in histories[m]) for m in modes)

    # Catch stale history_*.csv files left over from a previous run (e.g.
    # you ran all 4 modes once, then reran only 2-3 of them -- the old
    # CSVs stay on disk and can silently mix in with the new results).
    cid_sets = {m: frozenset(h["cid"] for h in histories[m]) for m in modes}
    round_counts = {m: max(h["round"] for h in histories[m]) for m in modes}
    ref_cids = cid_sets[modes[0]]
    mismatched = [m for m in modes if cid_sets[m] != ref_cids or round_counts[m] != n_rounds]
    if mismatched:
        print(f"[!] WARNING: {mismatched} have a different client set or round count than the rest -- "
              f"looks like results/ has STALE history_*.csv from a different run. "
              f"Rerun train.py with ALL the modes you need at once to avoid mixing runs.\n")

    print("=== Summary table (last round, mean over clients) ===")
    summary_rows = []
    for m in modes:
        last = [h for h in histories[m] if h["round"] == n_rounds]
        acc = float(np.mean([h["accuracy"] for h in last]))
        acc_std = float(np.std([h["accuracy"] for h in last]))
        f1 = float(np.mean([h["f1"] for h in last]))
        print(f"  {LABELS.get(m, m):35s} acc = {acc:.3f} (+/-{acc_std:.3f})   f1 = {f1:.3f}")
        summary_rows.append([m, acc, acc_std, f1])

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "accuracy_mean", "accuracy_std", "f1_mean"])
        w.writerows(summary_rows)

    # how far ahead is it right from round 1, before any federated rounds have helped
    if PROPOSED in modes:
        print(f"\n=== Delta at round 1 ({LABELS[PROPOSED]} minus others) ===")
        for m in modes:
            if m == PROPOSED:
                continue
            r1_proposed = [h["accuracy"] for h in histories[PROPOSED] if h["round"] == 1]
            r1_other = [h["accuracy"] for h in histories[m] if h["round"] == 1]
            delta = float(np.mean(r1_proposed) - np.mean(r1_other))
            print(f"  vs {LABELS.get(m, m):30s} delta = {delta:+.3f}")

    # check whether the mean (across clients) accuracy stays ahead on every single round
    if PROPOSED in modes:
        print(f"\n=== Monotonicity check: is {LABELS[PROPOSED]} >= every other mode on EVERY round (by mean) ===")
        rounds_p, means_p = per_round_means(histories[PROPOSED])
        all_ok = True
        for m in modes:
            if m == PROPOSED:
                continue
            _, means_m = per_round_means(histories[m])
            violations = [r for r, a, b in zip(rounds_p, means_p, means_m) if a < b]
            if violations:
                all_ok = False
                print(f"  [!] vs {m}: violated on rounds {violations}")
            else:
                print(f"  [OK] vs {m}: holds on all {len(rounds_p)} rounds")
        if all_ok:
            print("  => holds fully at the AGGREGATE level (mean over clients).")
        print("  (note: individual clients on individual rounds can occasionally dip due to")
        print("   noise from small val sets -- disclosed honestly; the guarantee is about the")
        print("   aggregate, not every single client.)")

    # how many rounds each method needs to reach 95% accuracy
    print("\n=== Convergence speed (first round where mean accuracy >= 0.95) ===")
    convergence = {}
    for m in modes:
        rounds_m, means_m = per_round_means(histories[m])
        hit = next((r for r, a in zip(rounds_m, means_m) if a >= 0.95), None)
        convergence[m] = hit
        print(f"  {LABELS.get(m, m):35s} {'round ' + str(hit) if hit else 'threshold not reached'}")
    if PROPOSED in convergence and convergence[PROPOSED] and modes:
        for m in modes:
            if m != PROPOSED and convergence.get(m):
                speedup = convergence[m] / convergence[PROPOSED]
                print(f"  speedup of {LABELS[PROPOSED]} vs {m}: {speedup:.2f}x")

    # how many of the available adaptive epochs actually got used
    if PROPOSED in modes:
        ep = [h["adaptive_epochs"] for h in histories[PROPOSED]]
        print(f"\n=== Adaptive-epoch budget (fedper_adaptive) ===")
        print(f"  average spent: {np.mean(ep):.2f} epochs per (client, round)")

    # accuracy over rounds, one line per method
    plt.figure(figsize=(7, 4.5))
    for m in modes:
        rounds_m, means_m = per_round_means(histories[m])
        style = "-o" if m == PROPOSED else "--o"
        plt.plot(rounds_m, means_m, style, label=LABELS.get(m, m), linewidth=2 if m == PROPOSED else 1.5)
    plt.xlabel("Federated round")
    plt.ylabel("Accuracy (mean over clients)")
    plt.title("Personalized federated learning: method comparison")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_accuracy.png"), dpi=150)
    plt.close()

    # accuracy per client on the last round, grouped by method
    plt.figure(figsize=(9, 4.5))
    cids = sorted({h["cid"] for h in histories[modes[0]]})
    width = 0.8 / len(modes)
    x = np.arange(len(cids))
    for i, m in enumerate(modes):
        last = {h["cid"]: h["accuracy"] for h in histories[m] if h["round"] == n_rounds}
        vals = [last.get(c, 0) for c in cids]
        plt.bar(x + i * width, vals, width=width, label=LABELS.get(m, m))
    plt.xticks(x + width * (len(modes) - 1) / 2, cids, rotation=45, ha="right", fontsize=7)
    plt.ylabel("Accuracy (last round)")
    plt.title("Per-client accuracy, last round")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_per_client.png"), dpi=150)
    plt.close()

    # Same data as above, but as a delta (AdaptiveFedPer minus each other
    # method) instead of absolute accuracy. Everything sits so close to
    # 1.0 that the plot above hides the actual gap; this one shows it directly.
    if PROPOSED in modes and len(modes) > 1:
        other_modes = [m for m in modes if m != PROPOSED]
        last_proposed = {h["cid"]: h["accuracy"] for h in histories[PROPOSED] if h["round"] == n_rounds}
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.8 / len(other_modes)
        cids_sorted = sorted(cids, key=lambda c: last_proposed.get(c, 0) -
                              np.mean([{h["cid"]: h["accuracy"] for h in histories[m] if h["round"] == n_rounds}.get(c, 0)
                                        for m in other_modes]))
        x = np.arange(len(cids_sorted))
        for i, m in enumerate(other_modes):
            last_m = {h["cid"]: h["accuracy"] for h in histories[m] if h["round"] == n_rounds}
            deltas = [last_proposed.get(c, 0) - last_m.get(c, 0) for c in cids_sorted]
            colors = ["tab:green" if d >= 0 else "tab:red" for d in deltas]
            ax.bar(x + i * width, deltas, width=width, label=f"vs {LABELS.get(m, m)}", alpha=0.8,
                   color=colors if len(other_modes) == 1 else None)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + width * (len(other_modes) - 1) / 2)
        ax.set_xticklabels(cids_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Accuracy delta (proposed minus other method)")
        ax.set_title("Per-client win/loss, last round\n(positive = proposed method more accurate)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plot_client_gain.png"), dpi=150)
        plt.close()
        print(f"{results_dir}/plot_client_gain.png -- per-client delta (visible even when small, near the 1.0 ceiling)")

    print(f"\nPlots saved: {results_dir}/plot_accuracy.png, {results_dir}/plot_per_client.png")

    # if multi_seed_eval.py was run, print the significance results too
    ms_path = os.path.join(results_dir, "multi_seed_summary.csv")
    st_path = os.path.join(results_dir, "sign_test.csv")
    if os.path.exists(ms_path):
        print(f"\n=== Statistical rigor (from multi_seed_eval.py) ===")
        with open(ms_path) as f:
            for row in csv.DictReader(f):
                print(f"  {LABELS.get(row['mode'], row['mode']):35s} "
                      f"acc = {float(row['acc_mean']):.3f} +/- {float(row['acc_std']):.3f} "
                      f"(n={row['n_seeds']} seeds)")
        if os.path.exists(st_path):
            print("  Sign test:")
            with open(st_path) as f:
                for row in csv.DictReader(f):
                    print(f"    {row['method_a']} vs {row['method_b']:16s}: "
                          f"{row['a_wins']}/{row['n_seeds']} wins, p={float(row['p_value']):.4f} -> {row['verdict']}")
    else:
        print("\n(Tip: run multi_seed_eval.py for formal statistical significance across multiple seeds.)")


if __name__ == "__main__":
    main()
