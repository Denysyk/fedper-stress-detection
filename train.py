"""
Runs the federated training rounds. Instead of flwr.simulation (which
needs the ray package, not installed here), this is a plain loop written
around Flower's own building blocks: NumPyClient and the aggregate
function. That means switching to flwr.simulation or `flwr run` later
shouldn't require changing client.py or model.py.
"""
import argparse
import csv
import os
import numpy as np
from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate

from client import StressFedClient
from data import generate_synthetic_federation, load_wesad_all, load_stress_wearable_all

MODES = ["fedavg", "fedper", "fedper_adaptive", "ditto"]


def make_clients(clients_data, mode, arch, run_seed=0, **client_kwargs):
    return {
        cid: StressFedClient(cid, windows, labels, mode=mode, arch=arch, run_seed=run_seed, **client_kwargs)
        for cid, (windows, labels) in clients_data.items()
    }


def run_federation(clients_data, mode, arch="mlp", n_rounds=15, seed=0, client_fraction=1.0, **client_kwargs):
    # seed feeds both the clients (run_seed -> weight init, train/val split,
    # batch shuffle order) and per-round client selection. That's what makes
    # different --seed values on the SAME real dataset genuinely different
    # trials for multi_seed_eval.py.
    clients = make_clients(clients_data, mode, arch, run_seed=seed, **client_kwargs)
    client_ids = list(clients.keys())
    rng = np.random.default_rng(seed)

    global_params = clients[client_ids[0]].get_parameters()
    history = []  # list of dict: round, cid, loss, accuracy, f1, adaptive_epochs, selected

    for rnd in range(1, n_rounds + 1):
        n_select = max(1, int(round(len(client_ids) * client_fraction)))
        selected = set(rng.choice(client_ids, size=n_select, replace=False))

        fit_results = []
        fit_metrics_by_cid = {}
        for cid in client_ids:
            if cid not in selected:
                continue
            params, n_examples, metrics = clients[cid].fit(global_params, {})
            fit_results.append((params, n_examples))
            fit_metrics_by_cid[cid] = metrics

        global_params = fedavg_aggregate(fit_results)

        for cid in client_ids:
            loss, n_examples, metrics = clients[cid].evaluate(global_params, {})
            adaptive_epochs = fit_metrics_by_cid.get(cid, {}).get("adaptive_epochs", 0)
            history.append({
                "round": rnd,
                "cid": cid,
                "loss": loss,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "adaptive_epochs": adaptive_epochs,
                "selected": cid in selected,
            })

    return history


def save_history(history, mode, out_dir="results", suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"history_{mode}{suffix}.csv")
    fieldnames = ["round", "cid", "loss", "accuracy", "f1", "adaptive_epochs", "selected"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None, help="Folder with real data (format depends on --dataset)")
    p.add_argument("--dataset", choices=["wesad", "stress_wearable"], default="wesad",
                    help="wesad = original dataset; stress_wearable = Hongn et al. 2025 "
                         "(Empatica E4, PhysioNet) -- --data-dir should point at the STRESS "
                         "subfolder (.../Wearable_Dataset/STRESS)")
    p.add_argument("--n-clients", type=int, default=15)
    p.add_argument("--n-windows", type=int, default=140)
    p.add_argument("--n-rounds", type=int, default=15)
    p.add_argument("--modes", nargs="+", default=["fedavg", "fedper", "fedper_adaptive"])
    p.add_argument("--arch", choices=["mlp", "cnn", "cnn_deep"], default="mlp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--adaptive-max-epochs", type=int, default=8)
    p.add_argument("--improvement-threshold", type=float, default=0.002)
    p.add_argument("--client-fraction", type=float, default=1.0)
    p.add_argument("--reversal-prob", type=float, default=0.25)
    p.add_argument("--mu", type=float, default=0.1, help="Proximal-term coefficient for ditto")
    p.add_argument("--balanced-loss", action="store_true", help="Class-balanced cross-entropy for adaptive fine-tuning")
    p.add_argument("--ema-alpha", type=float, default=0.0, help="Val-loss smoothing for early-stopping (0=disabled)")
    p.add_argument("--dp-clip-norm", type=float, default=None, help="Enable DP: L2 clipping of the client update")
    p.add_argument("--dp-noise-multiplier", type=float, default=0.0, help="Gaussian noise multiplier (used with --dp-clip-norm)")
    p.add_argument("--quant-bits", type=int, default=None, help="Simulate quantizing uploaded params to N bits")
    p.add_argument("--momentum", type=float, default=0.0, help="Momentum SGD (0=disabled, plain SGD as before)")
    p.add_argument("--out-suffix", type=str, default="")
    p.add_argument("--results-dir", type=str, default="results",
                    help="Where to save history_*.csv -- useful to keep different datasets/architectures separate")
    args = p.parse_args()

    if args.data_dir:
        if args.dataset == "stress_wearable":
            clients_data = load_stress_wearable_all(args.data_dir)
            print(f"[train] loaded {len(clients_data)} real stress_wearable clients "
                  f"(Hongn et al. 2025) from {args.data_dir}")
        else:
            clients_data = load_wesad_all(args.data_dir)
            print(f"[train] loaded {len(clients_data)} real WESAD clients from {args.data_dir}")
    else:
        clients_data = generate_synthetic_federation(
            n_clients=args.n_clients, n_windows=args.n_windows,
            reversal_prob=args.reversal_prob, seed=args.seed,
        )
        print(f"[train] generated {len(clients_data)} synthetic clients")

    client_kwargs = dict(
        local_epochs=args.local_epochs,
        lr=args.lr,
        adaptive_max_epochs=args.adaptive_max_epochs,
        improvement_threshold=args.improvement_threshold,
        mu=args.mu,
        balanced_loss=args.balanced_loss,
        ema_alpha=args.ema_alpha,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        quant_bits=args.quant_bits,
        momentum=args.momentum,
    )

    for mode in args.modes:
        print(f"\n=== Mode: {mode} | arch: {args.arch} ===")
        history = run_federation(
            clients_data, mode, arch=args.arch, n_rounds=args.n_rounds, seed=args.seed,
            client_fraction=args.client_fraction, **client_kwargs,
        )
        path = save_history(history, mode, out_dir=args.results_dir, suffix=args.out_suffix)
        last_round = [h for h in history if h["round"] == args.n_rounds]
        mean_acc = np.mean([h["accuracy"] for h in last_round])
        mean_f1 = np.mean([h["f1"] for h in last_round])
        print(f"[{mode}] round {args.n_rounds}: mean acc={mean_acc:.3f}, mean f1={mean_f1:.3f} -> {path}")


if __name__ == "__main__":
    main()
