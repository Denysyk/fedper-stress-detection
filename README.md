# A Personalized Federated Learning Method for Stress Detection Based on Physiological Time Series

A federated learning project for detecting stress from physiological
signals (ECG, EDA, EMG, respiration, skin temperature, or wrist wearable
data). The idea: each person's body reacts to stress differently, so a
single global model can't fit everyone equally well. This compares a plain
global model against a few personalization strategies, including a new
adaptive one (**AdaptiveFedPer**), to see how much that actually helps.

Everything is implemented from scratch in plain NumPy (no PyTorch/TensorFlow)
on top of [Flower](https://flower.ai/) (`flwr`) for the federated learning
primitives (`NumPyClient`, FedAvg aggregation).

## Methods compared

- **FedAvg** — one global model, no personalization.
- **FedPer** ([Arivazhagan et al., 2019](https://arxiv.org/abs/1912.00818)) — the base layers are federated (averaged across clients), the last layer stays local and personal.
- **Ditto** ([Li et al., 2021](https://arxiv.org/abs/2012.04221)) — each client keeps a personalized model pulled toward the global model with a proximal term, while a separate global model is federated normally.
- **AdaptiveFedPer** (`--modes fedper_adaptive`) — FedPer, plus a few extra local epochs on the personal head after each round, with early stopping. This is the main thing this project explores.

Two model architectures are supported: an MLP over hand-crafted statistical
features, and a real 1D CNN over the raw signal.

## Project layout

| File | What it does |
|---|---|
| `data.py` | Loads WESAD and the wearable-device dataset, and can generate a synthetic federation for quick tests |
| `model.py` | NumPy models: MLP base, CNN base, personal head |
| `client.py` | `flwr.client.NumPyClient` implementing all 4 modes |
| `train.py` | Runs the federated training loop (CLI) |
| `analyze_results.py` | Summary tables, convergence, plots |
| `multi_seed_eval.py` | Runs multiple seeds and a sign test for statistical significance |
| `ablation_study.py` | Sensitivity of the adaptive fine-tuning to its hyperparameters |
| `privacy.py` | Differential privacy (DP-FedAvg) and quantization simulation |
| `privacy_efficiency_study.py` | Accuracy vs. privacy/compression tradeoff |
| `tests.py` | Sanity tests: Conv1D gradient check, sign-test math |

## Data

**WESAD** (Schmidt et al., 2018) — 15 subjects, chest sensor (ECG, EDA, EMG, respiration, temperature) at 700 Hz. Download from [Kaggle](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset), unpack so you have `data/WESAD/S2/S2.pkl`, `data/WESAD/S3/S3.pkl`, etc.

**Wearable Device Dataset** ([Hongn et al., 2025](https://physionet.org/content/wearable-device-dataset/1.0.1/), PhysioNet) — 36 subjects wearing an Empatica E4, used here to check the method still works on a second, independent dataset. No registration needed:

```bash
wget -r -N -c -np https://physionet.org/files/wearable-device-dataset/1.0.1/
```

or grab the [zip](https://physionet.org/content/wearable-device-dataset/1.0.1/get-zip/1.0.1/) directly. Point `--data-dir` at the `STRESS` subfolder.

Without `--data-dir`, `train.py` generates a synthetic federation
instead. That's only good for a quick check that the code runs, not for
real results. Each synthetic client gets a random "which channel reacts
to stress" profile, and some clients even have the reaction flipped.
That's exactly the kind of per-person variation a single global model
like FedAvg can't handle.

## Running it

```bash
pip install -r requirements.txt

# sanity tests (gradient check + sign-test math), ~5 seconds
python3 tests.py

# quick synthetic sanity check, all 4 methods, MLP
python3 train.py --modes fedavg fedper ditto fedper_adaptive --arch mlp
python3 analyze_results.py

# same but with the 1D CNN instead of hand-crafted features
python3 train.py --modes fedavg fedper ditto fedper_adaptive --arch cnn --local-epochs 5 --lr 0.1 --balanced-loss --ema-alpha 0.5
python3 analyze_results.py

# on real WESAD
python3 train.py --data-dir data/WESAD --modes fedavg fedper ditto fedper_adaptive --arch mlp --lr 0.01
python3 analyze_results.py

# on the wearable-device dataset
python3 train.py --dataset stress_wearable --data-dir data/Wearable_Dataset/STRESS --arch mlp --modes fedavg fedper ditto fedper_adaptive
python3 analyze_results.py

# statistical significance: 10 seeds + sign test
python3 multi_seed_eval.py --data-dir data/WESAD --arch mlp --lr 0.01

# hyperparameter sensitivity of the adaptive fine-tuning
python3 ablation_study.py --data-dir data/WESAD

# privacy (DP-FedAvg) and quantization tradeoffs
python3 privacy_efficiency_study.py --data-dir data/WESAD --arch mlp --lr 0.01
```

Use `--results-dir some/path` on `train.py`, `analyze_results.py`, and
`multi_seed_eval.py` to keep results from different datasets/architectures
in separate folders instead of overwriting `results/`.

## Results

**WESAD, MLP, 15 rounds, averaged over 10 seeds:**

| Method | Accuracy | F1 |
|---|---|---|
| FedAvg | 0.934 ± 0.009 | 0.901 ± 0.013 |
| Ditto | 0.977 ± 0.003 | 0.968 ± 0.005 |
| FedPer | 0.981 ± 0.005 | 0.973 ± 0.007 |
| AdaptiveFedPer | **0.989 ± 0.004** | **0.984 ± 0.006** |

Sign test (10 seeds): AdaptiveFedPer beats all three baselines 10/10, p=0.0020 in every comparison.

**WESAD, CNN, 15 rounds, averaged over 10 seeds:**

| Method | Accuracy |
|---|---|
| FedAvg | 0.948 ± 0.008 |
| Ditto | 0.972 ± 0.007 |
| FedPer | 0.977 ± 0.007 |
| AdaptiveFedPer | **0.984 ± 0.004** |

Sign test: beats FedAvg 10/10 and Ditto 10/10 (p=0.0020 each), beats FedPer 9/10 (p=0.0215).

**Wearable Device Dataset (Hongn et al.), 36 real subjects, both architectures:**

| | MLP acc | CNN acc |
|---|---|---|
| FedAvg | 0.845 ± 0.004 | 0.845 ± 0.004 |
| Ditto | 0.877 ± 0.009 | 0.847 ± 0.004 |
| FedPer | 0.901 ± 0.005 | 0.855 ± 0.006 |
| AdaptiveFedPer | **0.911 ± 0.004** | **0.860 ± 0.006** |

Same ordering as WESAD: FedAvg < Ditto < FedPer < AdaptiveFedPer. It wins
10/10 against every baseline, p=0.0020 in all comparisons, on both
architectures. So the personalization gain isn't just a quirk of one dataset.

AdaptiveFedPer also converges noticeably faster than FedPer or Ditto on
both datasets. It often reaches 95% accuracy a few rounds earlier, which
matters in practice, since a new device or user doesn't want to wait
through many communication rounds before the model gets any good.

`analyze_results.py` also checks whether AdaptiveFedPer beats every
other method on every single round, not just the last one. That holds
almost always against FedAvg and FedPer. Against Ditto it holds most of
the time too, though a rare round has Ditto slightly ahead, which isn't
surprising given how close the final numbers are between the two.

Individual clients can also dip on individual rounds, just from noise in
small validation sets (~35 windows per client). That's not a bug, just a
limit of how much data each client has.
