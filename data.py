"""
Loads the two real datasets (WESAD and the Hongn et al. 2025 wearable
dataset), and can also generate a synthetic federation for quick tests.

Each synthetic client gets a random profile for "which channel reacts to
stress the most," and some clients even have that reaction flipped. This
is meant to model how different people's bodies react differently to
stress -- a single global model can't handle that, but personalization can.
"""
import os
import pickle
import datetime
import numpy as np

CHEST_CHANNELS = ["ECG", "EDA", "EMG", "Resp", "Temp"]
N_CHANNELS = len(CHEST_CHANNELS)
FS_CHEST = 700
WINDOW_SEC = 10
WINDOW_LEN = WINDOW_SEC * FS_CHEST  # 7000
STEP_SEC = 5
STEP_LEN = STEP_SEC * FS_CHEST  # 3500

N_FEATURES = N_CHANNELS * 6  # for the MLP arch (statistical features)
CNN_DOWNSAMPLE = 10  # 700 Hz -> ~70 Hz for the 1D CNN

# WESAD

def load_wesad_subject(pkl_path):
    """Load one S{N}.pkl, return (windows, labels).

    windows: list of (N_CHANNELS, WINDOW_LEN) raw signal arrays.
    labels: 0 (baseline) / 1 (stress); windows with other labels
    (amusement, meditation, transitions, undefined) are dropped.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    chest = raw["signal"]["chest"]
    per_channel = []
    for ch in CHEST_CHANNELS:
        arr = np.asarray(chest[ch], dtype=np.float64).reshape(-1)
        per_channel.append(arr)
    n_samples = min(len(a) for a in per_channel)
    signal = np.stack([a[:n_samples] for a in per_channel], axis=0)  # (5, n_samples)

    labels_raw = np.asarray(raw["label"]).reshape(-1)[:n_samples]

    windows, labels = [], []
    for start in range(0, n_samples - WINDOW_LEN + 1, STEP_LEN):
        end = start + WINDOW_LEN
        seg_labels = labels_raw[start:end]
        vals, counts = np.unique(seg_labels, return_counts=True)
        majority = vals[np.argmax(counts)]
        if majority == 1:
            y = 0  # baseline
        elif majority == 2:
            y = 1  # stress
        else:
            continue  # amusement / meditation / undefined / transition
        if counts.max() / len(seg_labels) < 0.9:
            continue  # require a "clean" (>=90%) label, skip boundary windows
        windows.append(signal[:, start:end].copy())
        labels.append(y)

    return windows, np.array(labels, dtype=np.int64)


def load_wesad_all(data_dir):
    """Scan data_dir for S{N}.pkl subfolders/files, return {cid: (windows, labels)}."""
    clients = {}
    for name in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, name)
        pkl_path = None
        if os.path.isdir(subj_dir):
            candidate = os.path.join(subj_dir, f"{name}.pkl")
            if os.path.exists(candidate):
                pkl_path = candidate
        elif name.endswith(".pkl"):
            pkl_path = subj_dir
            name = name[:-4]
        if pkl_path is None:
            continue
        try:
            windows, labels = load_wesad_subject(pkl_path)
        except Exception as e:
            print(f"[data] skipping {name}: {e}")
            continue
        if len(windows) < 8:
            continue
        clients[name] = (windows, labels)
    if not clients:
        raise RuntimeError(f"No usable S*.pkl found in {data_dir}")
    return clients


# This is the second real dataset: the Wearable Device Dataset from
# Induced Stress and Structured Exercise Sessions (Hongn et al.,
# Scientific Data, 2025, on PhysioNet, recorded with an Empatica E4).
#
# The paper's text and the dataset's own notebook don't fully agree on
# details like the tag pairs and the timestamp format, so the
# segmentation logic below follows what Wearable_Dataset.ipynb actually
# does. See README for more on that.

E4_CHANNELS = ["EDA", "TEMP", "BVP", "ACC"]  # ACC collapsed to 1 channel (magnitude)
N_CHANNELS_E4 = len(E4_CHANNELS)
E4_FS_COMMON = 100  # Hz. Has to be well above the fastest native channel
                     # rate (EDA/TEMP=4Hz, ACC=32Hz, BVP=64Hz), so that
                     # after CNN_DOWNSAMPLE=10 there are still enough
                     # samples left for the CNN's kernels (sizes 15 and 9).
E4_WINDOW_SEC = 10
E4_WINDOW_LEN = E4_WINDOW_SEC * E4_FS_COMMON  # 1000
E4_STEP_SEC = 5
E4_STEP_LEN = E4_STEP_SEC * E4_FS_COMMON  # 500
E4_LABEL_PURITY = 0.9  # window must be >=90% stress or >=90% rest, else dropped

E4_EXPECTED_TAGS = {"v1": 12, "v2": 9}
E4_STRESS_SPANS_V1 = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]  # Stroop, TMCT, Real_Opinion, Opposite_Opinion, Subtract
E4_STRESS_SPANS_V2 = [(2, 3), (4, 5), (6, 7), (8, 9)]  # TMCT, Real_Opinion, Opposite_Opinion, Subtract


def _e4_version(subject_id):
    """S-participants -> 'v1', f-participants -> 'v2'."""
    return "v1" if "s" in subject_id.lower() else "v2"


def _parse_e4_timestamp(raw):
    """E4 timestamps here are 'YYYY-MM-DD HH:MM:SS' strings; fall back to
    a plain unix float if parsing fails."""
    raw = raw.strip()
    try:
        return datetime.datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return float(raw)


def _read_e4_signal(path):
    """Empatica E4 export format: row1=session start time, row2=sample
    rate (Hz), then data (1 column, or 3 for ACC x/y/z)."""
    with open(path, "r") as f:
        t0 = _parse_e4_timestamp(f.readline().split(",")[0])
        fs = float(f.readline().strip().split(",")[0])
        rest = np.loadtxt(f, delimiter=",")
    if rest.ndim == 1:
        rest = rest.reshape(-1, 1)
    return t0, fs, rest


def _resample_to(x, fs_in, fs_out):
    """Linear interpolation to a common sample rate (channels here have
    different native rates, so this is alignment, not just decimation)."""
    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[0]
    duration = n_in / fs_in
    n_out = max(1, int(round(duration * fs_out)))
    t_in = np.arange(n_in) / fs_in
    t_out = np.arange(n_out) / fs_out
    if x.ndim == 1:
        return np.interp(t_out, t_in, x)
    return np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], axis=1)


def load_stress_wearable_subject(subject_dir, subject_id):
    """Load one subject from the STRESS folder of the Hongn et al. dataset,
    return (windows, labels) in the same format as load_wesad_subject."""
    eda_t0, eda_fs, eda = _read_e4_signal(os.path.join(subject_dir, "EDA.csv"))
    temp_t0, temp_fs, temp = _read_e4_signal(os.path.join(subject_dir, "TEMP.csv"))
    bvp_t0, bvp_fs, bvp = _read_e4_signal(os.path.join(subject_dir, "BVP.csv"))
    acc_t0, acc_fs, acc = _read_e4_signal(os.path.join(subject_dir, "ACC.csv"))
    acc_mag = np.sqrt((acc.astype(np.float64) ** 2).sum(axis=1))  # x,y,z -> magnitude

    # Assume the 4 files share the same t0 (documented as time-shifted
    # consistently across records), like the reference notebook does.
    channels_raw = {
        "EDA": (eda_fs, eda[:, 0]),
        "TEMP": (temp_fs, temp[:, 0]),
        "BVP": (bvp_fs, bvp[:, 0]),
        "ACC": (acc_fs, acc_mag),
    }
    resampled = {name: _resample_to(x, fs, E4_FS_COMMON) for name, (fs, x) in channels_raw.items()}
    n_common = min(len(resampled[c]) for c in E4_CHANNELS)
    signal = np.stack([resampled[c][:n_common] for c in E4_CHANNELS], axis=0)  # (4, n_common)

    # tags.csv: raw timestamps, no header, relative to EDA.csv's t0.
    with open(os.path.join(subject_dir, "tags.csv")) as f:
        tag_lines = [ln.strip() for ln in f if ln.strip()]
    tags_relative = [0.0] + [_parse_e4_timestamp(ln) - eda_t0 for ln in tag_lines]

    version = _e4_version(subject_id)
    expected = E4_EXPECTED_TAGS[version]
    n_tags = len(tags_relative) - 1
    if n_tags < expected:
        raise ValueError(
            f"{n_tags} tags in tags.csv, expected at least {expected} "
            f"for protocol {version} -- skipping"
        )
    if n_tags > expected:
        print(f"  [data] {subject_id}: {n_tags} tags (>{expected}), extra ones "
              f"ignored, using tags[1..{expected}] only")

    span_idx = E4_STRESS_SPANS_V1 if version == "v1" else E4_STRESS_SPANS_V2
    stress_spans_sec = [(tags_relative[a], tags_relative[b]) for a, b in span_idx]
    stress_spans = [
        (max(0, int(round(a * E4_FS_COMMON))), min(n_common, int(round(b * E4_FS_COMMON))))
        for a, b in stress_spans_sec
    ]

    windows, labels = [], []
    for ws in range(0, n_common - E4_WINDOW_LEN + 1, E4_STEP_LEN):
        we = ws + E4_WINDOW_LEN
        stress_samples = sum(max(0, min(we, b) - max(ws, a)) for a, b in stress_spans)
        frac = stress_samples / E4_WINDOW_LEN
        if frac >= E4_LABEL_PURITY:
            y = 1
        elif frac <= (1 - E4_LABEL_PURITY):
            y = 0
        else:
            continue  # mixed window, drop it
        windows.append(signal[:, ws:we].copy())
        labels.append(y)

    return windows, np.array(labels, dtype=np.int64)


def load_stress_wearable_all(data_dir):
    """Scan data_dir (the STRESS subfolder of the Hongn et al. dataset)
    for S*/f* subjects, return {cid: (windows, labels)} -- same format
    as load_wesad_all, so it plugs into the rest of the pipeline as-is."""
    clients = {}
    for name in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, name)
        if not os.path.isdir(subj_dir):
            continue
        if not (name.lower().startswith("s") or name.lower().startswith("f")):
            continue
        try:
            windows, labels = load_stress_wearable_subject(subj_dir, name)
        except Exception as e:
            print(f"[data] skipping {name}: {e}")
            continue
        if len(windows) < 8:
            continue
        clients[name] = (windows, labels)
    if not clients:
        raise RuntimeError(f"No usable S*/f* subjects found in {data_dir}")
    return clients


# Synthetic federation (for quick experiments without real data)

_BASE_FREQS = np.array([1.1, 0.15, 0.08, 0.25, 0.02])  # Hz, per-channel base frequencies
_FREQ_SHIFT_SCALE = 0.6
_AMP_SHIFT_SCALE = 0.8
_NOISE_STD = 0.35


def _make_window(rng, profile, reversal, label, length=WINDOW_LEN, fs=FS_CHEST):
    t = np.arange(length) / fs
    window = np.zeros((N_CHANNELS, length))
    for c in range(N_CHANNELS):
        amp, freq = 1.0, _BASE_FREQS[c]
        if label == 1:
            direction = -1.0 if reversal[c] else 1.0
            freq = _BASE_FREQS[c] * (1 + direction * profile[c] * _FREQ_SHIFT_SCALE)
            amp = 1.0 + direction * profile[c] * _AMP_SHIFT_SCALE
        phase = rng.uniform(0, 2 * np.pi)
        drift_phase = rng.uniform(0, 2 * np.pi)
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        drift = 0.05 * np.sin(2 * np.pi * 0.05 * t + drift_phase)
        noise = rng.normal(0, _NOISE_STD, size=length)
        window[c] = signal + drift + noise
    return window


def generate_synthetic_client(cid, n_windows=140, reversal_prob=0.25, seed=0):
    rng = np.random.default_rng(seed)
    profile = rng.dirichlet(np.ones(N_CHANNELS) * 0.8)
    reversal = rng.choice([0, 1], size=N_CHANNELS, p=[1 - reversal_prob, reversal_prob]).astype(bool)

    windows, labels = [], []
    for i in range(n_windows):
        label = i % 2  # alternate baseline/stress, shuffle after
        windows.append(_make_window(rng, profile, reversal, label))
        labels.append(label)
    labels = np.array(labels, dtype=np.int64)
    order = rng.permutation(len(labels))
    windows = [windows[i] for i in order]
    labels = labels[order]
    return windows, labels


def generate_synthetic_federation(n_clients=15, n_windows=140, reversal_prob=0.25, seed=0):
    clients = {}
    for i in range(n_clients):
        cid = f"client_{i}"
        clients[cid] = generate_synthetic_client(
            cid, n_windows=n_windows, reversal_prob=reversal_prob, seed=seed * 1000 + i
        )
    return clients


# Splitting and window representations (statistical features vs raw signal)

def train_val_split(windows, labels, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n_val = max(1, int(len(labels) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    tr = ([windows[i] for i in train_idx], labels[train_idx])
    va = ([windows[i] for i in val_idx], labels[val_idx])
    return tr, va


def extract_window_features(window):
    """6 stats per channel: mean, std, min, max, rms, zero-crossing rate."""
    feats = []
    for c in range(window.shape[0]):
        x = window[c]
        mean, std = x.mean(), x.std()
        mn, mx = x.min(), x.max()
        rms = np.sqrt(np.mean(x ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(x))) > 0)
        feats.extend([mean, std, mn, mx, rms, zcr])
    return np.array(feats, dtype=np.float64)


def downsample_window(window, factor=CNN_DOWNSAMPLE):
    """Mean-pool decimation along the time axis, per channel."""
    c, t = window.shape
    t_trim = (t // factor) * factor
    reshaped = window[:, :t_trim].reshape(c, t_trim // factor, factor)
    return reshaped.mean(axis=2)


class FeatureWindowDataset:
    """Windows as 30-d statistical feature vectors (for the MLP base)."""

    def __init__(self, windows, labels):
        X = np.stack([extract_window_features(w) for w in windows]).astype(np.float64)
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.X = (X - mu) / sigma
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def all(self):
        return self.X, self.y

    def batches(self, batch_size=16, shuffle=True, rng=None):
        n = len(self.y)
        idx = np.arange(n)
        if shuffle:
            rng = rng or np.random.default_rng()
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            yield self.X[b], self.y[b]


class RawWindowDataset:
    """Windows as (n_channels, T') raw, decimated, per-channel normalized
    signal -- input for the 1D CNN base."""

    def __init__(self, windows, labels, downsample=CNN_DOWNSAMPLE):
        ds = [downsample_window(w, downsample) for w in windows]
        X = np.stack(ds).astype(np.float64)  # (N, C, T')
        mu = X.mean(axis=(0, 2), keepdims=True)
        sigma = X.std(axis=(0, 2), keepdims=True) + 1e-8
        self.X = (X - mu) / sigma
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def all(self):
        return self.X, self.y

    def batches(self, batch_size=16, shuffle=True, rng=None):
        n = len(self.y)
        idx = np.arange(n)
        if shuffle:
            rng = rng or np.random.default_rng()
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            yield self.X[b], self.y[b]


def build_dataset(windows, labels, arch="mlp"):
    if arch in ("cnn", "cnn_deep"):
        return RawWindowDataset(windows, labels)
    return FeatureWindowDataset(windows, labels)
