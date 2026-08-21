"""
flwr.client.NumPyClient with four modes:

  fedavg          -- the whole model is federated with FedAvg.
  fedper          -- only the base is federated; the personal head is local.
  fedper_adaptive -- AdaptiveFedPer: fedper + a few extra local epochs
                     fine-tuning just the personal head, with early stopping.
  ditto           -- Li et al., 2021: a personalized model v_i is trained
                     with a proximal term pulling it toward the federated
                     global model w.

Personal state lives only on the client object itself (self.model). That
way it never leaks between separate runs of run_federation() when you
switch modes.
"""
import zlib
import numpy as np
import flwr as fl

from model import build_model
from data import N_FEATURES, build_dataset, train_val_split
from privacy import clip_and_add_noise, quantize_dequantize


def _stable_seed(cid, run_seed=0):
    """Uses crc32 instead of Python's hash(), since hash() is randomized
    differently every time the process starts. run_seed gets mixed in
    together with cid, not just cid alone -- without that, different
    --seed values ended up giving bit-identical results on real data
    (the std across "different" seeds came out at ~0)."""
    return zlib.crc32(f"{cid}::{run_seed}".encode("utf-8")) % 1_000_000


def _f1_binary(preds, y):
    tp = np.sum((preds == 1) & (y == 1))
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class StressFedClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        windows,
        labels,
        mode="fedper_adaptive",
        arch="mlp",
        local_epochs=1,
        lr=0.05,
        adaptive_max_epochs=8,
        improvement_threshold=0.002,
        mu=0.1,
        run_seed=0,
        balanced_loss=False,
        ema_alpha=0.0,
        dp_clip_norm=None,
        dp_noise_multiplier=0.0,
        quant_bits=None,
        momentum=0.0,
    ):
        self.cid = cid
        self.mode = mode
        self.arch = arch
        self.local_epochs = local_epochs
        self.lr = lr
        self.adaptive_max_epochs = adaptive_max_epochs
        self.improvement_threshold = improvement_threshold
        self.mu = mu  # proximal-term coefficient for ditto
        self.run_seed = run_seed
        self.balanced_loss = balanced_loss
        self.ema_alpha = ema_alpha  # 0 = disabled (raw val loss, as before)
        self.dp_clip_norm = dp_clip_norm  # None = DP disabled
        self.dp_noise_multiplier = dp_noise_multiplier
        self.quant_bits = quant_bits  # None = quantization disabled
        self.momentum = momentum  # 0 = plain SGD, as before

        seed = _stable_seed(cid, run_seed)
        (tr_w, tr_y), (va_w, va_y) = train_val_split(windows, labels, val_frac=0.25, seed=seed)
        self.train_ds = build_dataset(tr_w, tr_y, arch=arch)
        self.val_ds = build_dataset(va_w, va_y, arch=arch)

        # n_features/n_channels come from this client's ACTUAL data shape,
        # not a hardcoded WESAD constant (N_FEATURES=30 for 5 channels) --
        # otherwise this would silently break on a dataset with a different
        # channel count.
        if arch == "mlp":
            n_features, n_channels = self.train_ds.X.shape[1], 5
        else:
            n_features, n_channels = N_FEATURES, self.train_ds.X.shape[1]
        self.n_features = n_features  # needed again in _fit_ditto
        self.n_channels = n_channels  # to build the "shadow" w_model for the same dataset
        self.model = build_model(arch=arch, n_features=n_features, seed=seed, momentum=momentum, n_channels=n_channels)

        # class-balanced weights (n_total / (n_classes * count[c])), so the
        # rare class in a tiny local sample doesn't get lost during adaptive
        # fine-tuning of the personal head (found with CNN: early-stopping
        # on small val sets sometimes locked in a checkpoint that collapsed
        # to the majority class).
        classes, counts = np.unique(self.train_ds.y, return_counts=True)
        n_total = len(self.train_ds.y)
        self._class_weight = np.ones(2)
        for c, cnt in zip(classes, counts):
            self._class_weight[int(c)] = n_total / (len(classes) * cnt)

    # flwr.client.NumPyClient API

    def get_parameters(self, config=None):
        if self.mode in ("fedavg", "ditto"):
            return self.model.get_all_parameters()
        return self.model.get_base_parameters()

    def _set_parameters(self, parameters):
        if self.mode in ("fedavg", "ditto"):
            self.model.set_all_parameters(parameters)
        else:
            self.model.set_base_parameters(parameters)

    def fit(self, parameters, config=None):
        if self.mode == "ditto":
            return self._fit_ditto(parameters)

        received = [p.copy() for p in parameters]
        self._set_parameters(parameters)
        rng = np.random.default_rng(_stable_seed(self.cid, self.run_seed) + 7)

        for _ in range(self.local_epochs):
            for xb, yb in self.train_ds.batches(batch_size=16, shuffle=True, rng=rng):
                self.model.train_step(xb, yb, lr=self.lr, update_base=True)

        adaptive_epochs_used = 0
        if self.mode == "fedper_adaptive":
            adaptive_epochs_used = self._adaptive_finetune(rng)

        out_params = self._postprocess_upload(self.get_parameters(), received)
        return out_params, len(self.train_ds), {"adaptive_epochs": adaptive_epochs_used}

    def _adaptive_finetune(self, rng):
        """Trains a few extra local epochs, but only on the personal head,
        and stops early once the validation loss stops improving. It
        keeps the best checkpoint it saw, not just whatever it ends on.

        balanced_loss gives more weight to the rare class. ema_alpha, if
        set above 0, smooths the loss curve used for the stop/continue
        decision -- but the checkpoint it saves as "best" is still picked
        using the raw (unsmoothed) loss."""
        w = self._class_weight if self.balanced_loss else None
        prev_loss = self._val_loss(w)
        smoothed = prev_loss
        best_loss = prev_loss
        best_state = self.model.personal.state_dict()
        adaptive_epochs_used = 0
        for _ in range(self.adaptive_max_epochs):
            for xb, yb in self.train_ds.batches(batch_size=16, shuffle=True, rng=rng):
                sw = self._class_weight[yb] if self.balanced_loss else None
                self.model.train_step(xb, yb, lr=self.lr * 0.5, update_base=False, sample_weight=sw)
            adaptive_epochs_used += 1
            cur_loss = self._val_loss(w)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_state = self.model.personal.state_dict()
            if self.ema_alpha > 0:
                smoothed = self.ema_alpha * cur_loss + (1 - self.ema_alpha) * smoothed
                improvement = prev_loss - smoothed
                prev_loss = smoothed
            else:
                improvement = prev_loss - cur_loss
                prev_loss = cur_loss
            if improvement < self.improvement_threshold:
                break
        self.model.personal.load_state_dict(best_state)
        return adaptive_epochs_used

    def _postprocess_upload(self, trained_params, received_params):
        """DP and/or quantization applied to what actually goes over the
        wire. The personal head never reaches here -- it never leaves the
        device."""
        out = trained_params
        if self.dp_clip_norm is not None:
            delta = [t - r for t, r in zip(trained_params, received_params)]
            noise_seed = _stable_seed(self.cid, self.run_seed) + 99
            delta = clip_and_add_noise(delta, self.dp_clip_norm, self.dp_noise_multiplier, seed=noise_seed)
            out = [r + d for r, d in zip(received_params, delta)]
        if self.quant_bits is not None:
            out = quantize_dequantize(out, self.quant_bits)
        return out

    def _fit_ditto(self, global_params):
        rng = np.random.default_rng(_stable_seed(self.cid, self.run_seed) + 7)
        received = [p.copy() for p in global_params]

        # 1) Personalized model v (self.model): a local SGD step, then a
        #    proximal correction pulling v toward the global w.
        for _ in range(self.local_epochs):
            for xb, yb in self.train_ds.batches(batch_size=16, shuffle=True, rng=rng):
                self.model.train_step(xb, yb, lr=self.lr, update_base=True)
                cur = self.model.get_all_parameters()
                corrected = [c - self.lr * self.mu * (c - r) for c, r in zip(cur, global_params)]
                self.model.set_all_parameters(corrected)

        # 2) A separate copy of the global model w: plain local SGD with NO
        #    regularization -- this is the one that gets federated (aggregated).
        w_model = build_model(arch=self.arch, n_features=self.n_features, seed=0, momentum=self.momentum, n_channels=self.n_channels)
        w_model.set_all_parameters(global_params)
        for _ in range(self.local_epochs):
            for xb, yb in self.train_ds.batches(batch_size=16, shuffle=True, rng=rng):
                w_model.train_step(xb, yb, lr=self.lr, update_base=True)

        out_params = self._postprocess_upload(w_model.get_all_parameters(), received)
        return out_params, len(self.train_ds), {"adaptive_epochs": 0}

    def evaluate(self, parameters, config=None):
        if self.mode == "ditto":
            # Evaluated with the PERSONALIZED model v (self.model), not w --
            # matches how Ditto reports results in the original paper.
            X, y = self.val_ds.all()
            preds, probs = self.model.predict(X)
            loss = self.model.loss(X, y)
        else:
            self._set_parameters(parameters)
            X, y = self.val_ds.all()
            preds, probs = self.model.predict(X)
            loss = self.model.loss(X, y)

        acc = float((preds == y).mean())
        f1 = _f1_binary(preds, y)
        return float(loss), len(self.val_ds), {"accuracy": acc, "f1": f1, "cid": self.cid}

    def _val_loss(self, class_weight=None):
        X, y = self.val_ds.all()
        sw = class_weight[y] if class_weight is not None else None
        return self.model.loss(X, y, sample_weight=sw)
