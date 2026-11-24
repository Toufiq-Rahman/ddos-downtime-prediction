"""Cross-validation protocols and the training loop (Section IV)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

METRICS = ["accuracy", "precision", "recall", "f1"]


# ---- protocols -----------------------------------------------------------------
def window_level_folds(y, groups, seed, k=5):
    """(i) Window-level CV: windows pooled and split at random (sessions may span folds)."""
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def session_independent_folds(y, groups, seed, k=5):
    """(ii) Session-independent CV: all windows of a session stay in one fold."""
    folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "session leakage between train and test"
    return folds


PROTOCOLS = {"window": window_level_folds, "session": session_independent_folds}


# ---- helpers -------------------------------------------------------------------
def standardize(X, train_idx):
    """Standardize with statistics of the training windows only."""
    flat = X[train_idx].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def validation_sessions(y, groups, train_idx, rng, n_benign=2, n_attack=1):
    """Pick held-out validation sessions (from the training folds) for early stopping."""
    sessions = np.array(sorted(set(groups[train_idx])))
    rng.shuffle(sessions)
    lab = {s: y[groups == s][0] for s in sessions}
    val = [s for s in sessions if lab[s] == 0][:n_benign] + [s for s in sessions if lab[s] == 1][:n_attack]
    return np.isin(groups[train_idx], val)


def scores(y_true, y_pred):
    return [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)]


def train_torch(model, X_tr, y_tr, X_val, y_val, epochs=50, patience=10, batch_size=32, lr=1e-3):
    """Adam + cross-entropy, early stopping on validation loss, best weights restored."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_tr, y_tr = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X_tr[b]), y_tr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_val), y_val).item()
        if v < best:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_fold_torch(model_fn, X, y, groups, tr, te, seed, epochs=50):
    """Train a FRESH model on one fold and return predictions for the test windows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Z = standardize(X, tr)
    vmask = validation_sessions(y, groups, tr, rng)
    tr_fit, tr_val = tr[~vmask], tr[vmask]
    model = train_torch(model_fn(X.shape[-1]), Z[tr_fit], y[tr_fit], Z[tr_val], y[tr_val], epochs=epochs)
    with torch.no_grad():
        return model(torch.from_numpy(Z[te])).argmax(1).numpy()


def flatten_window_features(Z):
    """Raw window + per-feature mean/std/min/max/slope, for classical ML models."""
    return np.concatenate([Z.reshape(len(Z), -1), Z.mean(1), Z.std(1), Z.min(1), Z.max(1), Z[:, -1] - Z[:, 0]], 1)


def run_fold_sklearn(make_model, X, y, tr, te, seed):
    Z = standardize(X, tr)
    model = make_model(seed).fit(flatten_window_features(Z[tr]), y[tr])
    return model.predict(flatten_window_features(Z[te]))


def summarize(rows):
    r = np.asarray(rows)
    return r.mean(0), r.std(0)
