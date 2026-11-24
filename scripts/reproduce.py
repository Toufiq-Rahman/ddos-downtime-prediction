#!/usr/bin/env python
"""Reproduce the experiments of the paper.

Examples
--------
# Main comparison (Tables I/II) under both protocols
python scripts/reproduce.py main --protocol window
python scripts/reproduce.py main --protocol session
# Ablation without the SNMP downtime channel (Section V-C)
python scripts/reproduce.py main --protocol session --features no_downtime
# Additional models (Section V-C)
python scripts/reproduce.py extra
# PCA study (Table III)
python scripts/reproduce.py pca --protocol session

Results are printed and written as CSV files to results/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from ddos_downtime.data import FEATURE_SETS, load_dataset, make_windows  # noqa: E402
from ddos_downtime.evaluation import (METRICS, PROTOCOLS, run_fold_sklearn, run_fold_torch,  # noqa: E402
                                      scores, summarize)
from ddos_downtime.models import EXTRA_DEEP_MODELS, MAIN_MODELS, TransformerBiLSTM  # noqa: E402

SKLEARN_MODELS = {
    "Random Forest": lambda s: RandomForestClassifier(300, random_state=s, n_jobs=-1, class_weight="balanced"),
    "Extra Trees": lambda s: ExtraTreesClassifier(300, random_state=s, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting": lambda s: HistGradientBoostingClassifier(random_state=s),
    "Logistic Regression": lambda s: LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
    "SVM (RBF)": lambda s: SVC(C=1.0, class_weight="balanced"),
}


def evaluate(name, X, y, groups, protocol, seeds, epochs, model_fn=None, sk_fn=None, transform=None):
    rows = []
    for seed in seeds:
        for tr, te in PROTOCOLS[protocol](y, groups, seed):
            Xs = transform(X, tr) if transform else X
            if sk_fn is not None:
                pred = run_fold_sklearn(sk_fn, Xs, y, tr, te, seed)
            else:
                pred = run_fold_torch(model_fn, Xs, y, groups, tr, te, seed, epochs=epochs)
            rows.append(scores(y[te], pred))
    mean, std = summarize(rows)
    print(f"  {name:22s} " + "  ".join(f"{m}={a:.3f}±{s:.3f}" for m, a, s in zip(METRICS, mean, std)), flush=True)
    return {"model": name, **{m: round(a, 4) for m, a in zip(METRICS, mean)},
            **{f"{m}_std": round(s, 4) for m, s in zip(METRICS, std)}, "runs": len(rows)}


def pca_transform(k):
    def f(X, tr):
        flat_tr = X[tr].reshape(-1, X.shape[-1])
        mu, sd = flat_tr.mean(0), flat_tr.std(0) + 1e-6
        Z = (X - mu) / sd
        p = PCA(k).fit(Z[tr].reshape(-1, X.shape[-1]))
        return p.transform(Z.reshape(-1, X.shape[-1])).reshape(len(X), X.shape[1], k).astype(np.float32)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["main", "extra", "pca"])
    ap.add_argument("--data", default="data/merged_dataset.csv")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="session")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="full")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated random seeds (5 folds each)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if not os.path.exists(a.data):
        sys.exit(f"Dataset not found at {a.data}. Download merged_dataset.csv from IEEE DataPort "
                 "(https://dx.doi.org/10.21227/pg53-f517) and place it in data/.")
    df = load_dataset(a.data)
    X, y, groups = make_windows(df, FEATURE_SETS[a.features])
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"{len(set(groups))} sessions -> {len(y)} windows ({(y == 0).sum()} benign, {(y == 1).sum()} attack), "
          f"{X.shape[-1]} features | protocol={a.protocol} | seeds={seeds}", flush=True)

    t0, results = time.time(), []
    if a.experiment == "main":
        for name, fn in MAIN_MODELS.items():
            results.append(evaluate(name, X, y, groups, a.protocol, seeds, a.epochs, model_fn=fn))
        tag = f"main_{a.protocol}_{a.features}"
    elif a.experiment == "extra":
        for name, fn in EXTRA_DEEP_MODELS.items():
            results.append(evaluate(name, X, y, groups, a.protocol, seeds, a.epochs, model_fn=fn))
        for name, fn in SKLEARN_MODELS.items():
            results.append(evaluate(name, X, y, groups, a.protocol, seeds, a.epochs, sk_fn=fn))
        tag = f"extra_{a.protocol}_{a.features}"
    else:
        for k in (3, 5, 7, 9):
            results.append(evaluate(f"PCA({k})", X, y, groups, a.protocol, seeds, a.epochs,
                                    model_fn=TransformerBiLSTM, transform=pca_transform(k)))
        tag = f"pca_{a.protocol}_{a.features}"

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{tag}.csv")
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Saved {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
