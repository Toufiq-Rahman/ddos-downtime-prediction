"""Dataset loading and windowing.

The dataset (31 sessions, 1,860 one-minute PRTG readings) is published on
IEEE DataPort: https://dx.doi.org/10.21227/pg53-f517
Download `merged_dataset.csv` and place it in `data/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 5   # five one-minute readings per sample
STRIDE = 3   # window shift (consecutive windows overlap by two readings)

SESSION_COL = "source_file"
LABEL_COL = "session_label"

# The 15 monitored features used in the paper (Section III-C).
FEATURES = [
    "cpu_total_pct", "cpu_core1_pct", "cpu_core2_pct", "cpu_core3_pct", "cpu_core4_pct",
    "downtime_snmp_ms",
    "ping_time_ms", "packet_loss_pct", "http_loading_time_ms",
    "traffic_total_volume_bytes", "traffic_total_speed_bps",
    "traffic_in_volume_bytes", "traffic_in_speed_bps",
    "traffic_out_volume_bytes", "traffic_out_speed_bps",
]
# Ablation: the same features without the SNMP downtime channel (14 features).
FEATURES_NO_DOWNTIME = [f for f in FEATURES if f != "downtime_snmp_ms"]
FEATURE_SETS = {"full": FEATURES, "no_downtime": FEATURES_NO_DOWNTIME}


def load_dataset(path: str) -> pd.DataFrame:
    """Read merged_dataset.csv and return it with a binary `label` column (1 = attack)."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [SESSION_COL, LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {path}: {missing}")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FEATURES] = df[FEATURES].fillna(0.0)
    df["label"] = (df[LABEL_COL].astype(str).str.lower() == "attack").astype(int)
    return df


def make_windows(df: pd.DataFrame, features: list[str]):
    """Slice every session into overlapping windows of shape (WINDOW, n_features).

    Windows never cross session boundaries; each window inherits its session label.
    Returns X (n, WINDOW, F), y (n,), groups (n,) = session id of each window.
    """
    X, y, groups = [], [], []
    for sid, d in df.groupby(SESSION_COL, sort=False):
        a = d[features].to_numpy(np.float32)
        lab = int(d["label"].iloc[0])
        for start in range(0, len(a) - WINDOW + 1, STRIDE):
            X.append(a[start:start + WINDOW])
            y.append(lab)
            groups.append(str(sid))
    return np.stack(X), np.array(y), np.array(groups)
