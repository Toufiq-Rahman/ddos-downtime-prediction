"""PyTorch implementations of the proposed model and all compared models.

Layer sizes follow the paper; parameter counts reported in the paper are the
equivalent Keras counts (PyTorch LSTM/GRU layers carry one extra bias vector).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Keras-style multi-head attention: per-head key dimension independent of input size."""

    def __init__(self, d_in: int, heads: int = 4, key_dim: int = 15):
        super().__init__()
        self.h, self.kd = heads, key_dim
        self.q = nn.Linear(d_in, heads * key_dim)
        self.k = nn.Linear(d_in, heads * key_dim)
        self.v = nn.Linear(d_in, heads * key_dim)
        self.o = nn.Linear(heads * key_dim, d_in)

    def forward(self, x):
        b, t, _ = x.shape
        split = lambda z: z.view(b, t, self.h, self.kd).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = torch.softmax(q @ k.transpose(-1, -2) / self.kd ** 0.5, dim=-1)
        return self.o((att @ v).transpose(1, 2).reshape(b, t, -1))


class TransformerBiLSTM(nn.Module):
    """Proposed model: LayerNorm -> MHSA (+residual, LayerNorm) -> BiLSTM(32)
    -> TimeDistributed Dense(32, ReLU) -> Dense(15) -> GAP -> Dense(32, ReLU) -> Dropout(0.5) -> 2."""

    def __init__(self, n_features: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_features, eps=1e-6)
        self.attn = MultiHeadSelfAttention(n_features)
        self.norm2 = nn.LayerNorm(n_features, eps=1e-6)
        self.bilstm = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
        self.td = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 15))
        self.head = nn.Sequential(nn.Linear(15, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))

    def forward(self, x):
        x = self.norm1(x)
        x = self.norm2(self.attn(x) + x)
        h, _ = self.bilstm(x)
        return self.head(self.td(h).mean(dim=1))


class TransformerOnly(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_features, eps=1e-6)
        self.attn = MultiHeadSelfAttention(n_features)
        self.norm2 = nn.LayerNorm(n_features, eps=1e-6)
        self.head = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))

    def forward(self, x):
        x = self.norm1(x)
        return self.head(self.norm2(self.attn(x) + x).mean(dim=1))


class CNNResNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.c0 = nn.Conv1d(n_features, 64, 3, padding=1)
        self.c1 = nn.Conv1d(64, 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 64, 3, padding=1)
        self.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        z = F.max_pool1d(torch.relu(self.c0(x.transpose(1, 2))), 2)
        z = self.c2(torch.relu(self.c1(z))) + z
        return self.head(F.avg_pool1d(z, 2).flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv = nn.Conv1d(n_features, 64, 3)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2))

    def forward(self, x):
        z = F.max_pool1d(torch.relu(self.conv(x.transpose(1, 2))), 2).transpose(1, 2)
        h, _ = self.lstm(z)
        return self.head(h.flatten(1))


class BiLSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[0], h[1]], dim=1))


# ---- additional models (Section V-C) ------------------------------------------
class Recurrent(nn.Module):
    def __init__(self, n_features: int, cell: str = "lstm", bidirectional: bool = False, layers: int = 1):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, 32, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional, dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(64 if bidirectional else 32, 32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h.mean(dim=1))


class TCN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.c1 = nn.Conv1d(n_features, 32, 2, dilation=1, padding=1)
        self.c2 = nn.Conv1d(32, 32, 2, dilation=2, padding=2)
        self.res = nn.Conv1d(n_features, 32, 1)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x):
        z = x.transpose(1, 2)
        t = z.shape[-1]
        a = torch.relu(self.c1(z))[..., :t]
        b = torch.relu(self.c2(a))[..., :t]
        return self.head((b + self.res(z)).mean(dim=2))


class MLP(nn.Module):
    def __init__(self, n_features: int, window: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(window * n_features, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


# Models of Tables I and II
MAIN_MODELS = {
    "CNN (ResNet)": CNNResNet,
    "CNN-LSTM": CNNLSTM,
    "BiLSTM": BiLSTM,
    "Transformer": TransformerOnly,
    "Proposed": TransformerBiLSTM,
}
# Additional deep models of Section V-C
EXTRA_DEEP_MODELS = {
    "LSTM": lambda f: Recurrent(f, "lstm"),
    "GRU": lambda f: Recurrent(f, "gru"),
    "Two-layer BiLSTM": lambda f: Recurrent(f, "lstm", bidirectional=True, layers=2),
    "TCN": TCN,
    "MLP": MLP,
}
