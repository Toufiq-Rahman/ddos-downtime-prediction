#!/usr/bin/env python
"""Print trainable-parameter counts of the PyTorch models (15 input features)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ddos_downtime.models import EXTRA_DEEP_MODELS, MAIN_MODELS  # noqa: E402

for name, fn in {**MAIN_MODELS, **EXTRA_DEEP_MODELS}.items():
    print(f"{name:18s} {sum(p.numel() for p in fn(15).parameters() if p.requires_grad):>7,d}")
