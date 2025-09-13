from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def set_seeds(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Set seeds for random, numpy, and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:  # pragma: no cover - optional path
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

