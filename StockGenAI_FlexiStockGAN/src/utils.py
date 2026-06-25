import os
import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dirs(base="outputs"):
    base = Path(base)
    for sub in ["models", "predictions", "metrics", "figures"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def inverse_close_from_scaled(y_scaled, close_min, close_max):
    return y_scaled * (close_max - close_min + 1e-12) + close_min
