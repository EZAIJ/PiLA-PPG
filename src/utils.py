import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: Optional[torch.device] = None) -> None:
    if device is None:
        state = torch.load(path)
    else:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
