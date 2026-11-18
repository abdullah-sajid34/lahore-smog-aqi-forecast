"""
utils.py -- shared utility helpers.

Keeping set_seed() here (rather than in config.py) separates *values*
(config) from *behaviour* (utils).  Every script that needs reproducibility
calls utils.set_seed() once at startup.
"""

import random
import numpy as np

from src.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy (and optionally PyTorch/TF if available).

    WHY: ML experiments must be reproducible.  A single call here covers all
    libraries in use; we do not scatter np.random.seed() calls across files.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Seed PyTorch if installed (not required for Phase 0/1)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass