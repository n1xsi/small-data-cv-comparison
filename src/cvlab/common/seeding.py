"""
Seed control for reproducible runs.

Every experiment in this project is seeded from a single value so that splits,
shuffles and weight initialisation can be reproduced exactly. Deep-learning
frameworks are seeded lazily: importing TensorFlow or PyTorch is expensive, so
they are only touched if they are already installed.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED, *, tensorflow: bool = True, torch: bool = False) -> int:
    """Seed Python, NumPy and (optionally) the deep-learning frameworks.

    Args:
        seed: Value applied to every random number generator.
        tensorflow: Seed TensorFlow if it is importable.
        torch: Seed PyTorch if it is importable.

    Returns:
        The seed that was applied, so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if tensorflow:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass

    if torch:
        try:
            import torch as torch_module

            torch_module.manual_seed(seed)
            if torch_module.cuda.is_available():
                torch_module.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    return seed


def new_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return an independent NumPy generator.

    Preferred over reseeding the global NumPy state when a function needs
    randomness of its own without disturbing the rest of the pipeline.
    """
    return np.random.default_rng(seed)
