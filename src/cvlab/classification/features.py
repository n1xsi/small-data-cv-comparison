"""
Hand-engineered feature extraction for the classical baselines.

Two representations are compared: raw flattened pixels (the weakest honest
baseline) and HOG descriptors, which discard absolute intensity and keep the
distribution of gradient orientations instead.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog
from tqdm.auto import tqdm

# Shared HOG parameters
# Kept in one place so the visualisation and the feature extractor cannot drift apart
HOG_PARAMS: dict[str, object] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def flatten_pixels(images: np.ndarray) -> np.ndarray:
    """Flatten each image into a 1-D feature vector.

    A 64x64 grayscale image becomes 4096 features with no spatial structure —
    which is precisely why a linear model performs near chance on this input.
    """
    return images.reshape(len(images), -1)


def extract_hog_features(images: np.ndarray, *, show_progress: bool = True) -> np.ndarray:
    """Compute HOG descriptors for a stack of grayscale images.

    Args:
        images: Array of shape `(n, h, w)` with values in [0, 1].
        show_progress: Display a progress bar.

    Returns:
        Array of shape `(n, n_features)`, dtype float32.
    """
    features: list[np.ndarray] = []
    iterator = tqdm(images, total=len(images), desc="Extracting HOG", disable=not show_progress)

    for image in iterator:
        descriptor = hog(image, visualize=False, feature_vector=True, **HOG_PARAMS)
        features.append(descriptor.astype(np.float32))

    return np.vstack(features)


def hog_with_visualisation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the HOG descriptor and its renderable visualisation.

    Separate from `extract_hog_features` because `visualize=True` roughly doubles
    the cost and is only wanted for a handful of illustrative examples.
    """
    descriptor, hog_image = hog(image, visualize=True, feature_vector=True, **HOG_PARAMS)
    return descriptor.astype(np.float32), hog_image
