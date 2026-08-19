"""
Grad-CAM saliency for the convolutional classifiers.

Grad-CAM weights the final convolutional feature maps by the gradient of the
predicted class with respect to those maps, producing a coarse heatmap of the
evidence the network actually used. On this dataset it is the quickest way to
check that a 98% model is looking at the animal and not at a background cue.

TensorFlow is imported lazily inside `make_gradcam_heatmap`, so importing this
module (and the test suite) does not require it installed.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..common.viz import save_or_show


def find_last_conv_layer(model: Any) -> str:
    """Name the deepest layer with a 4-D output.

    Nested models (MobileNetV2 wrapped in a classifier head) are searched
    recursively, since the useful feature maps live inside the backbone.
    """
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            try:
                inner = find_last_conv_layer(layer)
                return f"{layer.name}/{inner}"
            except ValueError:
                pass

        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.name

    raise ValueError("No 4-D convolutional output found; Grad-CAM does not apply.")


def _resolve_layer(model: Any, layer_name: str) -> tuple[Any, Any]:
    """Resolve a possibly nested `outer/inner` layer path.

    Returns the model that owns the layer and the layer itself.
    """
    if "/" not in layer_name:
        return model, model.get_layer(layer_name)

    outer_name, inner_name = layer_name.split("/", 1)
    outer = model.get_layer(outer_name)
    return outer, outer.get_layer(inner_name)


def load_image_array(
    image_path: str | Path,
    *,
    image_size: tuple[int, int] = (160, 160),
) -> np.ndarray:
    """Load one image as a batch of shape `(1, h, w, 3)` in [0, 255] float32.

    The scale matches the training pipeline, where rescaling and preprocessing
    happen inside the model rather than in the loader.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(image_size)
        array = np.asarray(image, dtype=np.float32)
    return array[np.newaxis, ...]


def make_gradcam_heatmap(
    image_batch: np.ndarray,
    model: Any,
    *,
    layer_name: str | None = None,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute a normalised Grad-CAM heatmap for a single image.

    Args:
        image_batch: Array of shape `(1, h, w, 3)`.
        model: Trained Keras classifier.
        layer_name: Target layer; the deepest 4-D layer is used when omitted.
        pred_index: Class to explain; the top prediction is used when omitted.

    Returns:
        The heatmap in [0, 1], the explained class index, and its confidence.
    """
    import tensorflow as tf

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    _, target_layer = _resolve_layer(model, layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, pred_index]
        confidence = float(predictions[0][pred_index])

    gradients = tape.gradient(class_score, conv_output)
    if gradients is None:
        raise ValueError(
            f"No gradient flows from the output to layer '{layer_name}'. "
            "Pick a layer inside the trainable path."
        )

    # Average each feature map's gradient into a single importance weight,
    # then take the weighted sum of the maps
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_output[0] @ weights[..., tf.newaxis])

    # Keep positive evidence only, then scale to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), confidence


def overlay_heatmap(
    image_path: str | Path,
    heatmap: np.ndarray,
    *,
    image_size: tuple[int, int] = (160, 160),
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap over the source image as an RGB uint8 array."""
    with Image.open(image_path) as image:
        base = np.asarray(image.convert("RGB").resize(image_size), dtype=np.float32) / 255.0

    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        image_size, resample=Image.BILINEAR
    )
    colored = plt.get_cmap(colormap)(np.asarray(heatmap_image) / 255.0)[..., :3]

    blended = (1 - alpha) * base + alpha * colored
    return np.uint8(np.clip(blended, 0, 1) * 255)


def plot_gradcam_examples(
    image_paths: list[str | Path],
    model: Any,
    class_names: list[str],
    *,
    image_size: tuple[int, int] = (160, 160),
    layer_name: str | None = None,
    save_path: Path | None = None,
) -> None:
    """Show original / heatmap / overlay for each image, one row apiece."""
    if not image_paths:
        return

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    plt.figure(figsize=(11, 3.6 * len(image_paths)))

    for row, path in enumerate(image_paths):
        batch = load_image_array(path, image_size=image_size)
        heatmap, pred_index, confidence = make_gradcam_heatmap(batch, model, layer_name=layer_name)
        overlay = overlay_heatmap(path, heatmap, image_size=image_size)

        plt.subplot(len(image_paths), 3, row * 3 + 1)
        plt.imshow(np.uint8(batch[0]))
        plt.title("Original", fontsize=10)
        plt.axis("off")

        plt.subplot(len(image_paths), 3, row * 3 + 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Grad-CAM", fontsize=10)
        plt.axis("off")

        plt.subplot(len(image_paths), 3, row * 3 + 3)
        plt.imshow(overlay)
        plt.title(f"{class_names[pred_index]} ({confidence:.1%})", fontsize=10)
        plt.axis("off")

    save_or_show(save_path)
