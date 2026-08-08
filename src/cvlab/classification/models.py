"""
The four classification approaches compared in part 1.

Ordered by increasing prior knowledge about images:

1. raw pixels + logistic regression — no spatial prior at all;
2. HOG + linear SVM — hand-designed gradient prior, tuned via grid search;
3. small CNN trained from scratch — learned convolutional prior;
4. MobileNetV2 transfer learning — convolutional prior *plus* ImageNet features.

TensorFlow is imported lazily inside the deep-learning helpers so the classical
baselines (and the test suite) can run without it installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tensorflow as tf


# --------------------------------------------------------------------------- #
# Classical baselines
# --------------------------------------------------------------------------- #


def build_pixel_baseline(*, use_class_weights: bool = True, max_iter: int = 1000) -> Pipeline:
    """Logistic regression on raw flattened pixels.

    `with_mean=False` keeps the scaler from densifying and re-centring the very
    wide pixel matrix; only per-feature variance is normalised.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced" if use_class_weights else None,
                ),
            ),
        ]
    )


def build_hog_svm_grid(
    *,
    use_class_weights: bool = True,
    c_values: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0),
    cv_folds: int = 3,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Linear SVM over HOG features with cross-validated `C`.

    The grid search runs on the training split only, so hyperparameter choice
    never sees validation or test data.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced" if use_class_weights else None)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={"clf__C": list(c_values)},
        scoring=scoring,
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def compute_class_weight_dict(y: np.ndarray) -> dict[int, float]:
    """Map each label to an inverse-frequency weight."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights, strict=True)}


# --------------------------------------------------------------------------- #
# tf.data input pipeline
# --------------------------------------------------------------------------- #


def build_augmentation_layer(name: str = "data_augmentation") -> Any:
    """Light geometric augmentation applied to training batches only.

    Deliberately conservative: horizontal flips plus small rotation and zoom.
    Aggressive augmentation on ~1400 training images tends to hurt more than it
    helps, and vertical flips are not realistic for photographs of animals.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
        ],
        name=name,
    )


def make_tf_dataset(
    df: pd.DataFrame,
    *,
    image_size: tuple[int, int] = (160, 160),
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a batched `tf.data.Dataset` of `(image, label)` pairs.

    Evaluation splits are never shuffled, so predictions stay aligned with the
    label order of the source DataFrame.
    """
    paths = df["file_path"].astype(str).to_numpy()
    labels = df["label"].astype(np.int32).to_numpy()

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Neural networks
# --------------------------------------------------------------------------- #


def build_small_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    augmentation: Any | None = None,
) -> Any:
    """Compact three-block CNN trained from scratch.

    `GlobalAveragePooling2D` replaces a wide flatten-plus-dense head, which cuts
    parameter count sharply — the right trade-off when training data is scarce.
    """
    if augmentation is None:
        augmentation = build_augmentation_layer()

    inputs = keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Dropout(0.30)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="small_cnn")


def build_mobilenetv2_transfer(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    augmentation: Any | None = None,
) -> tuple[Any, Any]:
    """MobileNetV2 with a frozen ImageNet backbone and a fresh classifier head.

    Returns both the full model and the backbone, since the fine-tuning stage
    needs a handle on the backbone to unfreeze its upper layers.
    """
    if augmentation is None:
        augmentation = build_augmentation_layer()

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    # training=False keeps the frozen BatchNorm layers in inference mode.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model, backbone


def compile_model(model: Any, *, learning_rate: float = 1e-3) -> Any:
    """Compile with Adam and sparse categorical cross-entropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(backbone: Any, *, fine_tune_at: int = 100) -> Any:
    """Unfreeze the backbone above `fine_tune_at`, keeping earlier layers frozen.

    Early layers encode generic edges and textures that transfer as-is; only the
    more task-specific upper layers are worth adapting on a small dataset.
    """
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return backbone


def early_stopping(*, patience: int = 3, monitor: str = "val_loss") -> Any:
    """Early stopping that restores the best weights seen."""
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
    )
