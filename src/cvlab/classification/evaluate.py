"""
Evaluation metrics, curves and error inspection.

Macro F1 is the primary metric throughout: it weights both classes equally, so a
model cannot look good by favouring the majority class. Accuracy is reported
alongside it for readability, and ROC-AUC / average precision are computed for
binary problems where a score is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..common.viz import save_or_show

METRIC_COLUMNS = ["accuracy", "f1_macro", "f1_weighted", "roc_auc", "average_precision"]


@dataclass
class EvalResult:
    """Metrics and predictions for one model on one split."""

    model_name: str
    split_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float | None
    average_precision: float | None
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    scores: np.ndarray | None = field(default=None, repr=False)

    def as_row(self) -> dict[str, Any]:
        """Flatten into a row for a summary table."""
        return {
            "model": self.model_name,
            "split": self.split_name,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
        }

    def report(self, class_names: list[str]) -> str:
        """Per-class precision/recall/F1 as text."""
        return classification_report(
            self.y_true, self.y_pred, target_names=class_names, digits=3, zero_division=0
        )


def _binary_scores(
    y_true: np.ndarray, scores: np.ndarray | None
) -> tuple[float | None, float | None]:
    """ROC-AUC and average precision, or (None, None) when not applicable."""
    if scores is None or len(np.unique(y_true)) != 2:
        return None, None
    return (
        float(roc_auc_score(y_true, scores)),
        float(average_precision_score(y_true, scores)),
    )


def _extract_scores(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Positive-class score from either `predict_proba` or `decision_function`."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_sklearn_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    model_name: str,
    split_name: str = "validation",
) -> EvalResult:
    """Evaluate a fitted scikit-learn estimator."""
    y_pred = model.predict(X)
    scores = _extract_scores(model, X)
    roc_auc, avg_precision = _binary_scores(y_true, scores)

    return EvalResult(
        model_name=model_name,
        split_name=split_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted")),
        roc_auc=roc_auc,
        average_precision=avg_precision,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        scores=scores,
    )


def evaluate_keras_model(
    model: Any,
    dataset: Any,
    *,
    model_name: str,
    split_name: str = "validation",
) -> EvalResult:
    """Evaluate a Keras model over a non-shuffled `tf.data.Dataset`.

    Labels are collected by iterating the dataset rather than reusing the source
    DataFrame, which guarantees they line up with the predictions even if the
    pipeline drops or reorders elements.
    """
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    probabilities = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction/label length mismatch: {len(y_pred)} predictions vs "
            f"{len(y_true)} labels. Ensure the evaluation dataset is not shuffled."
        )

    scores = probabilities[:, 1] if probabilities.shape[1] == 2 else None
    roc_auc, avg_precision = _binary_scores(y_true, scores)

    return EvalResult(
        model_name=model_name,
        split_name=split_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted")),
        roc_auc=roc_auc,
        average_precision=avg_precision,
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
    )


def summarize(results: list[EvalResult], *, sort_by: str = "f1_macro") -> pd.DataFrame:
    """Combine results into a table sorted by the primary metric."""
    frame = pd.DataFrame([r.as_row() for r in results])
    if sort_by in frame.columns:
        frame = frame.sort_values(
by=sort_by, ascending=False).reset_index(drop=True)
    return frame

# --------------------------------- Plots ---------------------------------


def plot_confusion_matrix(
    result: EvalResult,
    class_names: list[str],
    *,
    normalize: str | None = "true",
    save_path: Path | None = None,
) -> None:
    """Row-normalised confusion matrix, so per-class recall reads off the diagonal."""
    matrix = confusion_matrix(
        result.y_true, result.y_pred, normalize=normalize)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", colorbar=False, values_format=".2f")
    plt.title(f"{result.model_name} — {result.split_name}")
    plt.xticks(rotation=45)
    save_or_show(save_path)


def plot_roc_and_pr(result: EvalResult, *, save_path: Path | None = None) -> None:
    """Side-by-side ROC and precision-recall curves for a binary result."""
    if result.scores is None or result.roc_auc is None:
        return

    fpr, tpr, _ = roc_curve(result.y_true, result.scores)
    precision, recall, _ = precision_recall_curve(result.y_true, result.scores)

    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC-AUC = {result.roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{result.model_name}: ROC")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AP = {result.average_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{result.model_name}: Precision-Recall")
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show(save_path)


def plot_history(
    history: Any, *, title_prefix: str = "Model", save_path: Path | None = None
) -> None:
    """Training and validation accuracy/loss across epochs."""
    metrics = history.history if hasattr(history, "history") else history

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics["accuracy"], label="train")
    plt.plot(metrics["val_accuracy"], label="validation")
    plt.title(f"{title_prefix}: accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["loss"], label="train")
    plt.plot(metrics["val_loss"], label="validation")
    plt.title(f"{title_prefix}: loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    save_or_show(save_path)


def plot_model_comparison(
    summary: pd.DataFrame,
    *,
    metric: str = "f1_macro",
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart ranking models by one metric."""
    frame = summary.sort_values(by=metric)

    plt.figure(figsize=(9, 0.8 * len(frame) + 1.5))
    plt.barh(frame["model"], frame[metric], color="#4C78A8")
    plt.xlabel(metric)
    plt.xlim(0, 1.0)
    plt.title(f"Model comparison by {metric}")
    plt.grid(axis="x", alpha=0.3)

    for y, value in enumerate(frame[metric]):
        plt.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=9)

    save_or_show(save_path)


def show_misclassified(
    df: pd.DataFrame,
    result: EvalResult,
    class_names: list[str],
    *,
    n_examples: int = 9,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Plot the misclassified images and return them as a table.

    The DataFrame must be the split that produced `result`, with its index reset,
    so positional indices line up with the prediction arrays.
    """
    wrong = np.where(result.y_true != result.y_pred)[0]
    if len(wrong) == 0:
        print("No misclassified examples on this split.")
        return pd.DataFrame(columns=["file_path", "true", "predicted"])

    chosen = wrong[:n_examples]
    n_cols = 3
    n_rows = (len(chosen) + n_cols - 1) // n_cols

    plt.figure(figsize=(4 * n_cols, 3.6 * n_rows))
    rows = []

    for position, idx in enumerate(chosen):
        row = df.iloc[int(idx)]
        true_name = class_names[int(result.y_true[idx])]
        pred_name = class_names[int(result.y_pred[idx])]

        with Image.open(row["file_path"]) as image:
            image = image.convert("RGB")
            plt.subplot(n_rows, n_cols, position + 1)
            plt.imshow(image)

        plt.title(f"true: {true_name}\npredicted: {pred_name}", fontsize=10)
        plt.axis("off")
        rows.append({"file_path": row["file_path"],
                    "true": true_name, "predicted": pred_name})

    plt.suptitle(
        f"{result.model_name}: misclassified examples ({len(wrong)} total)")
    save_or_show(save_path)

    return pd.DataFrame(rows)
