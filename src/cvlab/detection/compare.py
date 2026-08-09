"""
Side-by-side detector comparison and error analysis.

YOLOv8 and D-FINE report metrics through completely different channels — an
Ultralytics object versus a parsed pycocotools log — so this module normalises
both onto one table before anything is compared or plotted.

Both are evaluated on the same COCO-derived split, which is what makes the
comparison meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..common.viz import save_or_show

# Shared metric columns; not every backend fills every one
COMPARISON_COLUMNS = ("model", "split", "mAP50_95", "mAP50", "mAP75", "precision", "recall")


@dataclass
class DetectorResult:
    """One detector evaluated on one split."""

    model: str
    split: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {"model": self.model, "split": self.split}
        for column in COMPARISON_COLUMNS[2:]:
            row[column] = self.metrics.get(column)
        if self.notes:
            row["notes"] = self.notes
        return row


def comparison_table(results: list[DetectorResult]) -> pd.DataFrame:
    """Assemble detector results into one table sorted by mAP50-95."""
    if not results:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))

    frame = pd.DataFrame([result.as_row() for result in results])
    return frame.sort_values(
        ["split", "mAP50_95"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


def plot_metric_comparison(
    table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("mAP50", "mAP50_95"),
    split: str = "test",
    save_path: Path | None = None,
) -> None:
    """Grouped bar chart of the chosen metrics per model."""
    subset = table[table["split"] == split] if "split" in table.columns else table
    if subset.empty:
        return

    available = [metric for metric in metrics if metric in subset.columns]
    models = subset["model"].tolist()
    positions = np.arange(len(models))
    width = 0.8 / max(1, len(available))

    plt.figure(figsize=(1.8 * len(models) + 4, 4.5))
    for offset, metric in enumerate(available):
        values = subset[metric].fillna(0.0).to_numpy(dtype=float)
        bars = plt.bar(positions + offset * width, values, width, label=metric)
        for bar, value in zip(bars, values, strict=False):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )

    plt.xticks(positions + width * (len(available) - 1) / 2, models)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title(f"Detector comparison ({split} split)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    save_or_show(save_path)


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over union for two `[x_min, y_min, x_max, y_max]` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def classify_errors(
    ground_truth: list[dict[str, object]],
    predictions: list[dict[str, object]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    """Bucket detections against ground truth into an error taxonomy.

    Each prediction is greedily matched (highest confidence first) to the
    unmatched ground-truth box with the largest IoU. Buckets:

    - `correct`: matched with the right class
    - `misclassified`: well-localised box, wrong class
    - `false_positive`: nothing to match, or a duplicate of an already-matched box
    - `false_negative`: ground-truth box no prediction covered

    Args:
        ground_truth: Dicts with `xyxy` and `class_name`.
        predictions: Dicts with `xyxy`, `class_name` and optionally `confidence`.
        iou_threshold: Minimum IoU counted as the same object.
    """
    counts = {"correct": 0, "misclassified": 0, "false_positive": 0, "false_negative": 0}
    matched: set[int] = set()

    ordered = sorted(
        predictions,
        key=lambda p: float(p.get("confidence", 0.0)),  # type: ignore[arg-type]
        reverse=True,
    )

    for prediction in ordered:
        best_iou = 0.0
        best_index = -1

        for index, truth in enumerate(ground_truth):
            if index in matched:
                continue
            score = iou(prediction["xyxy"], truth["xyxy"])  # type: ignore[arg-type]
            if score > best_iou:
                best_iou = score
                best_index = index

        if best_index < 0 or best_iou < iou_threshold:
            # Either genuinely nothing there, or a second box on an object that
            # was already claimed - a duplicate detection
            counts["false_positive"] += 1
            continue

        matched.add(best_index)
        if prediction["class_name"] == ground_truth[best_index]["class_name"]:
            counts["correct"] += 1
        else:
            counts["misclassified"] += 1

    counts["false_negative"] = len(ground_truth) - len(matched)
    return counts


def error_summary(counts: dict[str, int]) -> pd.DataFrame:
    """Turn error counts into a table with shares of the total."""
    total = sum(counts.values())
    rows = [
        {
            "error_type": name,
            "count": count,
            "share": round(count / total, 4) if total else 0.0,
        }
        for name, count in counts.items()
    ]
    return pd.DataFrame(rows)


def plot_predictions(
    image_path: str | Path,
    predictions: list[dict[str, object]],
    *,
    ground_truth: list[dict[str, object]] | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    """Draw predicted boxes (and optionally ground truth) over an image."""
    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(7, 7))
    axis = plt.gca()
    axis.imshow(image)
    axis.axis("off")

    if ground_truth:
        for truth in ground_truth:
            x1, y1, x2, y2 = truth["xyxy"]  # type: ignore[misc]
            axis.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="#1f9d55",
                    facecolor="none",
                    linestyle="--",
                )
            )

    for prediction in predictions:
        x1, y1, x2, y2 = prediction["xyxy"]  # type: ignore[misc]
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="#e3342f",
                facecolor="none",
            )
        )
        label = str(prediction.get("class_name", ""))
        confidence = prediction.get("confidence")
        if confidence is not None:
            label = f"{label} {float(confidence):.2f}"
        axis.text(
            x1,
            max(0, y1 - 6),
            label,
            color="white",
            fontsize=9,
            bbox={"facecolor": "#e3342f", "pad": 1.5, "edgecolor": "none"},
        )

    axis.set_title(title or Path(image_path).name)
    save_or_show(save_path)
