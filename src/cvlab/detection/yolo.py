"""
YOLOv8 training, evaluation and inference via the Ultralytics API.

The experiment trains from a `.yaml` architecture definition rather than a `.pt`
checkpoint, so no ImageNet or COCO weights enter the run. That is the point of
the comparison: both detectors start from random initialisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..common.io import ensure_dir
from ..common.viz import save_or_show


def _load_ultralytics(model_spec: str) -> Any:
    """Import Ultralytics lazily and instantiate a model."""
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "ultralytics is required for detection. Install it with `pip install ultralytics`."
        ) from error
    return YOLO(model_spec)


def train_yolo(
    dataset_yaml: str | Path,
    *,
    model_spec: str = "yolov8n.yaml",
    epochs: int = 80,
    image_size: int = 640,
    batch: int = 16,
    project: str | Path = "runs/yolo",
    name: str = "yolov8n_scratch",
    seed: int = 42,
    device: str | int | None = None,
    patience: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a YOLOv8 detector and return paths to its artifacts.

    Passing a `.yaml` spec builds the architecture from scratch; a `.pt` spec
    would load pretrained weights instead.
    """
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml}")

    project = ensure_dir(project)
    model = _load_ultralytics(model_spec)

    model.train(
        data=str(dataset_yaml.resolve()),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=str(project),
        name=name,
        seed=seed,
        device=device,
        patience=patience,
        verbose=verbose,
        exist_ok=True,
    )

    run_dir = project / name
    return {
        "run_dir": run_dir,
        "best_weights": run_dir / "weights" / "best.pt",
        "last_weights": run_dir / "weights" / "last.pt",
        "results_csv": run_dir / "results.csv",
    }


def extract_yolo_metrics(metrics: Any) -> dict[str, float | None]:
    """Pull mAP/precision/recall off an Ultralytics metrics object.

    Attribute names have moved between Ultralytics releases, so each value is
    looked up through a list of candidates instead of one hard-coded path.
    """

    def first_attr(obj: Any, names: tuple[str, ...]) -> float | None:
        for name in names:
            target = obj
            try:
                for part in name.split("."):
                    target = getattr(target, part)
            except AttributeError:
                continue
            try:
                if target is not None:
                    return float(target)
            except (TypeError, ValueError):
                continue
        return None

    return {
        "mAP50_95": first_attr(metrics, ("box.map", "map")),
        "mAP50": first_attr(metrics, ("box.map50", "map50")),
        "mAP75": first_attr(metrics, ("box.map75", "map75")),
        "precision": first_attr(metrics, ("box.mp", "mp")),
        "recall": first_attr(metrics, ("box.mr", "mr")),
    }


def evaluate_yolo(
    weights: str | Path,
    dataset_yaml: str | Path,
    *,
    split: str = "test",
    image_size: int = 640,
    batch: int = 16,
    project: str | Path = "runs/yolo",
    name: str | None = None,
    device: str | int | None = None,
) -> dict[str, float | None]:
    """Run validation on one split and return its metrics."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = _load_ultralytics(str(weights))
    metrics = model.val(
        data=str(Path(dataset_yaml).resolve()),
        split=split,
        imgsz=image_size,
        batch=batch,
        project=str(project),
        name=name or f"val_{split}",
        device=device,
        exist_ok=True,
    )

    result = extract_yolo_metrics(metrics)
    result["split"] = split  # type: ignore[assignment]
    return result


def predict_yolo(
    weights: str | Path,
    source: str | Path,
    *,
    conf: float = 0.40,
    image_size: int = 640,
    project: str | Path = "runs/yolo",
    name: str = "predict",
    save: bool = True,
    device: str | int | None = None,
) -> list[dict[str, Any]]:
    """Run inference on an image or directory and return per-image detections."""
    model = _load_ultralytics(str(weights))
    predictions = model.predict(
        source=str(source),
        conf=conf,
        imgsz=image_size,
        project=str(project),
        name=name,
        save=save,
        device=device,
        exist_ok=True,
        verbose=False,
    )

    detections = []
    for prediction in predictions:
        boxes = []
        names = prediction.names
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(value) for value in box.xyxy.flatten().tolist()],
                }
            )
        detections.append({"image_path": prediction.path, "boxes": boxes})

    return detections


def read_training_curves(results_csv: str | Path) -> pd.DataFrame:
    """Load `results.csv` from a run, normalising column names.

    Ultralytics pads its CSV headers with spaces, which turns naive column access
    into a debugging session.
    """
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def plot_training_curves(
    results_csv: str | Path,
    *,
    save_path: Path | None = None,
) -> None:
    """Plot losses and validation mAP over epochs."""
    frame = read_training_curves(results_csv)

    loss_columns = [c for c in frame.columns if "loss" in c and c.startswith("train")]
    map_columns = [c for c in frame.columns if "mAP" in c]
    epochs = frame["epoch"] if "epoch" in frame.columns else range(len(frame))

    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 2, 1)
    for column in loss_columns:
        plt.plot(epochs, frame[column], label=column.replace("train/", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLOv8 training losses")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    for column in map_columns:
        plt.plot(epochs, frame[column], label=column.split("/")[-1])
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("YOLOv8 validation mAP")
    plt.grid(alpha=0.3)
    plt.legend()

    save_or_show(save_path)
