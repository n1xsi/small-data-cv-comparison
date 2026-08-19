#!/usr/bin/env python
"""
Run a trained model on new images.

Examples:
    python scripts/predict.py --task classify --weights results/classification/weights/mobilenet.keras --source data/new_images
    python scripts/predict.py --task detect --weights runs/yolo/yolov8n_scratch/weights/best.pt --source data/new_images
    python scripts/predict.py --task detect --weights best.pt --source cat.jpg --conf 0.25 --save-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cvlab.common import (  # noqa: E402
    ensure_dir,
    list_images,
    load_classification_config,
    save_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task", choices=("classify", "detect"), required=True, help="Which model to run."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained weights.")
    parser.add_argument(
        "--source", type=Path, required=True, help="Image file or directory of images."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/predictions"),
        help="Where to write output.",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold.")
    parser.add_argument(
        "--image-size", type=int, default=None, help="Override the inference image size."
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Class names in label order (classification).",
    )
    parser.add_argument("--save-figures", action="store_true", help="Save annotated images.")
    return parser.parse_args()


def collect_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    return images


def run_classification(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    from tensorflow import keras

    from cvlab.classification.explain import load_image_array

    config = load_classification_config()
    class_names = args.class_names or config.class_names or ["class_0", "class_1"]
    image_size = (args.image_size, args.image_size) if args.image_size else config.dl_image_size

    print(f"Loading {args.weights} ...")
    model = keras.models.load_model(args.weights)

    rows = []
    for path in collect_sources(args.source):
        batch = load_image_array(path, image_size=image_size)
        probabilities = model.predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))

        name = class_names[index] if index < len(class_names) else str(index)
        confidence = float(probabilities[index])
        rows.append(
            {
                "image": path.name,
                "predicted_class": name,
                "confidence": round(confidence, 4),
                **{
                    f"p_{class_names[i] if i < len(class_names) else i}": round(float(p), 4)
                    for i, p in enumerate(probabilities)
                },
            }
        )
        print(f"  {path.name}: {name} ({confidence:.1%})")

    return pd.DataFrame(rows)


def run_detection(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    from cvlab.detection import predict_yolo

    print(f"Loading {args.weights} ...")
    detections = predict_yolo(
        args.weights,
        args.source,
        conf=args.conf,
        image_size=args.image_size or 640,
        project=output_dir,
        name="detect",
        save=args.save_figures,
    )

    rows = []
    for detection in detections:
        image_name = Path(str(detection["image_path"])).name
        boxes = detection["boxes"]

        if not boxes:
            print(f"  {image_name}: nothing detected above conf={args.conf}")
            rows.append(
                {
                    "image": image_name,
                    "class_name": None,
                    "confidence": None,
                    "x_min": None,
                    "y_min": None,
                    "x_max": None,
                    "y_max": None,
                }
            )
            continue

        summary = ", ".join(f"{b['class_name']} {b['confidence']:.2f}" for b in boxes)
        print(f"  {image_name}: {summary}")

        for box in boxes:
            x_min, y_min, x_max, y_max = box["xyxy"]
            rows.append(
                {
                    "image": image_name,
                    "class_name": box["class_name"],
                    "confidence": round(box["confidence"], 4),
                    "x_min": round(x_min, 1),
                    "y_min": round(y_min, 1),
                    "x_max": round(x_max, 1),
                    "y_max": round(y_max, 1),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    output_dir = ensure_dir(args.output_dir)

    if args.task == "classify":
        table = run_classification(args, output_dir)
    else:
        table = run_detection(args, output_dir)

    predictions_path = output_dir / f"{args.task}_predictions.csv"
    save_table(table, predictions_path)
    print(f"\nWrote {len(table)} rows to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
