"""
COCO to YOLO conversion, plus split construction and verification.

COCO stores absolute pixel boxes as `[x_min, y_min, width, height]`; YOLO wants
one text file per image with `class_id cx cy w h`, all normalised to [0, 1] and
measured from the box centre. Getting this wrong is the classic silent failure in
detection work: training runs fine and mAP stays near zero, so every converted
split is audited by reading the labels back off disk.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ..common.io import ensure_dir
from .coco import CocoAudit, dominant_class_per_image

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class DetectionSplits:
    """Image ids assigned to each split."""

    train: list[int]
    val: list[int]
    test: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def sizes(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self.as_dict().items()}


def split_detection_dataset(
    audit: CocoAudit,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DetectionSplits:
    """Split images, stratified by dominant class where possible.

    Stratification is skipped automatically when any class is too rare to appear
    in every split — `train_test_split` would otherwise raise.
    """
    image_ids = audit.images["image_id"].astype(int).tolist()
    dominant = dominant_class_per_image(audit)
    labels = [dominant.get(image_id, "unknown") for image_id in image_ids]

    def usable(strata: list[str], n_splits: int = 2) -> bool:
        counts = pd.Series(strata).value_counts()
        return len(counts) > 1 and counts.min() >= n_splits

    holdout_size = val_size + test_size
    stratify = labels if usable(labels) else None

    train_ids, holdout_ids, _, holdout_labels = train_test_split(
        image_ids,
        labels,
        test_size=holdout_size,
        stratify=stratify,
        random_state=seed,
    )

    relative_test_size = test_size / holdout_size
    inner_stratify = holdout_labels if usable(holdout_labels) else None

    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=relative_test_size,
        stratify=inner_stratify,
        random_state=seed,
    )

    return DetectionSplits(
        train=sorted(int(i) for i in train_ids),
        val=sorted(int(i) for i in val_ids),
        test=sorted(int(i) for i in test_ids),
    )


def summarize_splits(audit: CocoAudit, splits: DetectionSplits) -> pd.DataFrame:
    """Per-split image counts, box counts per class, and objects per image."""
    rows = []

    for split_name, image_ids in splits.as_dict().items():
        wanted = set(image_ids)
        annotations = (
            audit.annotations[audit.annotations["image_id"].isin(wanted)]
            if not audit.annotations.empty
            else audit.annotations
        )

        row: dict[str, object] = {"split": split_name, "images": len(image_ids)}
        row["boxes"] = int(len(annotations))
        for class_name in sorted(audit.categories.values()):
            row[f"boxes_{class_name}"] = int(
                (annotations["category_name"] == class_name).sum() if not annotations.empty else 0
            )
        row["boxes_per_image"] = round(len(annotations) / max(1, len(image_ids)), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """Convert `[x_min, y_min, w, h]` in pixels to normalised `(cx, cy, w, h)`."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")

    x_min, y_min, width, height = (float(value) for value in bbox)

    center_x = (x_min + width / 2.0) / image_width
    center_y = (y_min + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height

    # Guard against floating-point drift pushing a value just outside [0, 1].
    clamp = lambda value: min(1.0, max(0.0, value))  # noqa: E731
    return clamp(center_x), clamp(center_y), clamp(norm_width), clamp(norm_height)


def build_yolo_dataset(
    audit: CocoAudit,
    splits: DetectionSplits,
    output_root: str | Path,
    *,
    copy_images: bool = True,
) -> Path:
    """Materialise the YOLO directory layout and write `dataset.yaml`.

    Produces the structure Ultralytics expects:

        output_root/
          images/{train,val,test}/*.jpg
          labels/{train,val,test}/*.txt
          dataset.yaml
    """
    output_root = Path(output_root)
    class_names = [audit.categories[key] for key in sorted(audit.categories)]
    class_index = {name: idx for idx, name in enumerate(class_names)}

    by_image = (
        {int(k): v for k, v in audit.annotations.groupby("image_id")}
        if not audit.annotations.empty
        else {}
    )
    image_info = audit.images.set_index("image_id").to_dict("index")

    for split_name, image_ids in splits.as_dict().items():
        image_dir = ensure_dir(output_root / "images" / split_name)
        label_dir = ensure_dir(output_root / "labels" / split_name)

        for image_id in image_ids:
            info = image_info.get(int(image_id))
            if not info:
                continue

            source = Path(info["file_path"])
            if not source.exists():
                continue

            destination = image_dir / source.name
            if copy_images and not destination.exists():
                shutil.copy2(source, destination)

            lines = []
            for row in by_image.get(int(image_id), pd.DataFrame()).itertuples():
                cx, cy, width, height = coco_bbox_to_yolo(
                    [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h],
                    float(info["width"]),
                    float(info["height"]),
                )
                lines.append(
                    f"{class_index[row.category_name]} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
                )

            # An image with no boxes still needs an empty label file: YOLO treats
            # it as a valid background example rather than missing data
            (label_dir / f"{source.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {idx: name for idx, name in enumerate(class_names)},
                "nc": len(class_names),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return dataset_yaml


def read_yolo_labels(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into `(class_id, cx, cy, w, h)` tuples."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, *coords = parts
        boxes.append((int(float(class_id)), *(float(value) for value in coords)))

    return boxes


def audit_yolo_dataset(output_root: str | Path) -> pd.DataFrame:
    """Read the converted labels back and report what is actually on disk.

    Verifies image/label pairing and that every coordinate lies in [0, 1] — the
    two failure modes that produce a silently untrainable dataset.
    """
    output_root = Path(output_root)
    rows = []

    for split_name in SPLIT_NAMES:
        image_dir = output_root / "images" / split_name
        label_dir = output_root / "labels" / split_name
        if not image_dir.exists():
            continue

        images = sorted(p for p in image_dir.iterdir() if p.is_file())
        n_boxes = 0
        missing_labels = 0
        out_of_range = 0
        class_counts: dict[int, int] = {}

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels += 1
                continue

            for class_id, cx, cy, width, height in read_yolo_labels(label_path):
                n_boxes += 1
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
                    out_of_range += 1
                if width <= 0 or height <= 0:
                    out_of_range += 1

        rows.append(
            {
                "split": split_name,
                "images": len(images),
                "boxes": n_boxes,
                "missing_label_files": missing_labels,
                "out_of_range_boxes": out_of_range,
                "class_counts": dict(sorted(class_counts.items())),
            }
        )

    return pd.DataFrame(rows)
