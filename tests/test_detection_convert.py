"""Tests for COCO auditing and the COCO to YOLO conversion.

The bbox conversion gets the most attention here. A wrong formula produces a
dataset that trains without error and scores near zero, which is a far more
expensive bug to find later than a failing test now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cvlab.detection import (
    audit_coco,
    audit_yolo_dataset,
    build_yolo_dataset,
    coco_bbox_to_yolo,
    dominant_class_per_image,
    export_split_to_coco,
    read_yolo_labels,
    sanitize_bbox,
    split_detection_dataset,
    summarize_splits,
    validate_exported_coco,
)

# --------------------------------------------------------------------------- #
# bbox geometry
# --------------------------------------------------------------------------- #


def test_bbox_conversion_centre_and_size() -> None:
    # A 40x20 box at (10, 10) in a 100x80 image.
    cx, cy, w, h = coco_bbox_to_yolo([10, 10, 40, 20], 100, 80)

    assert cx == pytest.approx(30 / 100)  # 10 + 40/2
    assert cy == pytest.approx(20 / 80)  # 10 + 20/2
    assert w == pytest.approx(40 / 100)
    assert h == pytest.approx(20 / 80)


def test_bbox_covering_whole_image() -> None:
    cx, cy, w, h = coco_bbox_to_yolo([0, 0, 100, 80], 100, 80)

    assert (cx, cy, w, h) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_bbox_conversion_stays_normalised() -> None:
    cx, cy, w, h = coco_bbox_to_yolo([0, 0, 100.0001, 80.0001], 100, 80)

    assert all(0.0 <= value <= 1.0 for value in (cx, cy, w, h))


def test_bbox_conversion_rejects_degenerate_image_size() -> None:
    with pytest.raises(ValueError, match="Invalid image size"):
        coco_bbox_to_yolo([0, 0, 10, 10], 0, 80)


def test_sanitize_clips_overhanging_box() -> None:
    # Box runs from x=50 to x=80 in a 60px-wide image.
    box, was_clipped = sanitize_bbox([50, 10, 30, 20], 60, 60)

    assert was_clipped
    assert box == [50.0, 10.0, 10.0, 20.0]


def test_sanitize_keeps_valid_box_untouched() -> None:
    box, was_clipped = sanitize_bbox([10, 10, 30, 20], 100, 80)

    assert not was_clipped
    assert box == [10.0, 10.0, 30.0, 20.0]


def test_sanitize_drops_zero_area_box() -> None:
    box, _ = sanitize_bbox([10, 10, 0, 20], 100, 80)

    assert box is None


def test_sanitize_drops_fully_outside_box() -> None:
    box, _ = sanitize_bbox([200, 200, 30, 20], 100, 80)

    assert box is None


def test_sanitize_drops_non_finite_box() -> None:
    box, _ = sanitize_bbox([float("nan"), 10, 30, 20], 100, 80)

    assert box is None


# --------------------------------------------------------------------------- #
# COCO audit
# --------------------------------------------------------------------------- #


def test_audit_counts_and_problem_reporting(coco_dataset: tuple[Path, Path]) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)

    assert audit.n_images == 4
    # 5 annotations, minus the zero-width one that must be dropped.
    assert audit.n_annotations == 4
    assert len(audit.invalid_boxes) == 1
    assert len(audit.out_of_bounds) == 1
    assert audit.missing_images == []


def test_audit_class_counts(coco_dataset: tuple[Path, Path]) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)

    assert audit.objects_per_class() == {"cat": 2, "dog": 2}
    # Both dog boxes live in the same image.
    assert audit.images_per_class() == {"cat": 2, "dog": 1}


def test_audit_summary_mentions_key_numbers(coco_dataset: tuple[Path, Path]) -> None:
    json_path, images_root = coco_dataset
    summary = audit_coco(json_path, images_root).summary()

    assert "Images: 4" in summary
    assert "Annotations: 4" in summary


def test_dominant_class_uses_the_majority_class(coco_dataset: tuple[Path, Path]) -> None:
    json_path, images_root = coco_dataset
    dominant = dominant_class_per_image(audit_coco(json_path, images_root))

    assert dominant[1] == "cat"
    assert dominant[2] == "dog"  # two dog boxes, no cats


# --------------------------------------------------------------------------- #
# Split export
# --------------------------------------------------------------------------- #


def test_split_covers_every_image_exactly_once(coco_dataset: tuple[Path, Path]) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)

    splits = split_detection_dataset(audit, val_size=0.25, test_size=0.25, seed=42)
    all_ids = splits.train + splits.val + splits.test

    assert sorted(all_ids) == sorted(audit.images["image_id"].tolist())
    assert len(all_ids) == len(set(all_ids))


def test_summarize_splits_totals_match(coco_dataset: tuple[Path, Path]) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)
    splits = split_detection_dataset(audit, val_size=0.25, test_size=0.25, seed=42)

    summary = summarize_splits(audit, splits)

    assert summary["images"].sum() == audit.n_images
    assert summary["boxes"].sum() == audit.n_annotations


def test_export_remaps_categories_to_contiguous_range(
    coco_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)

    exported = export_split_to_coco(
        audit, audit.images["image_id"].tolist(), tmp_path / "out.json", remap_categories=True
    )

    category_ids = sorted(c["id"] for c in exported["categories"])
    assert category_ids == [1, 2]
    assert validate_exported_coco(exported, split_name="all") == []


def test_export_flags_annotations_pointing_at_missing_images() -> None:
    broken = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 99, "category_id": 1, "bbox": [0, 0, 5, 5]}],
        "categories": [{"id": 1, "name": "cat"}],
    }

    problems = validate_exported_coco(broken, split_name="train")

    assert any("missing image" in problem for problem in problems)


# --------------------------------------------------------------------------- #
# YOLO layout
# --------------------------------------------------------------------------- #


def test_yolo_dataset_layout_and_yaml(coco_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)
    splits = split_detection_dataset(audit, val_size=0.25, test_size=0.25, seed=42)

    output_root = tmp_path / "yolo"
    dataset_yaml = build_yolo_dataset(audit, splits, output_root)

    assert dataset_yaml.exists()

    import yaml

    spec = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert spec["nc"] == 2
    assert spec["names"] == {0: "cat", 1: "dog"}

    for split_name in ("train", "val", "test"):
        assert (output_root / "images" / split_name).is_dir()
        assert (output_root / "labels" / split_name).is_dir()


def test_yolo_audit_finds_no_problems(coco_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)
    splits = split_detection_dataset(audit, val_size=0.25, test_size=0.25, seed=42)

    output_root = tmp_path / "yolo"
    build_yolo_dataset(audit, splits, output_root)

    report = audit_yolo_dataset(output_root)

    assert report["missing_label_files"].sum() == 0
    assert report["out_of_range_boxes"].sum() == 0
    # Every surviving box must appear on disk.
    assert report["boxes"].sum() == audit.n_annotations
    assert report["images"].sum() == audit.n_images


def test_every_image_gets_a_label_file_even_when_empty(
    coco_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)
    splits = split_detection_dataset(audit, val_size=0.25, test_size=0.25, seed=42)

    output_root = tmp_path / "yolo"
    build_yolo_dataset(audit, splits, output_root)

    for split_name in ("train", "val", "test"):
        for image_path in (output_root / "images" / split_name).iterdir():
            label_path = output_root / "labels" / split_name / f"{image_path.stem}.txt"
            assert label_path.exists(), f"missing label for {image_path.name}"

    # Image 4's only box had zero area, so its label file must exist but be empty.
    empty_labels = [
        path
        for split_name in ("train", "val", "test")
        for path in (output_root / "labels" / split_name).iterdir()
        if path.stem == "img_4"
    ]
    assert len(empty_labels) == 1
    assert read_yolo_labels(empty_labels[0]) == []


def test_written_labels_round_trip_to_the_source_box(
    coco_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    """Reading a written label back must reproduce the original COCO box."""
    json_path, images_root = coco_dataset
    audit = audit_coco(json_path, images_root)
    splits = split_detection_dataset(audit, val_size=0.25, test_size=0.25, seed=42)

    output_root = tmp_path / "yolo"
    build_yolo_dataset(audit, splits, output_root)

    source = audit.annotations[audit.annotations["image_id"] == 1].iloc[0]
    info = audit.images[audit.images["image_id"] == 1].iloc[0]

    written = [
        path
        for split_name in ("train", "val", "test")
        for path in (output_root / "labels" / split_name).iterdir()
        if path.stem == "img_1"
    ]
    assert len(written) == 1

    (class_id, cx, cy, w, h) = read_yolo_labels(written[0])[0]
    assert class_id == 0  # "cat" is index 0

    # Convert back to absolute pixel corners.
    width, height = float(info["width"]), float(info["height"])
    x_min = (cx - w / 2) * width
    y_min = (cy - h / 2) * height

    assert x_min == pytest.approx(source["bbox_x"], abs=0.05)
    assert y_min == pytest.approx(source["bbox_y"], abs=0.05)
    assert w * width == pytest.approx(source["bbox_w"], abs=0.05)
    assert h * height == pytest.approx(source["bbox_h"], abs=0.05)
