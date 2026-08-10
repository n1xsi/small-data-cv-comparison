"""Tests for dataset auditing, splitting and feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvlab.classification import (
    build_dataset_dataframe,
    extract_hog_features,
    flatten_pixels,
    load_images_as_arrays,
    split_dataset,
)


def test_audit_finds_classes_and_rejects_corrupt_files(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)

    assert audit.class_names == ["cat", "dog"]
    assert audit.class_counts == {"cat": 12, "dog": 8}
    assert len(audit.broken_files) == 1
    assert "broken.png" in audit.broken_files[0]


def test_audit_respects_per_class_cap(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, max_images_per_class=5, show_progress=False)

    assert all(count <= 5 for count in audit.class_counts.values())


def test_audit_cap_is_reproducible(image_dataset: Path) -> None:
    first = build_dataset_dataframe(
        image_dataset, max_images_per_class=5, seed=7, show_progress=False
    )
    second = build_dataset_dataframe(
        image_dataset, max_images_per_class=5, seed=7, show_progress=False
    )

    assert first.frame["file_path"].tolist() == second.frame["file_path"].tolist()


def test_audit_requires_at_least_two_classes(tmp_path: Path) -> None:
    (tmp_path / "only_one").mkdir()

    with pytest.raises(ValueError, match="at least 2 class"):
        build_dataset_dataframe(tmp_path, show_progress=False)


def test_audit_reports_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_dataset_dataframe(tmp_path / "nope", show_progress=False)


def test_split_partitions_without_overlap(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)
    splits = split_dataset(audit, val_size=0.2, test_size=0.2, seed=42)

    sizes = splits.sizes()
    assert sum(sizes.values()) == len(audit.frame)

    train_files = set(splits.train["file_path"])
    val_files = set(splits.val["file_path"])
    test_files = set(splits.test["file_path"])

    assert not train_files & val_files
    assert not train_files & test_files
    assert not val_files & test_files


def test_split_keeps_both_classes_in_every_split(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)
    splits = split_dataset(audit, val_size=0.2, test_size=0.2, seed=42)

    distribution = splits.distribution()
    assert (distribution > 0).all().all()


def test_split_is_deterministic(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)

    first = split_dataset(audit, seed=42)
    second = split_dataset(audit, seed=42)

    assert first.test["file_path"].tolist() == second.test["file_path"].tolist()


def test_split_rejects_impossible_fractions(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)

    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        split_dataset(audit, val_size=0.6, test_size=0.6)


def test_labels_align_with_class_names(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)

    for _, row in audit.frame.iterrows():
        assert audit.class_names[int(row["label"])] == row["class_name"]


def test_load_images_shape_and_range(image_dataset: Path) -> None:
    audit = build_dataset_dataframe(image_dataset, show_progress=False)

    grayscale = load_images_as_arrays(audit.frame, image_size=(32, 32), show_progress=False)
    assert grayscale.shape == (len(audit.frame), 32, 32)
    assert grayscale.dtype == np.float32
    assert grayscale.min() >= 0.0 and grayscale.max() <= 1.0

    rgb = load_images_as_arrays(
        audit.frame, image_size=(32, 32), grayscale=False, show_progress=False
    )
    assert rgb.shape == (len(audit.frame), 32, 32, 3)


def test_flatten_pixels_dimensionality() -> None:
    images = np.zeros((4, 64, 64), dtype=np.float32)

    assert flatten_pixels(images).shape == (4, 64 * 64)


def test_hog_features_are_finite_and_consistent() -> None:
    rng = np.random.default_rng(1)
    images = rng.random((3, 64, 64)).astype(np.float32)

    features = extract_hog_features(images, show_progress=False)

    assert features.shape[0] == 3
    assert features.shape[1] > 0
    assert np.isfinite(features).all()

    # Same input must give the same descriptor: HOG has no random component
    repeat = extract_hog_features(images, show_progress=False)
    np.testing.assert_allclose(features, repeat)
