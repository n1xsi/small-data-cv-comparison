"""Tests for IoU, the error taxonomy, config loading and seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvlab.common import (
    ClassificationConfig,
    DetectionConfig,
    load_classification_config,
    load_detection_config,
    new_rng,
    set_seeds,
)
from cvlab.detection import DetectorResult, classify_errors, comparison_table, error_summary, iou

# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #


def test_iou_identical_boxes() -> None:
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_iou_half_overlap() -> None:
    # Intersection 5x10 = 50; union = 100 + 100 - 50 = 150.
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_nested_box() -> None:
    # Inner 5x5 = 25 fully inside outer 10x10 = 100.
    assert iou([0, 0, 10, 10], [2, 2, 7, 7]) == pytest.approx(25 / 100)


def test_iou_is_symmetric() -> None:
    a, b = [0, 0, 10, 10], [3, 4, 12, 14]
    assert iou(a, b) == pytest.approx(iou(b, a))


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #


def test_perfect_prediction() -> None:
    truth = [{"xyxy": [0, 0, 10, 10], "class_name": "cat"}]
    predicted = [{"xyxy": [0, 0, 10, 10], "class_name": "cat", "confidence": 0.9}]

    counts = classify_errors(truth, predicted)

    assert counts == {"correct": 1, "misclassified": 0, "false_positive": 0, "false_negative": 0}


def test_misclassification_is_not_counted_as_a_miss() -> None:
    """A well-localised box with the wrong label is its own error type."""
    truth = [{"xyxy": [0, 0, 10, 10], "class_name": "cat"}]
    predicted = [{"xyxy": [0, 0, 10, 10], "class_name": "dog", "confidence": 0.8}]

    counts = classify_errors(truth, predicted)

    assert counts["misclassified"] == 1
    assert counts["false_negative"] == 0
    assert counts["correct"] == 0


def test_missed_object_is_a_false_negative() -> None:
    truth = [{"xyxy": [0, 0, 10, 10], "class_name": "cat"}]

    counts = classify_errors(truth, [])

    assert counts["false_negative"] == 1
    assert counts["false_positive"] == 0


def test_spurious_box_is_a_false_positive() -> None:
    predicted = [{"xyxy": [50, 50, 60, 60], "class_name": "dog", "confidence": 0.7}]

    counts = classify_errors([], predicted)

    assert counts["false_positive"] == 1


def test_duplicate_detection_counts_once_plus_a_false_positive() -> None:
    """Two boxes on one object: the best matches, the second is a false positive."""
    truth = [{"xyxy": [0, 0, 10, 10], "class_name": "cat"}]
    predicted = [
        {"xyxy": [0, 0, 10, 10], "class_name": "cat", "confidence": 0.9},
        {"xyxy": [1, 1, 11, 11], "class_name": "dog", "confidence": 0.6},
    ]

    counts = classify_errors(truth, predicted)

    assert counts["correct"] == 1
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 0


def test_higher_confidence_prediction_claims_the_object() -> None:
    """Matching is greedy by confidence, so the confident box wins the match."""
    truth = [{"xyxy": [0, 0, 10, 10], "class_name": "cat"}]
    predicted = [
        {"xyxy": [2, 2, 12, 12], "class_name": "dog", "confidence": 0.4},
        {"xyxy": [0, 0, 10, 10], "class_name": "cat", "confidence": 0.95},
    ]

    counts = classify_errors(truth, predicted)

    assert counts["correct"] == 1
    assert counts["misclassified"] == 0
    assert counts["false_positive"] == 1


def test_poorly_localised_box_fails_the_iou_threshold() -> None:
    truth = [{"xyxy": [0, 0, 10, 10], "class_name": "cat"}]
    predicted = [{"xyxy": [8, 8, 18, 18], "class_name": "cat", "confidence": 0.9}]

    counts = classify_errors(truth, predicted, iou_threshold=0.5)

    assert counts["correct"] == 0
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 1


def test_error_summary_shares_sum_to_one() -> None:
    counts = {"correct": 6, "misclassified": 2, "false_positive": 1, "false_negative": 1}

    summary = error_summary(counts)

    assert summary["count"].sum() == 10
    assert summary["share"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #


def test_comparison_table_sorts_best_first() -> None:
    results = [
        DetectorResult(model="weak", split="test", metrics={"mAP50_95": 0.11, "mAP50": 0.25}),
        DetectorResult(model="strong", split="test", metrics={"mAP50_95": 0.26, "mAP50": 0.48}),
    ]

    table = comparison_table(results)

    assert table.iloc[0]["model"] == "strong"


def test_comparison_table_handles_missing_metrics() -> None:
    results = [DetectorResult(model="partial", split="test", metrics={"mAP50": 0.3})]

    table = comparison_table(results)

    assert table.iloc[0]["mAP50"] == pytest.approx(0.3)
    assert table.iloc[0]["precision"] is None or np.isnan(table.iloc[0]["precision"])


def test_comparison_table_of_nothing_is_empty_not_an_error() -> None:
    assert comparison_table([]).empty


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_repo_configs_load(tmp_path: Path) -> None:
    """The checked-in YAML files must parse into valid config objects."""
    classification = load_classification_config()
    detection = load_detection_config()

    assert classification.seed == 42
    assert classification.primary_metric == "f1_macro"
    # A .yaml spec means training from scratch; a .pt would mean pretrained weights.
    assert detection.yolo_model.endswith(".yaml")
    assert detection.dfine_backbone_pretrained is False


def test_config_paths_become_path_objects() -> None:
    config = ClassificationConfig(dataset_root="data/somewhere")

    assert isinstance(config.dataset_root, Path)


def test_config_rejects_splits_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        ClassificationConfig(train_size=0.8, val_size=0.15, test_size=0.15)

    with pytest.raises(ValueError, match="sum to 1.0"):
        DetectionConfig(train_size=0.5, val_size=0.2, test_size=0.2)


def test_config_ignores_unknown_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "classification.yaml"
    path.write_text("seed: 7\nsome_future_option: 123\n", encoding="utf-8")

    config = load_classification_config(path)

    assert config.seed == 7


def test_missing_config_falls_back_to_defaults(tmp_path: Path) -> None:
    config = load_classification_config(tmp_path / "absent.yaml")

    assert config.seed == 42


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def test_set_seeds_returns_the_seed_and_makes_numpy_reproducible() -> None:
    assert set_seeds(123, tensorflow=False) == 123
    first = np.random.rand(5)

    set_seeds(123, tensorflow=False)
    np.testing.assert_allclose(first, np.random.rand(5))


def test_new_rng_is_reproducible() -> None:
    np.testing.assert_allclose(new_rng(5).random(4), new_rng(5).random(4))
