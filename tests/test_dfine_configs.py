"""
Tests for D-FINE config generation and log parsing.

The configs are plain text handed to an external trainer, so nothing here is
validated by an import or a type checker. A config that points at the wrong
split trains and evaluates without complaint and reports a number for the wrong
data — which is exactly what happened before these tests existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cvlab.detection import parse_coco_metrics, write_dfine_configs


@pytest.fixture
def configs(tmp_path: Path):
    return write_dfine_configs(
        tmp_path / "configs",
        coco_root=tmp_path / "coco",
        num_classes=3,
        epochs=20,
    )


# --------------------------------------------------------------------------- #
# Split wiring
# --------------------------------------------------------------------------- #


def test_dataset_config_points_each_loader_at_its_own_split(configs) -> None:
    text = configs.dataset.read_text(encoding="utf-8")

    assert "img_folder: " in text
    assert "images/train" in text
    assert "annotations/train.json" in text
    assert "images/val" in text
    assert "annotations/val.json" in text


def test_test_config_evaluates_the_test_split(configs) -> None:
    """The whole point of the separate config: `--test-only` must read `test`."""
    text = configs.test.read_text(encoding="utf-8")

    assert "img_folder: " in text, "test config must restate img_folder, not inherit it"
    assert "images/test" in text
    assert "annotations/test.json" in text
    assert "images/val" not in text
    assert "annotations/val.json" not in text


def test_train_config_evaluates_the_val_split(configs) -> None:
    text = configs.train.read_text(encoding="utf-8")

    assert "images/val" in text
    assert "annotations/val.json" in text
    assert "images/test" not in text


def test_train_and_test_configs_differ(configs) -> None:
    """Regression: a no-op string replace once made these byte-identical."""
    assert configs.train.read_text(encoding="utf-8") != configs.test.read_text(encoding="utf-8")


def test_configs_differ_only_in_the_eval_split(configs) -> None:
    train_lines = configs.train.read_text(encoding="utf-8").splitlines()
    test_lines = configs.test.read_text(encoding="utf-8").splitlines()

    assert len(train_lines) == len(test_lines)
    differing = [(a, b) for a, b in zip(train_lines, test_lines, strict=True) if a != b]

    assert len(differing) == 2, f"unexpected differences: {differing}"
    assert all("val" in a and "test" in b for a, b in differing)


# --------------------------------------------------------------------------- #
# Model settings that silently produce garbage when wrong
# --------------------------------------------------------------------------- #


def test_backbone_is_not_pretrained_by_default(configs) -> None:
    """The comparison is only meaningful from random initialisation."""
    assert "pretrained: false" in configs.train.read_text(encoding="utf-8")


def test_num_classes_is_written_verbatim(tmp_path: Path) -> None:
    """D-FINE reserves index 0 for background, so callers pass n_classes + 1."""
    configs = write_dfine_configs(tmp_path / "configs", coco_root=tmp_path / "coco", num_classes=3)

    assert "num_classes: 3" in configs.dataset.read_text(encoding="utf-8")


def test_stop_epoch_precedes_the_final_epoch(tmp_path: Path) -> None:
    """Augmentation is switched off for the last 10 epochs."""
    configs = write_dfine_configs(
        tmp_path / "configs", coco_root=tmp_path / "coco", num_classes=3, epochs=80
    )
    text = configs.train.read_text(encoding="utf-8")

    assert "epochs: 80" in text
    assert "stop_epoch: 70" in text


def test_stop_epoch_stays_positive_on_very_short_runs(tmp_path: Path) -> None:
    configs = write_dfine_configs(
        tmp_path / "configs", coco_root=tmp_path / "coco", num_classes=3, epochs=2
    )

    assert "stop_epoch: 1" in configs.train.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Metric parsing
# --------------------------------------------------------------------------- #


def test_parse_coco_metrics_reads_the_verbose_block(tmp_path: Path) -> None:
    log = tmp_path / "dfine.log"
    log.write_text(
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.114\n"
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.251\n"
        " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.076\n"
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606\n",
        encoding="utf-8",
    )

    metrics = parse_coco_metrics(log)

    assert metrics["mAP50_95"] == pytest.approx(0.114)
    assert metrics["mAP50"] == pytest.approx(0.251)
    assert metrics["mAP75"] == pytest.approx(0.076)
    assert metrics["AR100"] == pytest.approx(0.606)


def test_parse_coco_metrics_selects_the_requested_block(tmp_path: Path) -> None:
    """Training logs hold one block per epoch; `which` picks between them."""
    block = (
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {best}\n"
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.400\n"
    )
    log = tmp_path / "dfine.log"
    log.write_text(block.format(best="0.300") + block.format(best="0.200"), encoding="utf-8")

    assert parse_coco_metrics(log, which="last")["mAP50_95"] == pytest.approx(0.200)
    assert parse_coco_metrics(log, which="best")["mAP50_95"] == pytest.approx(0.300)


def test_parse_coco_metrics_of_a_missing_log_is_empty_not_an_error(tmp_path: Path) -> None:
    assert parse_coco_metrics(tmp_path / "never-ran.log") == {}
