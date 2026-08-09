"""
Typed configuration objects loaded from YAML.

The original notebooks kept configuration in Colab form fields with absolute
`/content/...` paths. Here the same knobs live in `configs/*.yaml`, so a run can
be reproduced on a laptop, a workstation or a hosted GPU without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"

ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _split_sums_to_one(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")


@dataclass
class ClassificationConfig:
    """Settings for the classification experiments (part 1)."""

    dataset_root: Path = Path("data/raw/cats_dogs")
    output_dir: Path = Path("results/classification")
    class_names: list[str] | None = None
    max_images_per_class: int | None = 1000

    classical_image_size: tuple[int, int] = (64, 64)
    dl_image_size: tuple[int, int] = (160, 160)

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    seed: int = 42
    batch_size: int = 32
    cnn_epochs: int = 15
    tl_head_epochs: int = 5
    tl_fine_tune_epochs: int = 5
    fine_tune_at: int = 100

    use_class_weights: bool = True
    primary_metric: str = "f1_macro"

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_dir = Path(self.output_dir)
        self.classical_image_size = tuple(self.classical_image_size)  # type: ignore[assignment]
        self.dl_image_size = tuple(self.dl_image_size)  # type: ignore[assignment]
        _split_sums_to_one(self.train_size, self.val_size, self.test_size)


@dataclass
class DetectionConfig:
    """Settings for the detection experiments (part 2)."""

    coco_json: Path = Path("data/raw/annotations/result.json")
    images_root: Path = Path("data/raw/images")
    coco_out_root: Path = Path("data/dataset_coco")
    yolo_out_root: Path = Path("data/dataset_yolo")
    new_images_dir: Path = Path("data/new_images")
    runs_root: Path = Path("runs")
    output_dir: Path = Path("results/detection")

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    yolo_model: str = "yolov8n.yaml"
    yolo_pretrained: bool = False
    yolo_epochs: int = 80
    yolo_image_size: int = 640
    yolo_batch: int = 16

    dfine_size: str = "n"
    dfine_epochs: int = 80
    dfine_image_size: int = 640
    dfine_train_batch: int = 8
    dfine_val_batch: int = 8
    dfine_num_workers: int = 2
    dfine_backbone_pretrained: bool = False
    dfine_use_amp: bool = True
    dfine_repo_dir: Path = Path("third_party/D-FINE")

    conf_threshold: float = 0.40

    def __post_init__(self) -> None:
        for name in (
            "coco_json",
            "images_root",
            "coco_out_root",
            "yolo_out_root",
            "new_images_dir",
            "runs_root",
            "output_dir",
            "dfine_repo_dir",
        ):
            setattr(self, name, Path(getattr(self, name)))
        _split_sums_to_one(self.train_size, self.val_size, self.test_size)


@dataclass
class Config:
    """Top-level container holding both experiment configs."""

    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)


def _filter_known(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys so a stray YAML entry cannot crash the run."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in known}


def load_classification_config(path: str | Path | None = None) -> ClassificationConfig:
    """Load part 1 configuration, falling back to dataclass defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "classification.yaml"
    if not path.exists():
        return ClassificationConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ClassificationConfig(**_filter_known(ClassificationConfig, raw))


def load_detection_config(path: str | Path | None = None) -> DetectionConfig:
    """Load part 2 configuration, falling back to dataclass defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "detection.yaml"
    if not path.exists():
        return DetectionConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return DetectionConfig(**_filter_known(DetectionConfig, raw))


def resolve_path(path: str | Path) -> Path:
    """Interpret relative paths against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()
