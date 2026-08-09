"""
D-FINE (real-time DETR) config generation, training and metric parsing.

D-FINE ships as a research repository driven by YAML configs and a CLI trainer,
with no Python API. This module generates the configs, shells out to the trainer,
and parses COCO metrics back out of its logs.

The backbone is deliberately left unpretrained (`pretrained: false`): the whole
point of the comparison is a transformer detector trained from scratch on the
same ~280 training images as YOLOv8.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common.io import ensure_dir

# The 12 COCO summary values printed by pycocotools, in order
COCO_STAT_NAMES = (
    "mAP50_95",
    "mAP50",
    "mAP75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


@dataclass
class DFineConfigPaths:
    """Locations of the generated config files."""

    dataset: Path
    train: Path
    test: Path


def write_dfine_configs(
    output_dir: str | Path,
    *,
    coco_root: str | Path,
    num_classes: int,
    model_size: str = "n",
    epochs: int = 80,
    image_size: int = 640,
    train_batch: int = 8,
    val_batch: int = 8,
    num_workers: int = 2,
    backbone_pretrained: bool = False,
    use_amp: bool = True,
    run_output_dir: str | Path = "runs/dfine",
) -> DFineConfigPaths:
    """Generate the dataset and model configs D-FINE's trainer expects.

    `num_classes` must exceed the highest category id, because D-FINE reserves
    index 0 for background: two classes with ids 1 and 2 require `num_classes: 3`.
    """
    output_dir = ensure_dir(output_dir)
    coco_root = Path(coco_root)
    run_output_dir = Path(run_output_dir)

    train_images = (coco_root / "images" / "train").as_posix()
    val_images = (coco_root / "images" / "val").as_posix()
    test_images = (coco_root / "images" / "test").as_posix()
    train_json = (coco_root / "annotations" / "train.json").as_posix()
    val_json = (coco_root / "annotations" / "val.json").as_posix()
    test_json = (coco_root / "annotations" / "test.json").as_posix()

    dataset_config = f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

num_classes: {num_classes}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_images}
    ann_file: {train_json}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: {num_workers}
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {val_images}
    ann_file: {val_json}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: {num_workers}
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""

    stop_epoch = max(1, epochs - 10)
    model_config_head = f"""__include__: [
  '../dataset/custom_detection.yml',
  '../runtime.yml',
  './include/dataloader.yml',
  './include/optimizer.yml',
  './include/dfine_hgnetv2.yml',
]

output_dir: {run_output_dir.as_posix()}
sync_bn: False
find_unused_parameters: False
use_amp: {str(use_amp).lower()}
epochs: {epochs}

DFINE:
  backbone: HGNetv2

HGNetv2:
  name: 'B0'
  return_idx: [2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True
  pretrained: {str(backbone_pretrained).lower()}

HybridEncoder:
  in_channels: [512, 1024]
  feat_strides: [16, 32]
  hidden_dim: 128
  use_encoder_idx: [1]
  dim_feedforward: 512
  expansion: 0.34
  depth_mult: 0.5

DFINETransformer:
  feat_channels: [128, 128]
  feat_strides: [16, 32]
  hidden_dim: 128
  dim_feedforward: 512
  num_levels: 2
  num_layers: 3
  eval_idx: -1
  num_points: [6, 6]

optimizer:
  type: AdamW
  lr: 0.0008
  betas: [0.9, 0.999]
  weight_decay: 0.0001

train_dataloader:
  total_batch_size: {train_batch}
  num_workers: {num_workers}
  dataset:
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{image_size}, {image_size}]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        epoch: {stop_epoch}
  collate_fn:
    stop_epoch: {stop_epoch}
    ema_restart_decay: 0.9999
    base_size: {image_size}
    base_size_repeat: ~
  shuffle: True

"""

    def eval_dataloader_block(images: str, annotations: str) -> str:
        """The `val_dataloader` override, pointed at a specific split.

        `img_folder`/`ann_file` are restated here rather than inherited from the
        dataset config. The evaluation-only config differs from the training one
        by exactly these two lines, so they have to be settable per split.
        """
        return f"""val_dataloader:
  total_batch_size: {val_batch}
  num_workers: {num_workers}
  dataset:
    img_folder: {images}
    ann_file: {annotations}
    transforms:
      ops:
        - {{type: Resize, size: [{image_size}, {image_size}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
  shuffle: False
"""

    train_config = model_config_head + eval_dataloader_block(val_images, val_json)
    test_config = model_config_head + eval_dataloader_block(test_images, test_json)

    paths = DFineConfigPaths(
        dataset=output_dir / "custom_detection.yml",
        train=output_dir / f"dfine_hgnetv2_{model_size}_custom.yml",
        test=output_dir / f"dfine_hgnetv2_{model_size}_custom_test.yml",
    )

    paths.dataset.write_text(dataset_config, encoding="utf-8")
    paths.train.write_text(train_config, encoding="utf-8")
    paths.test.write_text(test_config, encoding="utf-8")

    return paths


def build_subprocess_env(repo_dir: str | Path) -> dict[str, str]:
    """Environment for the D-FINE trainer, with the repo on `PYTHONPATH`."""
    env = os.environ.copy()
    repo_dir = str(Path(repo_dir).resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_dir}{os.pathsep}{existing}" if existing else repo_dir
    return env


def run_dfine(
    repo_dir: str | Path,
    config_path: str | Path,
    log_path: str | Path,
    *,
    mode: str = "train",
    checkpoint: str | Path | None = None,
    device: str = "0",
    timeout: int | None = None,
) -> int:
    """Invoke D-FINE's `train.py` and stream its output to a log file.

    Args:
        repo_dir: Checkout of the D-FINE repository.
        config_path: Generated training or test config.
        log_path: Destination for combined stdout/stderr.
        mode: `"train"`, or `"test"` for evaluation-only.
        checkpoint: Weights to resume from or evaluate.
        device: CUDA device index.
        timeout: Optional wall-clock limit in seconds.

    Returns:
        The trainer's exit code.
    """
    repo_dir = Path(repo_dir)
    train_script = repo_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"D-FINE train.py not found at {train_script}. "
            "Clone https://github.com/Peterande/D-FINE into this directory first."
        )

    command = [sys.executable, "train.py", "-c", str(Path(config_path).resolve())]
    if mode == "test":
        command.append("--test-only")
    if checkpoint is not None:
        command += ["-r", str(Path(checkpoint).resolve())]

    log_path = Path(log_path)
    ensure_dir(log_path.parent)

    env = build_subprocess_env(repo_dir)
    env["CUDA_VISIBLE_DEVICES"] = device

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603 - fixed command, no shell
            command,
            cwd=str(repo_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise


def parse_coco_metrics(log_path: str | Path, *, which: str = "last") -> dict[str, float]:
    """Extract COCO metrics from a D-FINE log.

    The trainer prints a pycocotools summary per evaluation; `which` selects the
    `"last"` block (final epoch) or the `"best"` one by mAP50-95.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return {}

    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Preferred form: the compact "Averaged stats" line with 12 floats
    blocks: list[dict[str, float]] = []
    for match in re.finditer(r"Averaged stats.*?\[([^\]]+)\]", text, re.DOTALL):
        values = [float(v) for v in re.findall(r"-?\d+\.\d+", match.group(1))]
        if len(values) >= 12:
            blocks.append(dict(zip(COCO_STAT_NAMES, values[:12], strict=False)))

    # Fallback: parse the human-readable "Average Precision (AP) @[...] = 0.123" lines
    if not blocks:
        pattern = re.compile(
            r"Average (Precision|Recall)\s+\((AP|AR)\)\s+@\[\s*IoU=([\d.:]+)\s*\|"
            r"\s*area=\s*(\w+)\s*\|\s*maxDets=\s*(\d+)\s*\]\s*=\s*(-?[\d.]+)"
        )
        current: dict[str, float] = {}
        for match in pattern.finditer(text):
            kind, _, iou, area, max_dets, value = match.groups()
            number = float(value)

            if kind == "Precision" and area == "all":
                if iou == "0.50:0.95":
                    if "mAP50_95" in current:
                        blocks.append(current)
                        current = {}
                    current["mAP50_95"] = number
                elif iou == "0.50":
                    current["mAP50"] = number
                elif iou == "0.75":
                    current["mAP75"] = number
            elif kind == "Recall" and area == "all" and max_dets == "100":
                current["AR100"] = number

        if current:
            blocks.append(current)

    if not blocks:
        return {}

    if which == "best":
        return max(blocks, key=lambda block: block.get("mAP50_95", -1.0))
    return blocks[-1]


def parse_training_history(log_path: str | Path) -> pd.DataFrame:
    """Build a per-epoch loss/mAP table from a training log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()

    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float]] = []

    for match in re.finditer(r"Epoch:\s*\[(\d+)\].*?loss:\s*([\d.]+)", text):
        rows.append({"epoch": int(match.group(1)), "loss": float(match.group(2))})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # A log line appears per iteration; keep the last value seen for each epoch
    return frame.groupby("epoch", as_index=False).last()


def find_best_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the most suitable checkpoint from a D-FINE run directory."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    for name in ("best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate

    checkpoints = sorted(output_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None
