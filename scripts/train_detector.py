#!/usr/bin/env python
"""
Train and evaluate a detector from part 2.

Both models are trained from random initialisation on the same ~280 images, so
the comparison measures architecture and inductive bias rather than the size of
someone else's pretraining run.

Examples:
    python scripts/train_detector.py --model yolov8
    python scripts/train_detector.py --model yolov8 --epochs 120
    python scripts/train_detector.py --model dfine
    python scripts/train_detector.py --model yolov8 --eval-only --weights runs/yolo/yolov8n_scratch/weights/best.pt

D-FINE needs its upstream repository:
    git clone https://github.com/Peterande/D-FINE third_party/D-FINE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cvlab.common import (  # noqa: E402
    ensure_dir,
    load_detection_config,
    read_json,
    save_table,
    set_seeds,
    use_headless_backend,
)
from cvlab.detection import (  # noqa: E402
    DetectorResult,
    comparison_table,
    evaluate_yolo,
    find_best_checkpoint,
    parse_coco_metrics,
    plot_metric_comparison,
    plot_training_curves,
    run_dfine,
    train_yolo,
    write_dfine_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", choices=("yolov8", "dfine"), required=True, help="Detector to train."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to a detection YAML config."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count.")
    parser.add_argument("--batch", type=int, default=None, help="Override the batch size.")
    parser.add_argument(
        "--image-size", type=int, default=None, help="Override the training image size."
    )
    parser.add_argument("--device", default=None, help="CUDA device index, or 'cpu'.")
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training and evaluate existing weights."
    )
    parser.add_argument("--weights", type=Path, default=None, help="Checkpoint to evaluate.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def run_yolo(args: argparse.Namespace, config, output_dir: Path, figures_dir: Path | None) -> list:
    dataset_yaml = config.yolo_out_root / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} not found. Run scripts/prepare_detection.py first."
        )

    epochs = args.epochs or config.yolo_epochs
    batch = args.batch or config.yolo_batch
    image_size = args.image_size or config.yolo_image_size
    run_name = "yolov8n_scratch"

    weights = args.weights
    if not args.eval_only:
        print(f"Training {config.yolo_model} for {epochs} epochs (from scratch)...")
        artifacts = train_yolo(
            dataset_yaml,
            model_spec=config.yolo_model,
            epochs=epochs,
            image_size=image_size,
            batch=batch,
            project=config.runs_root / "yolo",
            name=run_name,
            seed=config.seed,
            device=args.device,
        )
        weights = artifacts["best_weights"]
        print(f"Best weights: {weights}")

        if figures_dir is not None and artifacts["results_csv"].exists():
            plot_training_curves(
                artifacts["results_csv"], save_path=figures_dir / "yolov8_training_curves.png"
            )

    if weights is None:
        weights = config.runs_root / "yolo" / run_name / "weights" / "best.pt"

    results = []
    for split in ("val", "test"):
        print(f"\nEvaluating on the {split} split...")
        metrics = evaluate_yolo(
            weights,
            dataset_yaml,
            split=split,
            image_size=image_size,
            batch=batch,
            project=config.runs_root / "yolo",
            name=f"{run_name}_{split}",
            device=args.device,
        )
        printable = {k: v for k, v in metrics.items() if k != "split"}
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in printable.items() if v is not None))
        results.append(DetectorResult(model="YOLOv8n (scratch)", split=split, metrics=metrics))

    return results


def run_dfine_experiment(
    args: argparse.Namespace, config, output_dir: Path, figures_dir: Path | None
) -> list:
    annotations_dir = config.coco_out_root / "annotations"
    if not (annotations_dir / "train.json").exists():
        raise FileNotFoundError(
            f"{annotations_dir / 'train.json'} not found. Run scripts/prepare_detection.py first."
        )

    epochs = args.epochs or config.dfine_epochs
    batch = args.batch or config.dfine_train_batch
    image_size = args.image_size or config.dfine_image_size

    # D-FINE reserves class index 0 for background, so num_classes is (number of real classes + 1)
    categories = read_json(annotations_dir / "train.json").get("categories", [])
    num_classes = max(int(c["id"]) for c in categories) + 1 if categories else 3
    print(f"Categories: {[c['name'] for c in categories]} -> num_classes={num_classes}")

    repo_dir = config.dfine_repo_dir
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"D-FINE repository not found at {repo_dir}. Clone it with:\n"
            f"  git clone https://github.com/Peterande/D-FINE {repo_dir}"
        )

    # The generated configs must live inside the upstream repo's config tree,
    # because its `__include__` directives are resolved relative to that tree
    config_dir = repo_dir / "configs" / "dfine" / "custom"
    dataset_config_dir = repo_dir / "configs" / "dataset"
    ensure_dir(dataset_config_dir)

    run_output_dir = config.runs_root / "dfine" / f"dfine_hgnetv2_{config.dfine_size}_scratch"
    paths = write_dfine_configs(
        config_dir,
        coco_root=config.coco_out_root.resolve(),
        num_classes=num_classes,
        model_size=config.dfine_size,
        epochs=epochs,
        image_size=image_size,
        train_batch=batch,
        val_batch=config.dfine_val_batch,
        num_workers=config.dfine_num_workers,
        backbone_pretrained=config.dfine_backbone_pretrained,
        use_amp=config.dfine_use_amp,
        run_output_dir=run_output_dir.resolve(),
    )

    # `write_dfine_configs` puts the dataset config next to the model configs;
    # the model config's `__include__` expects it one level up, in configs/dataset/
    (dataset_config_dir / "custom_detection.yml").write_text(
        paths.dataset.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Configs written to {config_dir}")

    logs_dir = ensure_dir(output_dir / "logs")
    train_log = logs_dir / "dfine_train.log"

    if not args.eval_only:
        print(f"Training D-FINE-{config.dfine_size} for {epochs} epochs (from scratch)...")
        print(f"  Streaming output to {train_log}")
        code = run_dfine(
            repo_dir,
            paths.train,
            train_log,
            mode="train",
            device=args.device or "0",
        )
        if code != 0:
            print(f"D-FINE trainer exited with code {code}. See {train_log}.")
            return []

    checkpoint = args.weights or find_best_checkpoint(run_output_dir)
    if checkpoint is None:
        print(f"No checkpoint found in {run_output_dir}.")
        return []
    print(f"Checkpoint: {checkpoint}")

    results = []

    val_metrics = parse_coco_metrics(train_log, which="best")
    if val_metrics:
        print("  val: " + "  ".join(f"{k}={v:.4f}" for k, v in list(val_metrics.items())[:4]))
        results.append(
            DetectorResult(
                model=f"D-FINE-{config.dfine_size} (scratch)", split="val", metrics=val_metrics
            )
        )

    test_log = logs_dir / "dfine_test.log"
    print(f"\nEvaluating on the test split (log: {test_log})...")
    code = run_dfine(
        repo_dir,
        paths.test,
        test_log,
        mode="test",
        checkpoint=checkpoint,
        device=args.device or "0",
    )
    test_metrics = parse_coco_metrics(test_log, which="last")
    if code == 0 and test_metrics:
        print("  test: " + "  ".join(f"{k}={v:.4f}" for k, v in list(test_metrics.items())[:4]))
        results.append(
            DetectorResult(
                model=f"D-FINE-{config.dfine_size} (scratch)", split="test", metrics=test_metrics
            )
        )
    else:
        print(f"Could not parse test metrics (exit code {code}). See {test_log}.")

    return results


def main() -> int:
    args = parse_args()
    config = load_detection_config(args.config)
    set_seeds(config.seed, tensorflow=False, torch=True)

    output_dir = ensure_dir(config.output_dir)
    figures_dir = None
    if not args.no_figures:
        figures_dir = ensure_dir(output_dir / "figures")
        use_headless_backend()

    if args.model == "yolov8":
        results = run_yolo(args, config, output_dir, figures_dir)
    else:
        results = run_dfine_experiment(args, config, output_dir, figures_dir)

    if not results:
        print("No metrics produced.")
        return 1

    table = comparison_table(results)
    metrics_path = output_dir / f"metrics_{args.model}.csv"
    save_table(table, metrics_path)

    print("\n=== Results ===")
    print(table.to_string(index=False))
    print(f"\nMetrics written to {metrics_path}")

    # Merge with any other detector's results so the comparison chart covers both
    all_tables = [pd.read_csv(path) for path in sorted(output_dir.glob("metrics_*.csv"))]
    if figures_dir is not None and len(all_tables) > 1:
        combined = pd.concat(all_tables, ignore_index=True)
        save_table(combined, output_dir / "metrics.csv")
        plot_metric_comparison(
            combined, split="test", save_path=figures_dir / "detector_comparison.png"
        )
        print(f"Combined comparison written to {output_dir / 'metrics.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
