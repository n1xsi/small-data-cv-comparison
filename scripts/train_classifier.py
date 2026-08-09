#!/usr/bin/env python
"""
Train and evaluate one classification approach from part 1.

Examples:
    python scripts/train_classifier.py --model pixels
    python scripts/train_classifier.py --model hog
    python scripts/train_classifier.py --model cnn --epochs 20
    python scripts/train_classifier.py --model mobilenet
    python scripts/train_classifier.py --model all

Every run writes metrics to `results/classification/` and figures to
`results/classification/figures/`, so the summary table can be rebuilt from the
CSVs without re-training anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from a clone, without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cvlab.classification import (  # noqa: E402
    build_dataset_dataframe,
    build_hog_svm_grid,
    build_pixel_baseline,
    evaluate_sklearn_model,
    extract_hog_features,
    flatten_pixels,
    load_images_as_arrays,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_and_pr,
    split_dataset,
    summarize,
)
from cvlab.common import (  # noqa: E402
    ensure_dir,
    load_classification_config,
    save_table,
    set_seeds,
    use_headless_backend,
)

MODEL_CHOICES = ("pixels", "hog", "cnn", "mobilenet", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default="all", help="Which approach to train."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to a classification YAML config."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None, help="Override the dataset directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override the results directory."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override CNN epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Cap images per class (0 for no cap).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace):
    if args.dataset_root is not None:
        config.dataset_root = args.dataset_root
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.cnn_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_images_per_class is not None:
        config.max_images_per_class = args.max_images_per_class or None
    if args.seed is not None:
        config.seed = args.seed
    return config


def train_classical(name: str, splits, config, figures_dir: Path | None) -> list:
    """Fit and evaluate a scikit-learn pipeline on raw pixels or HOG features."""
    print(f"\n=== {name} ===")
    print("Loading images into memory...")

    X_train_images = load_images_as_arrays(splits.train, image_size=config.classical_image_size)
    X_val_images = load_images_as_arrays(splits.val, image_size=config.classical_image_size)
    X_test_images = load_images_as_arrays(splits.test, image_size=config.classical_image_size)

    if name == "pixels":
        X_train = flatten_pixels(X_train_images)
        X_val = flatten_pixels(X_val_images)
        X_test = flatten_pixels(X_test_images)
        model = build_pixel_baseline(use_class_weights=config.use_class_weights)
        label = "Raw pixels + LogisticRegression"
    else:
        X_train = extract_hog_features(X_train_images)
        X_val = extract_hog_features(X_val_images)
        X_test = extract_hog_features(X_test_images)
        model = build_hog_svm_grid(use_class_weights=config.use_class_weights)
        label = "HOG + LinearSVC"

    print(f"Feature dimensionality: {X_train.shape[1]}")
    model.fit(X_train, splits.y_train)

    # GridSearchCV exposes the chosen hyperparameters; a plain pipeline does not
    if hasattr(model, "best_params_"):
        print(
            f"Best params: {model.best_params_}  (CV {config.primary_metric}: {model.best_score_:.4f})"
        )

    results = [
        evaluate_sklearn_model(
            model, X_val, splits.y_val, model_name=label, split_name="validation"
        ),
        evaluate_sklearn_model(model, X_test, splits.y_test, model_name=label, split_name="test"),
    ]

    for result in results:
        print(
            f"\n[{result.split_name}] accuracy={result.accuracy:.4f} f1_macro={result.f1_macro:.4f}"
        )
        print(result.report(splits.class_names))

    if figures_dir is not None:
        slug = name
        plot_confusion_matrix(
            results[-1], splits.class_names, save_path=figures_dir / f"{slug}_confusion.png"
        )
        plot_roc_and_pr(results[-1], save_path=figures_dir / f"{slug}_roc_pr.png")

    return results


def train_deep(name: str, splits, config, figures_dir: Path | None) -> list:
    """Train either the from-scratch CNN or the MobileNetV2 transfer model."""
    from cvlab.classification import (
        build_mobilenetv2_transfer,
        build_small_cnn,
        compile_model,
        early_stopping,
        evaluate_keras_model,
        make_tf_dataset,
        plot_gradcam_examples,
        plot_history,
        unfreeze_top_layers,
    )

    print(f"\n=== {name} ===")
    image_size = config.dl_image_size
    input_shape = (*image_size, 3)
    num_classes = len(splits.class_names)

    train_ds = make_tf_dataset(
        splits.train,
        image_size=image_size,
        batch_size=config.batch_size,
        training=True,
        seed=config.seed,
    )
    val_ds = make_tf_dataset(splits.val, image_size=image_size, batch_size=config.batch_size)
    test_ds = make_tf_dataset(splits.test, image_size=image_size, batch_size=config.batch_size)

    if name == "cnn":
        model = compile_model(build_small_cnn(input_shape, num_classes))
        label = "Small CNN (from scratch)"
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.cnn_epochs,
            callbacks=[early_stopping()],
        )
        histories = [("CNN", history)]
    else:
        model, backbone = build_mobilenetv2_transfer(input_shape, num_classes)
        label = "MobileNetV2 (transfer learning)"

        # Stage 1: train the head only, with the ImageNet backbone frozen
        compile_model(model, learning_rate=1e-3)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.tl_head_epochs,
            callbacks=[early_stopping()],
        )

        # Stage 2: unfreeze the top of the backbone at a much lower learning rate
        # A high rate here would destroy the pretrained features it is meant to adapt
        unfreeze_top_layers(backbone, fine_tune_at=config.fine_tune_at)
        compile_model(model, learning_rate=1e-5)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.tl_head_epochs + config.tl_fine_tune_epochs,
            initial_epoch=len(head_history.history["loss"]),
            callbacks=[early_stopping()],
        )
        histories = [
            ("MobileNetV2 head", head_history),
            ("MobileNetV2 fine-tuning", fine_tune_history),
        ]

    results = [
        evaluate_keras_model(model, val_ds, model_name=label, split_name="validation"),
        evaluate_keras_model(model, test_ds, model_name=label, split_name="test"),
    ]

    for result in results:
        print(
            f"\n[{result.split_name}] accuracy={result.accuracy:.4f} f1_macro={result.f1_macro:.4f}"
        )
        print(result.report(splits.class_names))

    if figures_dir is not None:
        for title, history in histories:
            slug = title.lower().replace(" ", "_")
            plot_history(history, title_prefix=title, save_path=figures_dir / f"{slug}_history.png")

        plot_confusion_matrix(
            results[-1], splits.class_names, save_path=figures_dir / f"{name}_confusion.png"
        )
        plot_roc_and_pr(results[-1], save_path=figures_dir / f"{name}_roc_pr.png")

        if name == "mobilenet":
            # A couple of examples per class is enough to see where the model looks
            sample_paths = (
                splits.test.groupby("class_name", group_keys=False).head(2)["file_path"].tolist()
            )
            plot_gradcam_examples(
                sample_paths,
                model,
                splits.class_names,
                image_size=image_size,
                save_path=figures_dir / "mobilenet_gradcam.png",
            )

    weights_dir = ensure_dir(config.output_dir / "weights")
    model.save(weights_dir / f"{name}.keras")
    print(f"Saved weights to {weights_dir / f'{name}.keras'}")

    return results


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_classification_config(args.config), args)

    set_seeds(config.seed, tensorflow=args.model in {"cnn", "mobilenet", "all"})

    output_dir = ensure_dir(config.output_dir)
    figures_dir = None if args.no_figures else ensure_dir(output_dir / "figures")
    if figures_dir is not None:
        use_headless_backend()

    print(f"Dataset: {config.dataset_root}")
    audit = build_dataset_dataframe(
        config.dataset_root,
        keep_only_classes=config.class_names,
        max_images_per_class=config.max_images_per_class,
        seed=config.seed,
    )
    print(audit.summary())
    if audit.broken_files:
        print(f"Skipped {len(audit.broken_files)} unreadable files.")

    splits = split_dataset(
        audit, val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )
    print(f"\nSplit sizes: {splits.sizes()}")
    print(splits.distribution())

    requested = ("pixels", "hog", "cnn", "mobilenet") if args.model == "all" else (args.model,)
    results = []

    for name in requested:
        if name in {"pixels", "hog"}:
            results += train_classical(name, splits, config, figures_dir)
        else:
            results += train_deep(name, splits, config, figures_dir)

    summary = summarize(results, sort_by=config.primary_metric)
    save_table(summary, output_dir / "metrics.csv")

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    if figures_dir is not None:
        test_summary = summary[summary["split"] == "test"]
        if not test_summary.empty:
            plot_model_comparison(
                test_summary,
                metric=config.primary_metric,
                save_path=figures_dir / "model_comparison.png",
            )

    print(f"\nMetrics written to {output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    raise SystemExit(main())
