#!/usr/bin/env python
"""
Turn a Label Studio COCO export into the two dataset layouts part 2 needs.

The COCO export is the single source of truth: it is audited once, split once,
and then written out in both formats. Both detectors therefore see byte-identical
splits, which is what makes their scores comparable.

Examples:
    python scripts/prepare_detection.py
    python scripts/prepare_detection.py --coco-json data/raw/annotations/result.json
    python scripts/prepare_detection.py --skip-coco     # YOLO layout only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvlab.common import ensure_dir, load_detection_config, save_table, set_seeds  # noqa: E402
from cvlab.detection import (  # noqa: E402
    audit_coco,
    audit_yolo_dataset,
    build_yolo_dataset,
    export_split_to_coco,
    find_coco_json,
    split_detection_dataset,
    summarize_splits,
    validate_exported_coco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to a detection YAML config."
    )
    parser.add_argument(
        "--coco-json", type=Path, default=None, help="Override the COCO export path."
    )
    parser.add_argument(
        "--images-root", type=Path, default=None, help="Override the image directory."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the split seed.")
    parser.add_argument("--skip-coco", action="store_true", help="Do not write the COCO layout.")
    parser.add_argument("--skip-yolo", action="store_true", help="Do not write the YOLO layout.")
    return parser.parse_args()


def write_coco_layout(audit, splits, output_root: Path) -> None:
    """Write `images/{split}/` plus `annotations/{split}.json` for D-FINE."""
    annotations_dir = ensure_dir(output_root / "annotations")
    image_paths = audit.images.set_index("image_id")["file_path"].to_dict()

    for split_name, image_ids in splits.as_dict().items():
        image_dir = ensure_dir(output_root / "images" / split_name)

        for image_id in image_ids:
            source = Path(str(image_paths.get(int(image_id), "")))
            if not source.exists():
                continue
            destination = image_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)

        coco = export_split_to_coco(
            audit, image_ids, annotations_dir / f"{split_name}.json", remap_categories=True
        )
        problems = validate_exported_coco(coco, split_name=split_name)

        print(
            f"  {split_name}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} boxes -> {annotations_dir / f'{split_name}.json'}"
        )
        for problem in problems:
            print(f"    WARNING: {problem}")


def main() -> int:
    args = parse_args()
    config = load_detection_config(args.config)

    if args.coco_json is not None:
        config.coco_json = args.coco_json
    if args.images_root is not None:
        config.images_root = args.images_root
    if args.seed is not None:
        config.seed = args.seed

    set_seeds(config.seed, tensorflow=False)

    # Accept either a direct path to the JSON or a directory to search
    coco_json = config.coco_json
    if not coco_json.exists():
        search_root = coco_json if coco_json.is_dir() else coco_json.parent
        if not search_root.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at {coco_json}. See data/README.md for the expected layout."
            )
        coco_json = find_coco_json(search_root)
        print(f"Using annotation file: {coco_json}")

    print(f"Auditing {coco_json} against {config.images_root} ...")
    audit = audit_coco(coco_json, config.images_root)
    print(audit.summary())

    if audit.missing_images:
        print(f"\nWARNING: {len(audit.missing_images)} annotated images were not found on disk.")
        for name in audit.missing_images[:5]:
            print(f"  {name}")

    splits = split_detection_dataset(
        audit, val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )

    split_summary = summarize_splits(audit, splits)
    print("\n=== Splits ===")
    print(split_summary.to_string(index=False))

    output_dir = ensure_dir(config.output_dir)
    save_table(split_summary, output_dir / "split_summary.csv")

    if not args.skip_coco:
        print(f"\nWriting COCO layout to {config.coco_out_root} ...")
        write_coco_layout(audit, splits, config.coco_out_root)

    if not args.skip_yolo:
        print(f"\nWriting YOLO layout to {config.yolo_out_root} ...")
        dataset_yaml = build_yolo_dataset(audit, splits, config.yolo_out_root)
        print(f"  dataset.yaml -> {dataset_yaml}")

        # Read the labels back off disk: a conversion bug here trains silently to
        # near-zero mAP, so it is worth catching before an 80-epoch run
        yolo_audit = audit_yolo_dataset(config.yolo_out_root)
        print("\n=== YOLO layout verification ===")
        print(yolo_audit.to_string(index=False))
        save_table(yolo_audit, output_dir / "yolo_audit.csv")

        problems = yolo_audit[
            (yolo_audit["missing_label_files"] > 0) | (yolo_audit["out_of_range_boxes"] > 0)
        ]
        if not problems.empty:
            print("\nWARNING: conversion problems detected in the splits above.")
            return 1

    print(f"\nDone. Summary tables in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
