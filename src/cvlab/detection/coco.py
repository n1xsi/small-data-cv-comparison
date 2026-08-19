"""
COCO annotation auditing, sanitising and split export.

COCO JSON is the single source of truth in this project: annotations are exported
once from Label Studio, audited here, then converted to whatever format a given
detector expects. Keeping one canonical format means the split is defined exactly
once and both detectors train on identical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd

from ..common.config import ALLOWED_IMAGE_EXTENSIONS
from ..common.io import read_json, write_json

# Label Studio and OS tooling leave these behind next to real exports
EXCLUDED_PATH_PARTS = frozenset({"__MACOSX", ".ipynb_checkpoints", "node_modules", ".git"})


@dataclass
class CocoAudit:
    """Result of validating a COCO export against the images on disk."""

    images: pd.DataFrame
    annotations: pd.DataFrame
    categories: dict[int, str]
    missing_images: list[str]
    invalid_boxes: list[dict[str, Any]]
    out_of_bounds: list[int]

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_annotations(self) -> int:
        return len(self.annotations)

    def objects_per_class(self) -> dict[str, int]:
        """Number of annotated boxes per class."""
        if self.annotations.empty:
            return {}
        counts = self.annotations["category_name"].value_counts().sort_index()
        return counts.to_dict()

    def images_per_class(self) -> dict[str, int]:
        """Number of images containing at least one box of each class."""
        if self.annotations.empty:
            return {}
        pairs = self.annotations[["image_id", "category_name"]].drop_duplicates()
        return pairs["category_name"].value_counts().sort_index().to_dict()

    def summary(self) -> str:
        lines = [
            f"Images: {self.n_images}",
            f"Annotations: {self.n_annotations}",
            f"Categories: {sorted(self.categories.values())}",
            f"Objects per class: {self.objects_per_class()}",
            f"Images per class: {self.images_per_class()}",
            f"Missing image files: {len(self.missing_images)}",
            f"Invalid boxes (zero area / non-finite): {len(self.invalid_boxes)}",
            f"Boxes clipped to image bounds: {len(self.out_of_bounds)}",
        ]
        return "\n".join(lines)


def find_coco_json(search_root: str | Path) -> Path:
    """Locate a COCO export under `search_root`.

    Prefers a file literally named `result.json` (the Label Studio default),
    then the shallowest and largest candidate.
    """
    search_root = Path(search_root)
    if search_root.is_file():
        return search_root

    candidates = [
        path
        for path in search_root.rglob("*.json")
        if not EXCLUDED_PATH_PARTS.intersection(path.parts)
    ]
    if not candidates:
        raise FileNotFoundError(f"No JSON annotation file found under {search_root}")

    def sort_key(path: Path) -> tuple[int, int, int]:
        return (
            0 if path.name == "result.json" else 1,
            len(path.relative_to(search_root).parts),
            -path.stat().st_size,
        )

    return sorted(candidates, key=sort_key)[0]


def build_image_index(images_root: str | Path) -> dict[str, Path]:
    """Index image files by basename, for resolving annotation references.

    Label Studio rewrites `file_name` with upload prefixes and URL escaping, so
    matching on basename is far more robust than trusting the recorded path.
    """
    images_root = Path(images_root)
    index: dict[str, Path] = {}

    for path in images_root.rglob("*"):
        if EXCLUDED_PATH_PARTS.intersection(path.parts):
            continue
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            index.setdefault(path.name, path)

    return index


def resolve_image_path(file_name: str, index: dict[str, Path]) -> Path | None:
    """Map a COCO `file_name` onto a real file, tolerating LS-style prefixes."""
    candidate = Path(unquote(file_name)).name
    if candidate in index:
        return index[candidate]

    # Label Studio prefixes uploads with a hash, e.g. "a1b2c3d4-cat_01.jpg"
    if "-" in candidate:
        stripped = candidate.split("-", 1)[1]
        if stripped in index:
            return index[stripped]

    return None


def sanitize_bbox(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[list[float] | None, bool]:
    """Clip a COCO `[x, y, w, h]` box to the image frame.

    Returns the clipped box and whether clipping changed it. A box that ends up
    with zero area - or was never finite - yields `None` and is dropped.
    """
    if len(bbox) != 4 or not all(np.isfinite(value) for value in bbox):
        return None, False

    x, y, width, height = (float(value) for value in bbox)
    x_min, y_min = x, y
    x_max, y_max = x + width, y + height

    clipped_x_min = max(0.0, min(x_min, image_width))
    clipped_y_min = max(0.0, min(y_min, image_height))
    clipped_x_max = max(0.0, min(x_max, image_width))
    clipped_y_max = max(0.0, min(y_max, image_height))

    new_width = clipped_x_max - clipped_x_min
    new_height = clipped_y_max - clipped_y_min

    if new_width <= 0 or new_height <= 0:
        return None, False

    was_clipped = not (
        abs(clipped_x_min - x_min) < 1e-6
        and abs(clipped_y_min - y_min) < 1e-6
        and abs(new_width - width) < 1e-6
        and abs(new_height - height) < 1e-6
    )
    return [clipped_x_min, clipped_y_min, new_width, new_height], was_clipped


def audit_coco(
    coco_json: str | Path,
    images_root: str | Path,
    *,
    drop_missing: bool = True,
) -> CocoAudit:
    """Validate a COCO export and return tables plus a list of problems found."""
    coco = read_json(coco_json)
    index = build_image_index(images_root)

    categories = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}

    image_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []

    for image in coco.get("images", []):
        resolved = resolve_image_path(str(image.get("file_name", "")), index)
        if resolved is None:
            missing_images.append(str(image.get("file_name", "")))
            if drop_missing:
                continue

        image_rows.append(
            {
                "image_id": int(image["id"]),
                "file_name": str(image.get("file_name", "")),
                "file_path": str(resolved) if resolved else "",
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
            }
        )

    images_df = pd.DataFrame(image_rows)
    valid_ids = set(images_df["image_id"]) if not images_df.empty else set()
    dimensions = (
        images_df.set_index("image_id")[["width", "height"]].to_dict("index")
        if not images_df.empty
        else {}
    )

    annotation_rows: list[dict[str, Any]] = []
    invalid_boxes: list[dict[str, Any]] = []
    out_of_bounds: list[int] = []

    for annotation in coco.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        if image_id not in valid_ids:
            continue

        size = dimensions.get(image_id, {"width": 0, "height": 0})
        cleaned, was_clipped = sanitize_bbox(
            list(annotation.get("bbox", [])),
            float(size["width"]),
            float(size["height"]),
        )

        if cleaned is None:
            invalid_boxes.append({"annotation_id": annotation.get("id"), "image_id": image_id})
            continue

        if was_clipped:
            out_of_bounds.append(int(annotation.get("id", -1)))

        category_id = int(annotation.get("category_id", -1))
        annotation_rows.append(
            {
                "annotation_id": int(annotation.get("id", -1)),
                "image_id": image_id,
                "category_id": category_id,
                "category_name": categories.get(category_id, str(category_id)),
                "bbox_x": cleaned[0],
                "bbox_y": cleaned[1],
                "bbox_w": cleaned[2],
                "bbox_h": cleaned[3],
                "area": cleaned[2] * cleaned[3],
            }
        )

    return CocoAudit(
        images=images_df,
        annotations=pd.DataFrame(annotation_rows),
        categories=categories,
        missing_images=missing_images,
        invalid_boxes=invalid_boxes,
        out_of_bounds=out_of_bounds,
    )


def annotations_by_image(audit: CocoAudit) -> dict[int, pd.DataFrame]:
    """Group the annotation table by image id."""
    if audit.annotations.empty:
        return {}
    return {int(k): v for k, v in audit.annotations.groupby("image_id")}


def dominant_class_per_image(audit: CocoAudit) -> pd.Series:
    """Label each image by the class holding the most boxes in it.

    Detection images can contain several classes, so exact stratification is not
    possible; stratifying on the dominant class keeps the split close to balanced.
    """
    if audit.annotations.empty:
        return pd.Series(dtype=object)

    counts = (
        audit.annotations.groupby(["image_id", "category_name"]).size().rename("n").reset_index()
    )
    counts = counts.sort_values(["image_id", "n", "category_name"], ascending=[True, False, True])
    return counts.groupby("image_id").first()["category_name"]


def export_split_to_coco(
    audit: CocoAudit,
    image_ids: list[int],
    output_json: str | Path,
    *,
    remap_categories: bool = True,
) -> dict[str, Any]:
    """Write one split as a standalone COCO file.

    Args:
        audit: Source audit holding all images and annotations.
        image_ids: Image ids belonging to this split.
        output_json: Destination path.
        remap_categories: Renumber categories to a contiguous 1..N range, which
            several detector implementations assume.

    Returns:
        The COCO dictionary that was written.
    """
    wanted = set(int(i) for i in image_ids)
    images = audit.images[audit.images["image_id"].isin(wanted)]
    annotations = (
        audit.annotations[audit.annotations["image_id"].isin(wanted)]
        if not audit.annotations.empty
        else audit.annotations
    )

    original_ids = sorted(audit.categories)
    if remap_categories:
        id_map = {old: new for new, old in enumerate(original_ids, start=1)}
    else:
        id_map = {old: old for old in original_ids}

    coco: dict[str, Any] = {
        "info": {"description": "Cats vs Dogs detection split"},
        "licenses": [],
        "images": [
            {
                "id": int(row.image_id),
                "file_name": Path(row.file_path).name,
                "width": int(row.width),
                "height": int(row.height),
            }
            for row in images.itertuples()
        ],
        "annotations": [
            {
                "id": int(row.annotation_id),
                "image_id": int(row.image_id),
                "category_id": id_map[int(row.category_id)],
                "bbox": [
                    float(row.bbox_x),
                    float(row.bbox_y),
                    float(row.bbox_w),
                    float(row.bbox_h),
                ],
                "area": float(row.area),
                "iscrowd": 0,
            }
            for row in annotations.itertuples()
        ]
        if not annotations.empty
        else [],
        "categories": [
            {"id": id_map[old], "name": audit.categories[old], "supercategory": "animal"}
            for old in original_ids
        ],
    }

    write_json(coco, output_json)
    return coco


def validate_exported_coco(coco: dict[str, Any], *, split_name: str = "split") -> list[str]:
    """Check an exported split for the problems detectors trip over."""
    problems: list[str] = []

    if not coco.get("images"):
        problems.append(f"{split_name}: no images")
    if not coco.get("categories"):
        problems.append(f"{split_name}: no categories")

    category_ids = {int(c["id"]) for c in coco.get("categories", [])}
    if category_ids and min(category_ids) == 0:
        problems.append(
            f"{split_name}: category ids start at 0; a contiguous 1..N range is expected"
        )

    image_ids = {int(i["id"]) for i in coco.get("images", [])}
    for annotation in coco.get("annotations", []):
        if int(annotation["image_id"]) not in image_ids:
            problems.append(
                f"{split_name}: annotation {annotation['id']} references a missing image"
            )
            break
        if int(annotation["category_id"]) not in category_ids:
            problems.append(f"{split_name}: annotation {annotation['id']} has an unknown category")
            break
        _, _, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            problems.append(f"{split_name}: annotation {annotation['id']} has a degenerate box")
            break

    return problems
