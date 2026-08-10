"""
Shared pytest fixtures.

The suite deliberately avoids TensorFlow and PyTorch: everything tested here is
data handling and geometry, which is where the bugs that silently ruin a training
run actually live. CI therefore installs only the light dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_dataset(tmp_path: Path) -> Path:
    """A `<root>/<class>/*.png` dataset with 12 cats, 8 dogs and one corrupt file."""
    root = tmp_path / "dataset"
    rng = np.random.default_rng(0)

    for class_name, count in (("cat", 12), ("dog", 8)):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(count):
            pixels = rng.integers(0, 256, size=(40, 30, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(class_dir / f"{class_name}_{index:03d}.png")

    # Not a real image: the audit must reject it rather than crash later
    (root / "cat" / "broken.png").write_bytes(b"this is not a PNG")

    return root


@pytest.fixture
def coco_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """A small COCO export plus its images. Returns `(json_path, images_root)`.

    Includes the awkward cases on purpose: a box hanging off the right edge, a
    zero-area box, an image with two objects, and one with none.
    """
    images_root = tmp_path / "images"
    images_root.mkdir()

    sizes = {1: (100, 80), 2: (100, 80), 3: (60, 60), 4: (100, 80)}
    for image_id, (width, height) in sizes.items():
        Image.new("RGB", (width, height), color=(image_id * 40 % 256, 100, 150)).save(
            images_root / f"img_{image_id}.jpg"
        )

    coco = {
        "images": [
            {"id": image_id, "file_name": f"img_{image_id}.jpg", "width": w, "height": h}
            for image_id, (w, h) in sizes.items()
        ],
        "annotations": [
            # Ordinary box, fully inside the frame
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 20],
                "area": 600,
                "iscrowd": 0,
            },
            # Two objects in one image, so dominant-class logic gets exercised
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [5, 5, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "bbox": [40, 30, 25, 25],
                "area": 625,
                "iscrowd": 0,
            },
            # Overhangs the right edge by 20px: must be clipped, not dropped
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [50, 10, 30, 20],
                "area": 600,
                "iscrowd": 0,
            },
            # Zero width: must be dropped entirely
            {
                "id": 5,
                "image_id": 4,
                "category_id": 1,
                "bbox": [10, 10, 0, 20],
                "area": 0,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"},
        ],
    }

    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    return json_path, images_root
