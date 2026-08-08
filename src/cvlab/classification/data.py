"""
Dataset auditing and stratified splitting for the classification task.

The audit step exists because a folder of scraped images is never clean: some
files are truncated, mislabelled as JPEG, or animated. Decoding every file up
front means a corrupt image fails here — with a filename we can inspect — rather
than midway through training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from ..common.config import ALLOWED_IMAGE_EXTENSIONS


@dataclass
class DatasetAudit:
    """Outcome of scanning a class-per-directory image dataset."""

    frame: pd.DataFrame
    class_names: list[str]
    broken_files: list[str]

    @property
    def class_counts(self) -> dict[str, int]:
        """Number of valid images per class."""
        return self.frame["class_name"].value_counts().sort_index().to_dict()

    def summary(self) -> str:
        """Human-readable one-paragraph summary of the audit."""
        lines = [
            f"Classes: {self.class_names}",
            f"Valid images: {len(self.frame)}",
            f"Broken/unreadable files: {len(self.broken_files)}",
        ]
        lines += [f"  {name}: {count}" for name, count in self.class_counts.items()]
        return "\n".join(lines)


@dataclass
class DataSplits:
    """Stratified train/validation/test partition of the audited dataset."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]

    @property
    def y_train(self) -> np.ndarray:
        return self.train["label"].to_numpy(dtype=np.int32)

    @property
    def y_val(self) -> np.ndarray:
        return self.val["label"].to_numpy(dtype=np.int32)

    @property
    def y_test(self) -> np.ndarray:
        return self.test["label"].to_numpy(dtype=np.int32)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def distribution(self) -> pd.DataFrame:
        """Per-split class counts, to confirm stratification held."""
        return (
            pd.DataFrame(
                {
                    "train": self.train["class_name"].value_counts(),
                    "val": self.val["class_name"].value_counts(),
                    "test": self.test["class_name"].value_counts(),
                }
            )
            .fillna(0)
            .astype(int)
            .sort_index()
        )


def _is_readable_image(path: Path) -> bool:
    """Return True if Pillow can fully decode the file.

    `Image.load()` is deliberate: `Image.open()` alone is lazy and will happily
    accept a truncated file whose pixel data is unreadable.
    """
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except Exception:
        return False


def build_dataset_dataframe(
    dataset_root: str | Path,
    *,
    keep_only_classes: Sequence[str] | None = None,
    max_images_per_class: int | None = None,
    seed: int = 42,
    show_progress: bool = True,
) -> DatasetAudit:
    """Scan a `<root>/<class_name>/*.jpg` layout and validate every image.

    Args:
        dataset_root: Directory holding one subdirectory per class.
        keep_only_classes: Restrict to these class names, if given.
        max_images_per_class: Cap per class, sampled reproducibly.
        seed: Seed for the sampling cap.
        show_progress: Display a progress bar per class.

    Returns:
        A `DatasetAudit` with the valid-image table, class names and rejects.

    Raises:
        FileNotFoundError: The root directory is missing.
        ValueError: Fewer than two classes, or no readable images at all.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    class_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if keep_only_classes is not None:
        wanted = set(keep_only_classes)
        class_dirs = [p for p in class_dirs if p.name in wanted]

    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class subdirectories in {dataset_root}, found {len(class_dirs)}."
        )

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    broken_files: list[str] = []

    for label, class_dir in enumerate(class_dirs):
        paths = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )

        if max_images_per_class is not None and len(paths) > max_images_per_class:
            chosen = rng.choice(len(paths), size=max_images_per_class, replace=False)
            paths = [paths[i] for i in sorted(chosen)]

        iterator = tqdm(paths, desc=f"Validating {class_dir.name}", disable=not show_progress)
        for path in iterator:
            if _is_readable_image(path):
                records.append(
                    {"file_path": str(path), "class_name": class_dir.name, "label": label}
                )
            else:
                broken_files.append(str(path))

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(
            f"No readable images found under {dataset_root}. "
            "Check the directory layout and file formats."
        )

    return DatasetAudit(
        frame=frame,
        class_names=[p.name for p in class_dirs],
        broken_files=broken_files,
    )


def split_dataset(
    audit: DatasetAudit,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """Split the audited dataset, stratified by label.

    The split happens before any feature extraction or scaling, so no statistic
    derived from validation or test data can leak into training.
    """
    holdout_size = val_size + test_size
    if not 0.0 < holdout_size < 1.0:
        raise ValueError(f"val_size + test_size must be in (0, 1), got {holdout_size}")

    train_df, holdout_df = train_test_split(
        audit.frame,
        test_size=holdout_size,
        stratify=audit.frame["label"],
        random_state=seed,
    )

    # Rescale test_size to a fraction *of the holdout* rather than of the whole set.
    relative_test_size = test_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        stratify=holdout_df["label"],
        random_state=seed,
    )

    return DataSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        class_names=audit.class_names,
    )


def load_images_as_arrays(
    df: pd.DataFrame,
    *,
    image_size: tuple[int, int] = (64, 64),
    grayscale: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Load images into a float32 array scaled to [0, 1].

    Used by the classical-ML baselines, which need the whole split in memory.
    Returns shape `(n, h, w)` for grayscale and `(n, h, w, 3)` for RGB.
    """
    arrays: list[np.ndarray] = []
    iterator = tqdm(
        df["file_path"], total=len(df), desc="Loading images", disable=not show_progress
    )

    for file_path in iterator:
        with Image.open(file_path) as image:
            image = image.convert("L" if grayscale else "RGB")
            image = image.resize(image_size)
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)

    return np.stack(arrays)
