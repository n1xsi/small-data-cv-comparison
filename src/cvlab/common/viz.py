"""
Shared plotting helpers.

All figures go through `save_or_show` so the same code works in a notebook
(inline display) and in a CLI run (write a PNG into `results/figures/`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def use_headless_backend() -> None:
    """Switch Matplotlib to a non-interactive backend for CLI/CI use."""
    matplotlib.use("Agg")


def save_or_show(save_path: Path | None = None, *, dpi: int = 130, close: bool = True) -> None:
    """Persist the current figure when a path is given, otherwise display it."""
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close()
    else:
        plt.show()


def plot_class_distribution(
    counts: dict[str, int],
    *,
    title: str = "Images per class",
    save_path: Path | None = None,
) -> None:
    """Bar chart of per-class image counts."""
    plt.figure(figsize=(8, 4))
    plt.bar(list(counts.keys()), list(counts.values()), color="#4C78A8")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.grid(axis="y", alpha=0.3)
    save_or_show(save_path)


def get_color_palette(n: int) -> list[tuple[int, int, int]]:
    """Return `n` visually distinct RGB colours for drawing boxes."""
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i % cmap.N)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def image_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    *,
    n_cols: int = 3,
    figsize_per_cell: tuple[float, float] = (4.0, 3.5),
    cmap: str | None = None,
    suptitle: str | None = None,
    save_path: Path | None = None,
) -> None:
    """Render images in a grid with optional per-image titles."""
    if not images:
        return

    n_cols = max(1, min(n_cols, len(images)))
    n_rows = (len(images) + n_cols - 1) // n_cols

    plt.figure(figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows))
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image, cmap=cmap)
        if titles is not None and i < len(titles):
            plt.title(titles[i], fontsize=10)
        plt.axis("off")

    if suptitle:
        plt.suptitle(suptitle)
    save_or_show(save_path)
