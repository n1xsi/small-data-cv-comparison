#!/usr/bin/env python
"""
Fail if any committed notebook still carries outputs or execution counts.

Notebook outputs are what turn a repository into a hundred-megabyte download and
make every diff unreadable. This runs in CI so a notebook cannot be committed
dirty by accident.

Fix a failure with:
    pip install nbstripout && nbstripout notebooks/*.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = ("notebooks",)


def check_notebook(path: Path) -> list[str]:
    """Return a list of problems found in one notebook."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not be parsed: {error}"]

    problems = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            problems.append(f"cell {index} has {len(cell['outputs'])} stored output(s)")
        if cell.get("execution_count") is not None:
            problems.append(f"cell {index} has execution_count={cell['execution_count']}")

    return problems


def main() -> int:
    notebooks = sorted(
        path
        for directory in NOTEBOOK_DIRS
        for path in (REPO_ROOT / directory).glob("**/*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )

    if not notebooks:
        print("No notebooks found; nothing to check.")
        return 0

    failed = False
    for path in notebooks:
        relative = path.relative_to(REPO_ROOT)
        problems = check_notebook(path)

        if problems:
            failed = True
            print(f"FAIL {relative}")
            for problem in problems[:5]:
                print(f"     {problem}")
            if len(problems) > 5:
                print(f"     ... and {len(problems) - 5} more")
        else:
            print(f"OK   {relative}")

    if failed:
        print("\nStrip outputs with: nbstripout notebooks/*.ipynb")
        return 1

    print(f"\nAll {len(notebooks)} notebook(s) are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
