from __future__ import annotations

from pathlib import Path

from ..error import new_error


def check_parent_dir_exists(path: str) -> None:
    parent = Path(path).parent
    if str(parent) != "" and not parent.exists():
        raise new_error(f"File {parent} does not exist")

