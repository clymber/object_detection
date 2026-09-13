"""
Utilities related to file paths.
"""

from __future__ import annotations

import re
from pathlib import Path


def relative_to_userhome(path: Path | str) -> str:
    """
    Return a stable POSIX-style path relative to the user's home directory.

    The home directory will be replaced with a tilde (~) if applicable.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    userhome = str(Path.home())
    resolved = candidate.resolve().as_posix()
    return resolved.replace(userhome, "~", 1)


def _natural_name_key(path: Path) -> list[str | int]:
    """
    Return name components that sort embedded numbers numerically.
    """
    components: list[str | int] = []
    for component in re.split(r"(\d+)", path.name):
        if component.isdigit():
            components.append(int(component))
        else:
            components.append(component.casefold())

    return components


def dir_tree(
    path: Path | str,
    *,
    max_depth: int | None = None,
    max_children: int | None = 5,
) -> str:
    """
    Return a hierarchical preview of a directory.

    The root is at depth 0 and all of its immediate entries are displayed.
    Files and directories with dot-prefixed names are ignored.
    ``max_depth`` controls how deeply directory entries are expanded.
    ``max_children`` limits entries displayed inside expanded subdirectories,
    while leaving the root listing complete.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative or None.")
    if max_children is not None and max_children < 0:
        raise ValueError("max_children must be non-negative or None.")

    lines = [root.name]

    def add_children(directory: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth >= max_depth:
            return

        # Ignore hidden files and directories
        visible_entries = (
            ent for ent in directory.iterdir() if not ent.name.startswith(".")
        )

        # Display entries other than directories first
        entries = sorted(
            visible_entries,
            key=lambda entry: (entry.is_dir(), _natural_name_key(entry)),
        )

        if depth > 0 and max_children is not None:
            shown = entries[:max_children]
            omitted = len(entries) - len(shown)
        else:
            shown = entries
            omitted = 0

        display_entries: list[Path | None] = [*shown] # shallow copy
        if omitted:
            display_entries.append(None)

        for index, entry in enumerate(display_entries):
            is_last = index == len(display_entries) - 1
            branch = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if entry is None:
                noun = "entry" if omitted == 1 else "entries"
                lines.append(f"{prefix}{branch}... ({omitted} more {noun})")
                continue

            lines.append(f"{prefix}{branch}{entry.name}")
            if entry.is_dir() and not entry.is_symlink():
                add_children(entry, prefix + child_prefix, depth + 1)

    add_children(root, "", 0)
    return "\n".join(lines)


def increment_path(path: Path | str) -> Path:
    """
    Return a non-existing path by appending an incrementing numeric suffix.
    """
    base_path = Path(path)
    if not base_path.exists():
        return base_path

    suffix = 2
    while True:
        candidate = base_path.with_name(f"{base_path.name}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
