"""
dataset.py - JSON dataset loading and persistence utilities.

Provides robust file resolution supporting project execution from either
the workspace root directory or from inside the app/ directory.
"""

import json
import os
from pathlib import Path

# Base directories
CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
PROJECT_ROOT = APP_DIR.parent

def resolve_path(file_path: str, for_writing: bool = False) -> Path:
    """
    Resolve a relative or absolute file path sensibly whether running
    from the repository root or within the app/ directory.
    """
    path = Path(file_path)
    if path.is_absolute():
        return path

    # Search candidate paths in order of preference
    candidates = [
        path,
        PROJECT_ROOT / path,
        APP_DIR / path,
        PROJECT_ROOT / "data" / path.name if "data" in str(path) else None,
    ]

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    # If writing a new file, map target directory sensibly
    if for_writing:
        if str(path).startswith("data") or path.name.endswith(".json") or path.name.endswith(".csv"):
            target = PROJECT_ROOT / "data" / path.name
        elif str(path).startswith("models") or path.suffix == ".keras":
            target = PROJECT_ROOT / "models" / path.name
        elif str(path).startswith("artifacts") or path.suffix == ".png":
            target = PROJECT_ROOT / "artifacts" / path.name
        else:
            target = PROJECT_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # Default fallback
    return PROJECT_ROOT / path

def load_data(file_path: str):
    """
    Safely load a JSON dataset from disk.
    """
    target = resolve_path(file_path, for_writing=False)
    if not target.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path} (resolved: {target})")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def save_data(file_path: str, data: dict, message: str = "Saved Data"):
    """
    Safely write a dictionary as formatted JSON to disk.
    """
    target = resolve_path(file_path, for_writing=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    if message:
        print(f"Log: {message} -> {target.name}")
