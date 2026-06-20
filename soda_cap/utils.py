"""Utility functions for SODA-CAP."""

import json
import os
from typing import List, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "soda_cap", "output")

BASELINE_NAME = "baseline"
SODA_NAME = "soda"


DEFAULT_AUX_CANDIDATES = [
    "horizontal_roll",
    "vertical_roll",
    "time_mask",
    "frequency_mask",
    "gaussian_noise",
    "pink_noise",
    "cutout",
]


def ensure_dir(path: str) -> str:
    """Create a directory if it does not already exist."""

    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload) -> None:
    """Save a JSON file with UTF-8 encoding."""

    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    """Load a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def option_names(aux_names: List[str]) -> List[str]:
    """Convert auxiliary augmentation names into policy option names."""

    return ["soda_only"] + [f"soda+{name}" for name in aux_names]


def parse_option(option: str) -> Tuple[bool, str]:
    """Parse a policy option name into SODA and auxiliary components."""

    if option == "soda_only":
        return True, ""
    if option.startswith("soda+"):
        return True, option.split("+", 1)[1]
    raise ValueError(f"Unsupported SODA-CAP option: {option}")
