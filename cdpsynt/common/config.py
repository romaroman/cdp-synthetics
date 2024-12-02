from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "SEED",
    "LOG_LEVEL",
    "DIR_PROJECT",
    "DIR_DATA",
    "DIR_IMG",
    "DIR_OUT",
]

SEED = 42
LOG_LEVEL = "INFO"

DIR_PROJECT = Path(os.getenv("CDPSYNT_PROJECT_DIR"))
DIR_DATA = Path(os.getenv("CDPSYNT_DATA_DIR"))

if not DIR_DATA.exists():
    DATA_DIR_FALLBACK = Path(os.getenv("HOME")) / "data" / "cdpsynt"
    print(f"{DIR_DATA} doesn't exist\nThe fallback directory is {DATA_DIR_FALLBACK}")
    DIR_DATA = DATA_DIR_FALLBACK

    if not DIR_DATA.exists():
        raise FileNotFoundError(f"The fallback directory {DIR_DATA} doesn't exist")

DIR_IMG = DIR_DATA / "imgs"
DIR_OUT = DIR_DATA / "out"
