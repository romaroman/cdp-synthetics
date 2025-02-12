from enum import Enum
from pathlib import Path
from typing import Any
import re

import cv2 as cv
import numpy as np

__all__ = ["CustomEnum", "scan_subsets", "str2nmb", "flatten", "img2blocks", "imgray", "minmax", "normalize_img", "cycle_pad_lists", "globre"]

class CustomEnum(Enum):
    def __str__(self) -> str:
        return str(self.name)

    def blob(self) -> str:
        return str(self.value[0])

    @classmethod
    def to_list(cls) -> list[Any]:
        return list(map(lambda c: c, cls))

    @classmethod
    def to_string_list(cls) -> str:
        return ", ".join([str(c) for c in cls])

    @classmethod
    def match_enum(cls, search_string: str) -> "CustomEnum":
        for guess in sorted([str(x) for x in cls.to_list()], key=len, reverse=True):
            if search_string.find(guess) != -1:
                return cls[guess]
        else:
            raise ValueError(f"Could not find enum matching {search_string}")

    @classmethod
    def match_enum_strict(cls, search_string: str) -> "CustomEnum":
        for guess in cls:
            if guess.value == search_string:
                return guess
        else:
            raise ValueError(f"Could not find enum matching {search_string}")


def scan_subsets(
    root_dir: Path,
    included_keywords: list[str] | None = None,
    excluded_keywords: list[str] | None = None,
    depth: int = 3,
) -> list[Path]:
    excluded_keywords = excluded_keywords or []
    included_keywords = included_keywords or []

    dirs = Path(root_dir).resolve().glob("/".join(["*"] * depth))
    dirs = filter(
        lambda x: all([str(x).find(keyword) != -1 for keyword in included_keywords]),
        dirs,
    )
    dirs = filter(
        lambda x: all([str(x).find(keyword) == -1 for keyword in excluded_keywords]),
        dirs,
    )
    dirs = sorted(list(filter(lambda x: x.is_dir(), dirs)))
    return dirs

def str2nmb(string: str) -> float | int:
    string_filtered = "".join(c for c in string if c.isnumeric() or c in [".", "-"])
    return (
        -1
        if string_filtered == ""
        else (
            int(string_filtered)
            if string_filtered.find(".") == -1
            else float(string_filtered)
        )
    )

def flatten(list_):
    # if not isinstance(list_[0], list):
    #     return list_
    return [item for sublist in list_ for item in sublist]


def img2blocks(
    img: np.ndarray,
    block_size: tuple[int, int],
    step: int = 1,
    rows: list[int | None] = None,
    cols: list[int | None] = None,
) -> list[np.ndarray]:
    if not rows or not cols:
        rows, cols = __get_blocks_idxs(img.shape[:2], block_size, step)

    n = len(rows)
    blocks = list()
    for i in range(n):
        blocks.append(
            img[rows[i] : rows[i] + block_size[0], cols[i] : cols[i] + block_size[1]]
        )

    return blocks


def __get_blocks_idxs(
    img_shape: tuple[int, int], block_size: tuple[int, int], step: int = 1
) -> tuple[list[int], list[int]]:
    ss = img_shape - np.asarray(block_size) + 1

    img_mat = np.zeros((ss[0], ss[1]))
    img_mat[::step, ::step] = 1
    img_mat[img_mat.shape[0] - 1, ::step] = 1
    img_mat[::step, img_mat.shape[1] - 1] = 1
    img_mat[img_mat.shape[0] - 1, img_mat.shape[1] - 1] = 1

    return np.where(img_mat == 1)


def imgray(img_3c: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_3c, cv.COLOR_BGR2GRAY) if len(img_3c.shape) == 3 and img_3c.shape[2] == 3 else img_3c

def minmax(
    img: np.ndarray,
    vmin: int | None = None,
    vmax: int | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    vmin = vmin or img.min()
    vmax = vmax or img.max()
    return ((img - vmin) / (vmax - vmin)).astype(dtype)


def normalize_img(img: np.ndarray) -> np.ndarray:
    return (minmax(imgray(img)) * 255).astype(np.uint8)

def cycle_pad_lists(list_of_lists: list[Any]) -> Any:
    max_length = max(len(lst) for lst in list_of_lists)
    padded_lists = [
        lst + [lst[i % len(lst)] for i in range(max_length - len(lst))]
        for lst in list_of_lists
    ]
    return padded_lists

def globre(path: Path, pattern: str) -> list[str]:
    return list(filter(lambda x: re.compile(pattern).match(x.name), path.rglob("*")))