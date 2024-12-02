from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Any, Callable

import numpy as np

__all__ = ["Entity", "EntityCDP"]


class Entity:
    def __init__(self, img: np.ndarray) -> Entity:
        self.img: np.ndarray = img

        self.__applied: list[Callable] = []
        self.__img_bp: np.ndarray = self.img.copy()

    @property
    @abstractmethod
    def layout(self) -> dict[str, list[float]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self.img.shape

    @property
    def applied(self) -> list[Callable]:
        return self.__applied

    @property
    def is_applied(self) -> bool:
        return bool(self.__applied)

    @property
    def full(self) -> np.ndarray:
        return self.img

    def __bool__(self) -> bool:
        return bool(self.img.any())

    def __getitem__(self, part: str) -> np.ndarray:
        return self.get_part(part)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        cparts = [self.name, "x".join(map(str, self.shape)), str(self.img.dtype)]
        return " ".join(cparts)

    def __eq__(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        return np.array_equal(self.img, other.img)

    def get_part(self, part: str) -> np.ndarray:
        y1, y2, x1, x2 = self.get_part_coords(part)
        return self.img[y1:y2, x1:x2]

    def get_parts(self, parts: list[str]) -> list[np.ndarray]:
        return [self.get_part(part) for part in parts]

    def get_part_coords(self, part: str) -> tuple[int, int, int, int]:
        h, w = self.shape[:2]
        return tuple(map(lambda x: int(np.round(x)), np.asarray(self.layout[part]) * np.asarray([h, h, w, w], dtype=np.float32)))

    def reset(self) -> None:
        self.img = self.__img_bp.copy()

    def get_mask(self, parts: list[str]) -> np.ndarray:
        img_mask: np.ndarray = np.zeros(shape=self.img.shape[:2], dtype=np.uint8)
        for part in parts:
            y1, y2, x1, x2 = self.get_part_coords(part)
            img_mask[y1:y2, x1:x2] = 255
        return img_mask

    def apply(
        self,
        func: Callable,
        entity_other: Entity | None = None,
        parts: list[str] = None,
        **kwargs,
    ) -> None:
        self.__applied.append(partial(func, **kwargs))
        if not parts:
            self.img = (
                func(self.img, entity_other.img, **kwargs)
                if entity_other
                else func(self.img, **kwargs)
            )
            return self

        for part in parts:
            y1, y2, x1, x2 = self.get_part_coords(part)
            if entity_other:
                y1r, y2r, x1r, x2r = entity_other.get_part_coords(part)
                self.img[y1:y2, x1:x2] = func(
                    self.img[y1:y2, x1:x2], entity_other.img[y1r:y2r, x1r:x2r], **kwargs
                )
            else:
                self.img[y1:y2, x1:x2] = func(self.img[y1:y2, x1:x2], **kwargs)


class EntityCDP(Entity):
    @property
    def name(self) -> str:
        return "CDP"

    @property
    def layout(self) -> dict[str, list[float]]:
        return {
            "rcod": [
                0.21791044776119403,
                0.5582089552238806,
                0.28076923076923077,
                0.7192307692307692,
            ],
            "full": [0, 1, 0, 1],
            "cdp": [0, 1, 0, 1],
        }

    @property
    def rcod(self) -> np.ndarray:
        return self.get_part("rcod")

    @property
    def cdp(self) -> np.ndarray:
        return self.img
