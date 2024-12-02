import time
from functools import partial
from itertools import product
from typing import Any, Callable, ClassVar, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .. import utils
from ..common import LOGGER
from .subset import Subset

__all__ = ["Loader"]


class Loader:
    headers: ClassVar[list[str]] = ["uuid", "shot", "part", "block"]

    def __init__(
        self,
        parts: list[str],
        uuids: list[str] | None = None,
        block_size: tuple[int, int] | None = None,
        block_step: int | None = None,
        channeling: bool = False,
        padding: int = 0,
        cpus: int = 1,
    ) -> "Loader":
        self.parts: list[str] = parts
        self.uuids: list[str] | None = uuids

        self.block_size: tuple[int, int] | None = block_size
        self.block_step: int | None = block_step

        assert bool(self.block_size) == bool(
            self.block_step
        ), "Block size and step should be both set or both None"

        if not self.is_blockify:
            self.N_blocks = 1

        self.chanelling: bool = channeling
        self.pad: int = padding

        self.cpus: int = cpus

        self.union_size: int = 0

        self.subsets: list[Subset] | None = []
        self.subtypes: list[str] | None = []
        self.notations: list[str] | None = []

        self.indexes: list[tuple[str, int]] | None = []
        self.imgs: list[tuple[np.ndarray, ...]] | None = []
        self.exifs: dict[tuple[str, int], list[dict[str, Any]]] | None = []

    @property
    def N_blocks(self) -> int | None:
        if not self._N_blocks:
            LOGGER.warning("N_blocks is not set yet, run load method first")
        return self._N_blocks

    @N_blocks.setter
    def N_blocks(self, value: int) -> None:
        self._N_blocks = value

    @property
    def name(self) -> str:
        return self.__str__()

    @property
    def expected_size(self) -> int:
        if not self.union_size:
            LOGGER.warning("Union size is not set")
        return self.union_size * len(self.parts) * self.N_blocks

    @property
    def is_blockify(self) -> bool:
        return bool(self.block_size)

    @property
    def is_loaded(self) -> bool:
        return bool(self.indexes) and bool(self.imgs)

    def __str__(self) -> str:
        return "_".join(
            map(str, ["x".join(self.block_size), self.amount, "+".join(self.parts)])
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self) -> Iterator[tuple[int, tuple[np.ndarray, ...]]]:
        for index, imgs in zip(self.indexes, self.imgs):
            yield index, imgs, self.exifs[index[:2]]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[Any, ...], tuple[np.ndarray, ...], list[dict[str, Any]]]:
        index_ = self.indexes[index]
        return index_, self.imgs[index], self.exifs[index_[:2]]

    def load(
        self,
        subsets: list[Subset],
        subtypes: list[str] | None = None,
        notations: list[str] | str | None = None,
        **kwargs,
    ) -> None:
        start_call = time.time()

        subtypes = subtypes or [list(subset.files.keys())[-1] for subset in subsets]
        self.verify_subsets(subsets, subtypes)

        self.subsets = subsets
        self.subtypes = subtypes
        self.notations = notations or list("ytxfmwh")[: len(subsets)]

        union = Subset.union(
            self.subsets, self.subtypes, uuids=self.uuids, rescale=True, **kwargs
        )
        scale_factor = kwargs.pop("scale_to", max([s.scale_factor for s in subsets]))
        self.union_size = len(union)
        indexes, files = list(map(tuple, union.keys())), list(union.values())

        self.exifs = {
            i: dict(
                sorted(
                    {
                        f"{k}_{n}": v
                        for n, exif in zip(self.notations, [f.exif for f in fpair])
                        for k, v in exif.items() if exif
                    }.items()
                )
            )
            for i, fpair in zip(indexes, files)
        }

        # extract parts of CDP from objects
        imgs = zip(*files)
        imgs = [
            utils.flatten(
                [
                    *map(
                        lambda c: [
                            (
                                x[self.pad : -self.pad, self.pad : -self.pad]
                                if self.pad != 0
                                else x
                            )
                            for x in c.entity.get_parts(self.parts)
                        ],
                        imgs_,
                    )
                ]
            )
            for imgs_ in imgs
        ]
        indexes = [index + (part,) for index, part in product(indexes, self.parts)]

        if self.is_blockify:
            N_before = len(imgs[0])
            imgs = [*map(partial(self.blockify_imgs), imgs)]
            N_after = len(imgs[0])
            self.N_blocks = N_after // N_before

        if self.chanelling:
            imgs = [*map(lambda x: [self.channelize(y, scale_factor) for y in x], imgs)]

        indexes = [
            index + (block_i + 1,)
            for index, block_i in product(indexes, range(self.N_blocks))
        ]
        imgs = list(zip(*imgs))

        self.indexes = indexes
        self.imgs = imgs
        self.uuids = set([index[0] for index in indexes])
        self.verify_result()

        LOGGER.debug(f"Loader took {time.time() - start_call:.2f} seconds")

    def __call__(self, *args, **kwargs) -> None:
        return self.load(*args, **kwargs)

    def to_dataset(
        self, transform_func: Callable | None = None
    ) -> tuple[list[int], list[dict[str, Any]], set[str]]:
        ids = list(range(len(self)))
        dicts = []
        uuids = []
        for index, imgs, exif in self:
            d = dict(zip(self.headers, index))
            uuids.append(d["uuid"])
            if transform_func:
                imgs = [*map(transform_func, imgs)]
            d.update({notation: img for notation, img in zip(self.notations, imgs)})
            d.update(exif)
            dicts.append(d)
        return ids, dicts, set(uuids)

    def blockify_imgs(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        return utils.flatten(
            Parallel(n_jobs=self.cpus)(
                delayed(
                    partial(
                        utils.img2blocks,
                        block_size=self.block_size,
                        step=self.block_step,
                    )
                )(img)
                for img in imgs
            )
        )

    @staticmethod
    def channelize(img: np.ndarray, scale_factor: int) -> np.ndarray:
        s = int(img.shape[0] / scale_factor)
        return (
            utils.normalize_img(img)
            .reshape(s, scale_factor, s, scale_factor)
            .transpose(0, 2, 1, 3)
            .reshape(s, s, scale_factor**2)
        )

    @classmethod
    def verify_subsets(cls, subsets: list[Subset], subtypes: list[str]) -> None:
        for subset, subtype in zip(subsets, subtypes):
            assert subset.is_loaded, f"Subset {subset} is not loaded"
            assert subtype in subset.files, f"Subset {subset} has no {subtype} files"

    def verify_result(self) -> None:
        assert len(self.indexes) == len(
            self.imgs
        ), f"Indexes and images have different lengths: {len(self.indexes)} != {len(self.imgs)}"
        assert (
            len(self.indexes) == self.expected_size
        ), f"Indexes have wrong length: {len(self.indexes)} != expected {self.expected_size}"

    @property
    def info(self) -> dict[str, Any]:
        if not self.is_loaded():
            LOGGER.warning("Loader is not loaded yet")
            return {}

        return {
            "N_labels": self.uuids,
            "parts": self.parts_names,
            "block_size": self.block_size,
            "block_step": self.block_step,
            "blocks_N": self.N_blocks,
            "cross_label": self.cross_label,
            "cross_part": self.cross_part,
            "total": self.expected_size,
            "scale_common": max([folder.scale_factor for folder in self.subsets]),
        }

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([dict(zip(self.headers, index)) | exif for index, _, exif in self])
