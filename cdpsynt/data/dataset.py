from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..common import LOGGER


__all__ = ["DatasetNN", "SubsetNN"]


class SubsetNN(Dataset):
    def __init__(
        self,
        ids: list[int],
        imgs: list[dict[str, Any]],
        repeats: int,
        flip_n_rotate: bool,
    ) -> SubsetNN:
        super().__init__()

        self.ids: list[int] = ids
        self.imgs: list[dict[str, Any]] = imgs
        self.repeats: int = repeats
        self.flip_n_rotate: bool = flip_n_rotate

    def __len__(self) -> int:
        return len(self.ids) * self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = index % len(self.ids)
        angle = np.random.choice([0, 0, 0, 90, 180, 270]) if self.flip_n_rotate else 0
        flip_horizontal = torch.rand(1) > 0.5 and self.flip_n_rotate
        flip_vertical = torch.rand(1) > 0.5 and self.flip_n_rotate

        imgs = deepcopy(self.imgs[index])
        for k in imgs.keys():
            if not isinstance(imgs[k], np.ndarray):
                continue

            imgs[k] = Image.fromarray(imgs[k])
            imgs[k] = transforms.functional.rotate(imgs[k], int(angle))

            if flip_horizontal:
                imgs[k] = transforms.functional.hflip(imgs[k])
            if flip_vertical:
                imgs[k] = transforms.functional.vflip(imgs[k])

            imgs[k] = transforms.ToTensor()(imgs[k])

        return imgs


class DatasetNN(LightningDataModule):
    def __init__(
        self,
        ids: list[int],
        dicts: list[dict[str, Any]],
        uuids: set[str],
        train_size: float | None = None,
        test_size: float | None = None,
        train_uuids: set[str] | None = None,
        test_uuids: set[str] | None = None,
        val_uuids: set[str] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        repeats: int = 1,
        flip_n_rotate: bool = False,
    ) -> DatasetNN:
        super().__init__()

        self.ids: list[int] = ids
        self.dicts: list[dict[str, Any]] = dicts
        self.uuids: set[str] = uuids

        self.train_size: float | None = train_size
        self.test_size: float | None = test_size
        self.val_size: float | None = 1 - train_size - test_size if self.train_size and self.test_size else None

        self.train_uuids: set[str] | None = train_uuids
        self.test_uuids: set[str] | None = test_uuids
        self.val_uuids: set[str] | None = val_uuids

        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.repeats: int = repeats
        self.flip_n_rotate: bool = flip_n_rotate

        self.num_workers: int = 8

    def setup(self, stage: str):
        select_imgs = lambda ids, imgs: [imgs[id] for id in ids]

        train_ids, val_ids, test_ids = self.get_ids()
        if self.shuffle:
            train_ids, val_ids, test_ids = map(
                lambda ids: np.random.permutation(ids), (train_ids, val_ids, test_ids)
            )

        train_imgs, val_imgs, test_imgs = (
            select_imgs(ids, self.dicts) for ids in (train_ids, val_ids, test_ids)
        )

        self.train_set = SubsetNN(train_ids, train_imgs, self.repeats, self.flip_n_rotate)
        self.val_set = SubsetNN(val_ids, val_imgs, self.repeats, self.flip_n_rotate)
        self.test_set = SubsetNN(test_ids, test_imgs, self.repeats, self.flip_n_rotate)

    def get_ids(self) -> tuple[list[int], ...]:
        if bool(self.train_uuids) != bool(self.test_uuids) != bool(self.val_uuids):
            raise ValueError(
                "Both train_uuids and test_uuids must be specified or none of them"
            )

        if self.train_uuids:
            LOGGER.info(
                "Since train_uuids and test_uuids are specified, train_size and test_size are ignored"
            )

            uuids_mapping = {
                "train": self.train_uuids,
                "test": self.test_uuids,
                "val": self.val_uuids
            }

            ids_dict = {key: [id_ for id_, dict_ in zip(self.ids, self.dicts) if dict_["uuid"] in uuids] 
                        for key, uuids in uuids_mapping.items()}

            train_ids, test_ids, val_ids = ids_dict["train"], ids_dict["test"], ids_dict["val"]
        else:
            N = len(self.ids)
            train_ids = self.ids[: int(N * self.train_size)]
            test_ids = self.ids[
                int(N * self.train_size) : int(N * (self.train_size + self.test_size))
            ]
            val_ids = self.ids[int(N * (self.train_size + self.test_size)) :]

        LOGGER.info(
            f"Train size: {len(train_ids)}, test size: {len(test_ids)}, val size: {len(val_ids)} imgs"
        )
        return train_ids, val_ids, test_ids

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def teardown(self, stage: str):
        pass
