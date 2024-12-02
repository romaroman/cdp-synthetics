import numpy as np
import cv2 as cv
from omegaconf import DictConfig

from cdpsynt.common import LOGGER
from cdpsynt.data import UUIDCDP, DatasetNN, Loader
from cdpsynt.data.subset import Subset

from .phone import load_phone
from .scanner import load_scanner

__all__ = ["load_dataset"]


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


def calculate_step_size(
    scale_factor: int, base_image_size: int = 228, block_size: int = 128
):
    image_size = base_image_size * scale_factor

    num_steps = np.ceil((image_size - block_size) / block_size)
    step_size = np.floor((image_size - block_size) / num_steps)
    step_size = (step_size // scale_factor) * scale_factor

    return int(step_size)


def load_dataset(cfg: DictConfig, subsets: list[Subset] | None = None) -> DatasetNN:
    if subsets is None:
        subsets = load_scanner(cfg) if cfg.dataset.device == "Epson" else load_phone(cfg)

    scale_to = cfg.dataset.scale_to
    if scale_to == "auto":
        scale_to = max(map(lambda x: x.scale_factor, subsets.values()))
        LOGGER.info(f"Image loading autoscaling is set to: {scale_to}")
    block_size = cfg.dataset.block_size
    block_step = cfg.dataset.block_step
    if block_step == "auto":
        block_step = calculate_step_size(scale_to, block_size=block_size)
        LOGGER.info(f"Image block splitting step is set to: {block_step}")

    loader = Loader(
        parts=cfg.dataset.parts,
        block_size=(block_size, block_size),
        block_step=block_step,
        padding=cfg.dataset.padding,
        channeling=cfg.dataset.channeling,
    )

    loader.load(subsets.values(), notations=list(subsets.keys()), scale_to=scale_to)
    ids, dicts, uuids_loaded = loader.to_dataset(transform_func=normalize_img)

    uuids = cfg.dataset.uuid
    dataset = DatasetNN(
        ids,
        dicts,
        uuids_loaded,
        train_uuids=UUIDCDP.range(*uuids.train),
        test_uuids=UUIDCDP.range(*uuids.test),
        val_uuids=UUIDCDP.range(*uuids.val),
        batch_size=cfg.train.batch_size,
        repeats=1, # if cfg.mode == "test" else 2,
        shuffle=cfg.dataset.shuffle,
        flip_n_rotate=cfg.mode == "train",
    )

    dataset.setup("init")
    return dataset
