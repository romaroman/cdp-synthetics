from pathlib import Path

from omegaconf import DictConfig

from cdpsynt.common import LOGGER
from cdpsynt.data import EntityCDP, Subset, SubsetScanner, SubsetTemplateCDP, UUIDCDP

__all__ = ["load_scanner"]


def form_subset_list(printer: str, dir_root: Path, test: bool):
    acq_ref = "EpsonV850_run1_scandpi2400_scale3"
    acq_ver = "EpsonV850_run2_scandpi2400_scale3"

    path_orig = dir_root / "orig_scan" / f"HPI{printer}_printrun1_session2_InvercoteG"
    path_fake = (
        dir_root
        / "fake_scan"
        / f"HPI{printer}_printrun1_session2_InvercoteG_EHPI{76 if printer == 55 else 55}"
    )

    subsets = {
        "x": path_orig / (acq_ver if test else acq_ref),
        "f": path_fake / (acq_ver if test else acq_ref),
    }

    if test:
        subsets["xref"] = path_orig / acq_ref

    return subsets


def load_scanner(cfg: DictConfig) -> dict[str, Subset]:
    dir_root = Path(cfg.io.dir_root)
    is_train = cfg.mode == "train"
    subdirs_subsets = form_subset_list(cfg.dataset.printer, dir_root, not is_train)

    if not cfg.dataset.fake:
        subdirs_subsets.pop("f")

    uuids = UUIDCDP.range(*cfg.dataset.uuid.all)
    subsets = {}
    for notation, subdir_subset in subdirs_subsets.items():
        subset_y = SubsetScanner(subdir_subset, EntityCDP, True)

        LOGGER.info(f"Subset path: {subdir_subset}")
        subset_y.load(
            cfg.dataset.subtype,
            ignore_excluded=True,
            uuids=uuids,
            exif=False,
        )
        subsets[notation] = subset_y

        uuids = set(subset_y.uuids_loaded) & uuids

    subset_t = SubsetTemplateCDP()
    subset_t.load(cfg.dataset.subtype, uuids=uuids)
    subsets["t"] = subset_t

    return subsets
