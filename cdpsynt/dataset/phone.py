from pathlib import Path

from omegaconf import DictConfig

from cdpsynt.common import LOGGER
from cdpsynt.data import (UUIDCDP, EntityCDP, Subset, SubsetPhoneWIFS,
                          SubsetTemplateCDP)

__all__ = ["load_phone"]

acq_opts = {
    "iPXS": "iPhoneXS_LAB_run1_scale3_wide_2x_RC_office",
    "iP12w": "iPhone12Pro_LAB_run1_scale3_wide_2x_RC_office",
    "iP14w": "iPhone14Pro_LAB_run1_scale2_wide_3x_RC_office",
    "iP14uw": "iPhone14Pro_LAB_run1_scale5_uwide_2.5x_RC_overhead",
    "iP15w": "iPhone15ProMax_RC_run1_scale2_wide_2.5x_RC_office",
    "iP15uw": "iPhone15ProMax_RC_run1_scale5_uwide_2.5x_RC_lamp",
}


def form_subset_list(device: str, printer: str, dir_root: Path, test: bool):
    acq = acq_opts[device]
    subsets = {
        "x": dir_root
        / "orig_phone"
        / f"HPI{printer}_printrun1_session2_InvercoteG"
        / acq,
        "f": dir_root
        / "fake_phone"
        / f"HPI{printer}_printrun1_session2_InvercoteG_EHPI{76 if printer == 55 else 55}"
        / acq,
    }
    if test:
        subsets["xref"] = (
            dir_root
            / "orig_phone"
            / f"HPI{printer}_printrun1_session2_InvercoteG"
            / acq
        )
    return subsets


def load_phone(cfg: DictConfig) -> dict[str, Subset]:
    dir_root = Path(cfg.io.dir_root)
    is_train = cfg.mode == "train"
    subdirs_subsets = form_subset_list(cfg.dataset.device, cfg.dataset.printer, dir_root, not is_train)

    if not cfg.dataset.fake:
        subdirs_subsets.pop("f")

    uuids = UUIDCDP.range(*cfg.dataset.uuid.all)
    subsets = {}
    for notation, subdir_subset in subdirs_subsets.items():
        subset_y = SubsetPhoneWIFS(subdir_subset, EntityCDP, True)
        shot = 1 if is_train else {"x": 2, "xref": 1}.get(notation, 1)

        LOGGER.info(f"Subset path: {subdir_subset}")
        LOGGER.info(f"Loading subset {notation} with shot {shot}")
        subset_y.load(
            cfg.dataset.subtype,
            ignore_excluded=True,
            shots=[shot],
            uuids=uuids,
            exif=False,
        )
        subsets[notation] = subset_y
        LOGGER.info(f"Loaded #{len(subset_y.uuids_loaded)} of uuids: {','.join(map(str, subset_y.uuids_loaded))}")

        if not uuids:
            uuids = sorted(subset_y.uuids_loaded)

    subset_t = SubsetTemplateCDP()
    subset_t.load(cfg.dataset.subtype, uuids=uuids)
    subsets["t"] = subset_t

    return subsets
