from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, ClassVar

import exiftool
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .. import utils
from ..common import DIR_DATA, LOGGER
from .entity import *
from .params import Device, Origin, Printer, Substrate

__all__ = [
    "Subset",
    "SubsetFile",
    "SubsetTemplateCDP",
    "SubsetAcq",
    "SubsetScanner",
    "SubsetPhoneWIFS",
    "SubsetAny",
]

try:
    exif_tool = exiftool.ExifToolHelper()
except:
    exif_tool = None
    LOGGER.warning("ExifTool is not installed")


@dataclass
class SubsetFile:
    path: Path
    entity: Entity
    exif: dict[str, Any]
    multishot: bool = False

    @property
    def uuid(self) -> str:
        return self.path.parent.name if self.multishot else self.path.stem

    @property
    def shot(self) -> int:
        return int(self.path.stem) if self.multishot else 1

    @property
    def name(self) -> str:
        return f"{self.uuid}/{self.shot} {str(self.entity)}"

    @staticmethod
    def from_file(
        path: Path,
        entity_type: type,
        path_exif: Path | dict[str, Any] | None = None,
        multishot: bool = False,
    ) -> SubsetFile:
        exif = {}
        if not path_exif is None:
            if isinstance(path_exif, dict):
                exif = deepcopy(path_exif)
            elif exif_tool is not None and (path_exif := path_exif.resolve()).exists():
                exif = exif_tool.get_metadata(path_exif)

        return SubsetFile(path, entity_type(utils.imread(Path(path).resolve(), -1)), exif, multishot)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def path_to_uuid_and_shot(
        cls, path: Path, multishot: bool = False
    ) -> tuple[str, str]:
        path = Path(path)
        return (path.parent.name, int(path.stem)) if multishot else (path.stem, 1)


class Subset(ABC):
    pattern_default: ClassVar[str] = r"^[0-9]{4,6}\.tiff$"
    patterns: ClassVar[dict[str, str]] = {
        "raw": "^.*\.(tiff|tif|png|jpeg|jpg|JPG|DNG|dng|HEIC)?$"
    }

    def __init__(self, path: Path, entity_type: type, multishot: bool) -> Subset:
        self.path: Path = path
        self.entity_type: type = entity_type
        self.multishot: bool = multishot

        self.paths: dict[str, list[Path]] = {}
        self.files: dict[str, list[SubsetFile]] = {}

        self.discover()

    @property
    def path(self) -> Path:
        return self.__path

    @path.setter
    def path(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("Given path doesn't exist")
        self.__path: Path = path

    @staticmethod
    def auto(path: Path, entity_type: type, multishot: bool) -> SubsetAcq:
        path = Path(path)
        origin = path.parents[1].name
        usecase = path.parents[2].name

        if (usecase.find("wifs") != -1 or origin.find("wifs") != -1) and str(path).find(
            "phone"
        ) != -1:
            return SubsetPhoneWIFS(path, entity_type, multishot)
        if origin.find("scan") != -1:
            return SubsetScanner(path, entity_type, multishot)
        else:
            raise ValueError(f"Subset {path} is not valid")

    @property
    @abstractmethod
    def ftype(self) -> str:
        pass

    @property
    @abstractmethod
    def sdict(self) -> dict[str, Any]:
        return {}

    @property
    def is_loaded(self) -> bool:
        return bool(self.files)

    @classmethod
    def is_phone(cls, subset_in: Subset) -> bool:
        return any([isinstance(subset_in, cl) for cl in [SubsetPhoneWIFS]])

    @classmethod
    def is_scanner(cls, subset_in: Subset) -> bool:
        return isinstance(subset_in, SubsetScanner)

    def discover(self, subtypes: list[str] | None = None) -> None:
        discover_paths = lambda path: sorted(
            utils.globre(
                path=path, pattern=self.patterns.get(subtype, self.pattern_default)
            ),
            key=lambda x: str(x),
        )
        subtypes = subtypes or self.subtypes

        for subtype in subtypes:
            dir_subtype = self.path / subtype
            if not dir_subtype.exists():
                continue

            self.paths[subtype] = discover_paths(dir_subtype)

    def load(
        self,
        subtype: str,
        exif: bool = False,
        uuids: list[int | None] = None,
        shots: list[int | None] = None,
        reload: bool = True,
        cpus: int = 1,
        ignore_excluded: bool = False,
    ) -> None:
        if subtype not in self.subtypes:
            raise ValueError(f"Subtype {subtype} has no files in the directory")
        if subtype in self.files and not reload:
            LOGGER.info("Files are already loaded")
            return
        if reload:
            self.files[subtype] = []

        if not reload and self.files[subtype]:
            LOGGER.info("Files are already loaded, use reload=True to reload")
            return

        LOGGER.info(f"Loading files from {str(self)} in {cpus} threads")

        uuids_all = sorted(self.uuids(subtype))
        if uuids:
            uuids_all = list(set(uuids_all) & set(uuids))
        shots_all = sorted(self.shots(subtype))
        if shots:
            shots_all = list(set(shots_all) & set(shots))

        paths_selected = []
        for path in self.paths[subtype]:
            uuid, shot = SubsetFile.path_to_uuid_and_shot(path, self.multishot)
            if uuid in uuids_all and shot in shots_all:
                paths_selected.append(path)

        paths_selected = sorted(paths_selected)
        if ignore_excluded:
            paths_selected = list(set(paths_selected) - set(self.ignored))

        exifs = repeat(None) if not exif or not Subset.is_phone(self) else self.get_exifs(paths_selected, subtype)

        self.files[subtype] = Parallel(n_jobs=cpus)(
            delayed(SubsetFile.from_file)(
                path, self.entity_type, path_exif, self.multishot
            )
            for path, path_exif in tqdm(
                zip(paths_selected, exifs), total=len(paths_selected), desc="Loading"
            )
        )

    def get_exifs(self, paths: list[Path], subtype: str) -> list[Path] | list[dict[str, Any]]:
        exifs = []
        for path in paths:
            path_raw = Path(str(path).replace(subtype, f"links/{subtype}")).resolve()
            if path_raw.exists() and path_raw.name in self.exifs:
                if isinstance(self, SubsetPhoneWIFS):
                    exifs.append(self.exifs[path_raw.name])
                else:
                    exifs.append(path_raw)
        return exifs if exifs else repeat(None)

    @property
    def ignored(self) -> list[Path]:
        path_file = self.path / "ignored.txt"
        if path_file.exists():
            with open(path_file, "r") as f:
                return [self.path / line.strip() for line in f.readlines()]
        else:
            LOGGER.info(
                f"List of ignored files isn't found in the root of the subset\n{str(path_file)}"
            )
            return []

    def uuids(self, subtype: str) -> list[int]:
        return list(
            set(
                map(
                    lambda p: p.parent.name if self.multishot else p.stem,
                    self.paths[subtype],
                )
            )
        )

    def shots(self, subtype: str) -> list[int]:
        return (
            list(set(map(lambda p: int(p.stem), self.paths[subtype])))
            if self.multishot
            else [1]
        )

    @property
    def uuids_loaded(self) -> list[int]:
        return sorted(list(set(map(lambda p: p.uuid, next(iter(self.files.values()))))))

    @property
    def shots_loaded(self) -> list[int]:
        return list(set(map(lambda p: p.shot, next(iter(self.files.values())))))

    @property
    def paths_loaded(self) -> list[int]:
        return list(map(lambda p: p.path, next(iter(self.files.values()))))

    def as_short(self, separator: str = " | ") -> str:
        return separator.join(
            [self.path.parent.parent.name, self.path.parent.name, self.path.name]
        )

    def __str__(self) -> str:
        return self.as_short("/")

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def union(
        subsets: list[Subset],
        subtypes: list[str],
        shuffle: bool = False,
        uuids: list[int | None] = None,
        seed: int = 42,
        reload: bool = False,
        rescale: bool = False,
        scale_to: int | None = None,
    ) -> dict[str, list[np.ndarray]]:
        if reload:
            [
                subset.load(subtype=subtype, reload=True, uuids=uuids)
                for subset, subtype in (subsets, subtypes)
            ]

        if rescale:
            if scale_to:
                scales = [scale_to / s.scale_factor for s in subsets]
            else:
                scales = [
                    max([s.scale_factor for s in subsets]) / s.scale_factor
                    for s in subsets
                ]
            for subset in subsets:
                subset.scale_factor = max(scales)
        else:
            scales = [1] * len(subsets)

        uuids = [set(uuids)] if uuids else []
        uuids_all = set.intersection(
            *[set(subset.uuids_loaded) for subset, subtype in zip(subsets, subtypes)]
            + uuids
        )
        if shuffle:
            random.Random(seed).shuffle(uuids_all)

        results = {}

        for uuid in uuids_all:
            files = []
            for subset, subtype, scale in zip(subsets, subtypes, scales):
                files_subset = deepcopy(
                    sorted(
                        filter(lambda f: f.uuid == uuid, subset.files[subtype]),
                        key=lambda f: f.path,
                    )
                )
                if scale != 1:
                    [f.entity.apply(utils.imscale, scale=scale) for f in files_subset]
                files.append(files_subset)

            files = utils.cycle_pad_lists(files)
            for shot, files_uuid in enumerate(zip(*files), start=1):
                results[(uuid, shot)] = files_uuid

        return dict(sorted(results.items()))

    def info(
        self, suffix: str | None = None, include_path: bool = False
    ) -> dict[str, str]:
        info_ = self.sdict
        if include_path:
            info_["Path"] = str(self.path)

        for k, v in info_.items():
            if type(v) not in [int, float, str]:
                info_[k] = str(v)

        if suffix:
            info_ = {f"{k}{suffix}": v for k, v in info_.items()}
        return info_

    @staticmethod
    def infodiff(dirs: list[Subset], separator: str = " | ") -> tuple[list[str], str]:
        df = pd.DataFrame.from_records([d.sdict for d in dirs])
        df["Fake nature"].replace("", np.nan, inplace=True)
        cols = list(filter(lambda col: df[col].nunique() > 1, df.columns))
        ds = list(map(lambda x: x[1].to_dict(), df[cols].iterrows()))
        labels = [separator.join(map(str, d.values())) for d in ds]
        legend = separator.join(map(str, ds[0].keys()))
        return labels, legend


class SubsetTemplateCDP(Subset):

    DIR_TMPL_CDP = DIR_DATA / "orig_template"

    def __init__(
        self, multishot: bool = False, subtypes: list[str] | None = None
    ) -> SubsetTemplateCDP:
        self.subtypes: list[str] = subtypes or ["cdp", "rcod"]
        super().__init__(
            path=self.DIR_TMPL_CDP, entity_type=EntityCDP, multishot=multishot
        )
        self.scale_factor = 1

    @property
    def ftype(self) -> str:
        return "template"

    @property
    def sdict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def get_template(cls, uuid: str, subdir: str = "cdp") -> np.ndarray:
        return utils.imread(cls.DIR_TMPL_CDP / f"{subdir}/{uuid}.tiff", -1)


subtypes_acquisition: dict[type, str] = {
    EntityCDP: [
        "raw",
        "cdp",
        "rcod",
    ],
}


class SubsetAcq(Subset):
    def __init__(self, path: Path, entity_type: type, multishot: bool) -> None:
        self.subtypes: list[str] = subtypes_acquisition[entity_type]
        super().__init__(path, entity_type, multishot)

        self.scale_factor: int = 1

        self.set_origin_params(self.path.parent.parent.name)
        self.set_manufacturing_params(self.path.parent.name)
        self.set_acquisition_params(self.path.name)

    @property
    def ftype(self) -> str:
        return "acquisition"

    @property
    def sdict(self) -> dict[str, Any]:
        return {
            "Origin": self.origin.as_official,
            "Printer": self.printer.as_official,
            "Substrate": self.substrate.as_official,
            "Session": self.session,
            "PrintRun": self.print_run,
            "Fake": self.fake_origin,
        }

    def set_origin_params(self, subdir_origins: str) -> None:
        self.origin: Origin = Origin.match_enum(subdir_origins)

    def set_manufacturing_params(self, subdir_manufacture: str) -> None:
        parts = subdir_manufacture.split("_")

        self.printer: Printer = Printer.match_enum(parts[0])

        self.print_dpi: float = 812.8
        inc = 0

        if parts[1].startswith("printdpi"):
            self.print_dpi: float = utils.str2nmb(parts[1])
            inc = 1

        self.print_run: int = utils.str2nmb(parts[1 + inc])
        self.session: int = utils.str2nmb(parts[2 + inc])
        self.substrate: Substrate = Substrate.match_enum(parts[3 + inc])
        self.fake_origin: str = parts[4 + inc] if self.origin.is_fake else ""

    @abstractmethod
    def set_acquisition_params(self, subdir_acquisition: str) -> None:
        pass


class SubsetScanner(SubsetAcq):
    def __init__(self, path: Path, entity_type: type, multishot: bool) -> SubsetScanner:
        super().__init__(path, entity_type, multishot)

    def set_acquisition_params(self, subdir_acquisition: str) -> None:
        super().set_acquisition_params(subdir_acquisition)
        try:
            parts = subdir_acquisition.split("_")

            self.device: Device = Device.match_enum(parts[0])
            self.acquisition_run: int = utils.str2nmb(parts[1])
            self.scan_ppi: int = utils.str2nmb(parts[2])
            self.scale_factor: int = utils.str2nmb(parts[3])
            self.extra_info: str = " | ".join(parts[4:]) if len(parts) > 4 else ""
        except:
            self.set_acquisition_params_fallback(subdir_acquisition)

    def set_acquisition_params_fallback(self, subdir_acquisition: str) -> None:
        parts = subdir_acquisition.split("_")

        self.device: Device = Device.EpsonV850
        self.acquisition_run: int = utils.str2nmb(parts[0])
        self.scan_ppi: int = utils.str2nmb(parts[1])
        self.scale_factor: int = int(np.ceil(self.scan_ppi / self.print_dpi))
        self.extra_info: str = " | ".join(parts[2:]) if len(parts) > 2 else ""

    @property
    def sdict(self) -> dict[str, Any]:
        return super().sdict | {
            "Device": self.device.as_official,
            "AcquisitionRun": self.acquisition_run,
            "ScanPPI": self.scan_ppi,
            "ScaleFactor": self.scale_factor,
            "ExtraInfo": self.extra_info,
        }

    @property
    def ftype(self) -> str:
        return "scan"


class SubsetPhoneWIFS(SubsetAcq):
    def __init__(self, path: Path, entity_type: type, multishot: bool) -> None:
        super().__init__(path, entity_type, multishot)

    def set_acquisition_params(self, subdir_acquisition: str) -> None:
        super().set_acquisition_params(subdir_acquisition)

        parts = subdir_acquisition.split("_")

        self.device: Device = Device.match_enum(parts[0])
        self.owner: str = parts[1]
        self.acquisition_run: int = utils.str2nmb(parts[2])
        self.scale_factor: int = utils.str2nmb(parts[3])
        self.lens: str = parts[4]
        self.magnification: int = utils.str2nmb(parts[5])
        self.person: str = parts[6]
        self.illumination: str = parts[7]

        self.stego_info = dict()
        self.exifs = dict()
        self.load_stego_info()

    @property
    def sdict(self) -> dict[str, Any]:
        return super().sdict | {
            "Device": self.device.as_official,
            "Owner": self.owner,
            "AcquisitionRun": self.acquisition_run,
            "ScaleFactor": self.scale_factor,
            "Lens": self.lens,
            "Magnification": self.magnification,
            "Person": self.person,
            "Illumination": self.illumination,
        }

    def load_stego_info(self) -> None:
        self.stego_info = dict()
        self.exifs = dict()

        for jf in Path(self.path / "raw").glob("*.json"):
            with open(jf, "r") as f:
                json_d = json.load(f)
                photos_all = json_d.pop("photos")
                if not self.stego_info:
                    self.stego_info = deepcopy(json_d)
                for photo in photos_all:
                    name = photo.pop("name")
                    self.exifs[name] = photo

    @property
    def ftype(self) -> str:
        return "phone"


class SubsetAny(Subset):
    def __init__(
        self,
        path: Path,
        entity_type: type,
        multishot: bool,
        subtypes: list[str] | None = None,
        scale: float = 1,
    ) -> SubsetAny:
        self.subtypes: list[str] = subtypes or [
            s.name for s in sorted(Path(path).resolve().glob("*")) if s.is_dir()
        ]
        if not self.subtypes:
            raise ValueError(f"No subtypes found in {path} directory")
        LOGGER.info(f"Discovered subtypes: {self.subtypes} for {path} directory")

        self.scale_factor: float = scale

        super().__init__(path, entity_type, multishot)

    @property
    def ftype(self) -> str:
        return "any"

    @property
    def sdict(self) -> dict[str, Any]:
        return {
            "Subtypes": self.subtypes,
            "ScaleFactor": self.scale_factor,
        }

    def __str__(self) -> str:
        return str(self.path)
