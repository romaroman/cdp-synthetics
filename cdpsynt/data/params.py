from __future__ import annotations

from .. import utils

__all__ = [
    "Origin",
    "Printer",
    "Substrate",
    "Device",
]


class EnumParam(utils.CustomEnum):
    @property
    def as_official(self) -> str:
        return self.value

    @property
    def as_short(self) -> str:
        return self.name


class Printer(EnumParam):
    HPI55 = "HP Indigo 5500"
    HPI76 = "HP Indigo 7600"


class Substrate(EnumParam):
    InvercoteG = "Invercote G"
    AlgroDesign = "Algro Design"
    GalerieImage = "Galerie Image"
    ConquerorStonemarque = "Conqueror Stonemarque"
    Atelier = "Atelier"


class Origin(EnumParam):
    orig = "Original"
    fake = "Fake"
    fake_estimation = "Fake estimation"
    fake_copymachine = "Fake copymachine"
    fake_synthetic = "Fake synthetic"
    fake_adversarial = "Fake adversarial"

    @property
    def is_fake(self) -> bool:
        return str(self).lower().startswith("fake")

    @property
    def is_original(self) -> bool:
        return not self.is_fake()


class Device(EnumParam):
    EpsonV850 = "Epson Perfection V850 Pro"
    iPhoneXS = "iPhone XS"
    iPhone12Pro = "iPhone 12 Pro"
    iPhone12ProMax = "iPhone 12 Pro Max"
    iPhone14Pro = "iPhone 14 Pro"
    iPhone14ProMax = "iPhone 14 Pro Max"
    iPhone15Pro = "iPhone 15 Pro"
    iPhone15ProMax = "iPhone 15 Pro Max"
