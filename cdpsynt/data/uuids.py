from __future__ import annotations

from .. import utils as pu

__all__ = ["UUID", "UUIDCDP"]


class UUID:
    pass


class UUIDCDP(UUID):

    cdp_per_row: int = 12
    cdp_per_column: int = 12
    cdp_per_page: int = cdp_per_column * cdp_per_row

    def __init__(self, id: int | str) -> UUIDCDP:
        self.id: int = int(id)

    def __str__(self) -> str:
        return pu.zfill_n(self.id, 6)

    @property
    def page(self) -> int:
        return (self.id - 1) // self.cdp_per_page + 1

    @property
    def column(self) -> int:
        return ((self.id - 1) % self.cdp_per_page) % self.cdp_per_row + 1

    @property
    def row(self) -> int:
        return ((self.id - 1) % self.cdp_per_page) // self.cdp_per_row + 1

    @property
    def position(self) -> tuple[int, int, int]:
        return self.page, self.row, self.column

    @staticmethod
    def range(start: int, end: int, step: int = 1) -> set[str]:
        return set(map(lambda x: f"{x:06}", range(start, end, step)))
