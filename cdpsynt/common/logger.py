from __future__ import annotations

from .config import LOG_LEVEL

from loguru import logger as LOGGER
from rich.logging import RichHandler
from tqdm.auto import tqdm

__all__ = ["LOGGER", "SET_LOG_LEVEL"]

LOGGER.remove()
LOGGER.add(lambda msg: tqdm.write(msg, end=""))

def SET_LOG_LEVEL(level: str) -> None:
    LOGGER.configure(
        handlers=[
            {
                "sink": RichHandler(markup=True),
                "format": "[red]{function}[/red] {message}",
                "level": level,
            }
        ],
    )

SET_LOG_LEVEL(LOG_LEVEL)
