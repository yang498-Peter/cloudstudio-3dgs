"""Dataset adapters: ``detect(path) -> bool`` and ``load(path) -> DatasetBundle``.

Detection is cheap and structural (which files exist), never content-based, so
that a malformed dataset is *detected* and then *rejected with a specific
message* rather than silently falling through to another adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from ..errors import DatasetDetectionError
from . import colmap, pinhole_folder, s1_fisheye


class DatasetAdapter(Protocol):
    NAME: str
    VERSION: str

    @staticmethod
    def detect(path: Path) -> bool: ...

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Any: ...


#: Order matters only for the error report; detection must stay mutually exclusive.
ADAPTERS: tuple[Any, ...] = (s1_fisheye, colmap, pinhole_folder)


def adapter_by_name(name: str) -> Any:
    for adapter in ADAPTERS:
        if adapter.NAME == name:
            return adapter
    known = ", ".join(adapter.NAME for adapter in ADAPTERS)
    raise DatasetDetectionError(f"unknown adapter '{name}'; known adapters: {known}")


def detect_adapter(path: Path, *, candidates: Sequence[Any] | None = None) -> Any:
    """Return the single adapter that claims ``path``.

    Raises :class:`DatasetDetectionError` when none or several claim it; the
    message lists what each adapter looked for, because "unsupported dataset"
    on its own costs the caller an hour.
    """

    path = Path(path)
    if not path.is_dir():
        raise DatasetDetectionError(f"dataset path is not a directory: {path}")
    pool = tuple(candidates) if candidates is not None else ADAPTERS
    claimed = [adapter for adapter in pool if adapter.detect(path)]
    if len(claimed) == 1:
        return claimed[0]
    if not claimed:
        report = "; ".join(f"{a.NAME} wants {a.REQUIRES}" for a in pool)
        raise DatasetDetectionError(
            f"no adapter recognises {path}. Expected one of: {report}"
        )
    names = ", ".join(adapter.NAME for adapter in claimed)
    raise DatasetDetectionError(
        f"{path} matches more than one adapter ({names}); pass the adapter name explicitly"
    )


def load_dataset(path: Path, *, adapter: str | None = None, **kwargs: Any) -> Any:
    """Detect (or take) an adapter and load the bundle."""

    chosen = adapter_by_name(adapter) if adapter else detect_adapter(Path(path))
    return chosen.load(Path(path), **kwargs)


__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "adapter_by_name",
    "colmap",
    "detect_adapter",
    "load_dataset",
    "pinhole_folder",
    "s1_fisheye",
]
