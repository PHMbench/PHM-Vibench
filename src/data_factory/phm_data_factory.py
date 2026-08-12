"""PHMFactory training adapter backed by ``phm-data-factory`` v0.2."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from .data_factory import data_factory, register_data_factory
from .data_utils import MetadataAccessor
from .standalone import build_data_repository


class RepositorySignalMapping(Mapping[Any, np.ndarray]):
    """Read dense arrays lazily through the provider's stable training API."""

    def __init__(self, repository: Any, sample_ids: Sequence[Any]):
        self._repository = repository
        self._sample_ids = tuple(sample_ids)

    def __getitem__(self, sample_id: Any) -> np.ndarray:
        return np.asarray(self._repository.read_signal(sample_id))

    def __iter__(self) -> Iterator[Any]:
        return iter(self._sample_ids)

    def __len__(self) -> int:
        return len(self._sample_ids)


@register_data_factory("phm_data")
class PHMDataFactory(data_factory):
    """Reuse PHMFactory splits and loaders over the optional provider."""

    def __init__(self, args_data: Any, args_task: Any):
        self._repository: Any | None = None
        self._closed = False
        try:
            super().__init__(args_data, args_task)
        except BaseException:
            self.close()
            raise

    def _init_metadata(self, args_data: Any) -> MetadataAccessor:
        self._repository = build_data_repository(args_data)
        frame = self._repository.metadata_frame("phm_vibench_v1")
        if "Id" not in frame.columns:
            raise ValueError("phm-data-factory metadata profile has no Id column")
        return MetadataAccessor(frame, key_column="Id")

    def _init_data(self, args_data: Any, **_: Any) -> RepositorySignalMapping:
        del args_data
        if self._repository is None:
            raise RuntimeError("phm-data-factory repository was not initialized")
        target_metadata = self.search_dataset_id()
        return RepositorySignalMapping(self._repository, target_metadata.keys())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._repository is not None:
            self._repository.close()

    def __enter__(self) -> "PHMDataFactory":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
