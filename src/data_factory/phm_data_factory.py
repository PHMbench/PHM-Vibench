"""PHM-Vibench training factory backed by phm-data-factory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .data_factory import data_factory, register_data_factory
from .data_utils import MetadataAccessor
from .standalone import build_data_repository


class RepositorySignalMapping(Mapping[str, np.ndarray]):
    def __init__(self, repository: Any):
        self.repository = repository

    def __getitem__(self, sample_id: str | int) -> np.ndarray:
        return self.repository.read_signal(sample_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self.repository.metadata.keys())

    def __len__(self) -> int:
        return len(self.repository.metadata)

    def close(self) -> None:
        self.repository.close()


@register_data_factory("phm_data")
class phm_data_factory(data_factory):
    """Keep existing splits/Datasets/DataLoaders over a repository backend."""

    def __init__(self, args_data: Any, args_task: Any):
        self._repository = None
        self._closed = False
        super().__init__(args_data, args_task)

    def _init_metadata(self, args_data: Any) -> MetadataAccessor:
        self._repository = build_data_repository(args_data)
        frame = self._repository.metadata_frame("phm_vibench_v1")
        return MetadataAccessor(frame, key_column="Id")

    def _init_data(self, args_data: Any, **_: Any) -> RepositorySignalMapping:
        del args_data
        return RepositorySignalMapping(self._repository)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._repository is not None:
            self._repository.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
