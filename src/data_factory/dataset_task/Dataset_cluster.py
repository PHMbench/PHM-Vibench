from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import Dataset


class IdIncludedDataset(Dataset):
    """Flatten per-file datasets while preserving every sample's source file ID."""

    def __init__(self, dataset_dict, metadata=None):
        if not isinstance(dataset_dict, Mapping) or not dataset_dict:
            raise ValueError(
                "IdIncludedDataset requires a non-empty mapping of file IDs to datasets."
            )

        self.dataset_dict = dict(dataset_dict)
        self.file_windows_list: list[dict[str, object]] = []
        self.metadata = metadata

        for file_id, original_dataset in self.dataset_dict.items():
            if original_dataset is None:
                raise ValueError(
                    f"Selected file_id={file_id!r} has no dataset object. "
                    "Fix dataset construction instead of skipping the file."
                )
            sample_count = len(original_dataset)
            if sample_count == 0:
                raise ValueError(
                    f"Selected file_id={file_id!r} produced zero samples. "
                    "Fix windowing or split configuration instead of skipping the file."
                )

            for window_id in range(sample_count):
                self.file_windows_list.append(
                    {"file_id": file_id, "window_id": window_id}
                )

        self._total_samples = len(self.file_windows_list)
        if self._total_samples == 0:
            raise ValueError("IdIncludedDataset produced zero samples.")

    def __len__(self):
        return self._total_samples

    def expected_dataset_names(self) -> tuple[str, ...]:
        """Declare this dataset's Name-level populations without reading samples.

        Only selected file IDs participate. Multiple Dataset_id values may intentionally
        share one Name and retain the existing pooled metric namespace.
        """
        if not self.dataset_dict:
            raise ValueError("Evaluation requires a non-empty selected test population.")
        names: set[str] = set()
        for file_id in self.dataset_dict:
            try:
                name = self.metadata[file_id]["Name"]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    f"Cannot resolve evaluation Name for selected file_id={file_id!r}."
                ) from exc
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError(
                    f"Evaluation Name for file_id={file_id!r} must be a non-empty "
                    f"string without surrounding whitespace, got {name!r}."
                )
            names.add(name)
        return tuple(sorted(names))

    def get_file_windows_list(self):
        return self.file_windows_list

    def get_file_id(self, global_idx):
        return self.file_windows_list[global_idx]["file_id"]

    def __getitem__(self, global_idx):
        if global_idx < 0 or global_idx >= self._total_samples:
            raise IndexError(
                f"Global index {global_idx} is outside [0, {self._total_samples})."
            )

        sample_info = self.file_windows_list[global_idx]
        file_id = sample_info["file_id"]
        window_id = sample_info["window_id"]
        original_dataset = self.dataset_dict[file_id]
        output = original_dataset[window_id]
        if not isinstance(output, dict):
            raise TypeError(
                f"Dataset for file_id={file_id!r} must return a mapping, "
                f"got {type(output).__name__}."
            )

        result = dict(output)
        result["file_id"] = file_id
        return result
