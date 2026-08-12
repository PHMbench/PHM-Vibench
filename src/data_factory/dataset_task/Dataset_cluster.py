from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import Dataset


class IdIncludedDataset(Dataset):
    """Flatten per-file datasets while preserving every sample's source file ID."""

    def __init__(self, dataset_dict, metadata=None, physical_group_by_id=None):
        if not isinstance(dataset_dict, Mapping) or not dataset_dict:
            raise ValueError(
                "IdIncludedDataset requires a non-empty mapping of file IDs to datasets."
            )

        self.dataset_dict = dict(dataset_dict)
        self.file_windows_list: list[dict[str, object]] = []
        self.metadata = metadata
        self.physical_group_by_id = physical_group_by_id

        if physical_group_by_id is not None:
            missing_groups = sorted(
                set(self.dataset_dict) - set(physical_group_by_id),
                key=str,
            )
            if missing_groups:
                raise ValueError(
                    "Physical group identity is missing for selected file ID(s) "
                    f"{missing_groups}."
                )

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
        if self.physical_group_by_id is not None:
            result["physical_group_id"] = self.physical_group_by_id[file_id]
        return result
