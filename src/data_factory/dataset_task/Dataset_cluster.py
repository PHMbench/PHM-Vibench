from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path, PurePosixPath
import random

from torch.utils.data import Dataset

from ..grouped_split import write_frozen_json


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
        result["window_id"] = window_id
        if self.physical_group_by_id is not None:
            result["physical_group_id"] = self.physical_group_by_id[file_id]
        return result


class FrozenClassPairDataset(IdIncludedDataset):
    """Attach a deterministic within-group derangement for the 2D view."""

    def __init__(
        self,
        dataset,
        *,
        seed: int,
        split_name: str,
        manifest_dir: str,
        group_key: str,
        protocol_id: str,
        split_manifest_sha256: str,
    ):
        if not isinstance(dataset, IdIncludedDataset):
            raise TypeError("FrozenClassPairDataset requires IdIncludedDataset")
        self.dataset = dataset
        self.dataset_dict = dataset.dataset_dict
        self.file_windows_list = dataset.file_windows_list
        self.metadata = dataset.metadata
        self._total_samples = len(dataset)
        self.seed = int(seed)
        self.split_name = str(split_name)
        self.group_key = str(group_key)
        self.protocol_id = str(protocol_id)
        self.split_manifest_sha256 = str(split_manifest_sha256)
        if not self.protocol_id or not self.split_manifest_sha256:
            raise ValueError(
                "Frozen pairing requires protocol and split-manifest identifiers"
            )
        self.mapping = self._build_mapping()
        write_frozen_json(
            self._manifest(), Path(manifest_dir) / f"{self.split_name}.json"
        )

    def __len__(self):
        return len(self.dataset)

    def _sample_key(self, index):
        info = self.dataset.file_windows_list[index]
        return f"{info['file_id']}:{info['window_id']}"

    def _label(self, index):
        file_id = self.dataset.file_windows_list[index]["file_id"]
        return int(self.dataset.metadata[file_id]["Label"])

    def _identity(self, index):
        file_id = self.dataset.file_windows_list[index]["file_id"]
        metadata = self.dataset.metadata[file_id]
        if self.group_key == "FileParent":
            return str(PurePosixPath(str(metadata["File"])).parent)
        if self.group_key == "Id":
            return str(file_id)
        if self.group_key not in metadata:
            raise ValueError(
                f"Pairing group key '{self.group_key}' is absent from metadata"
            )
        return str(metadata[self.group_key])

    def _build_mapping(self):
        by_stratum = {}
        for index in range(len(self.dataset)):
            stratum = (self._label(index), self._identity(index))
            by_stratum.setdefault(stratum, []).append(index)

        mapping = {}
        for label, identity in sorted(
            by_stratum, key=lambda value: (value[0], value[1])
        ):
            samples = sorted(by_stratum[(label, identity)], key=self._sample_key)
            if len(samples) < 2:
                raise ValueError(
                    f"Cannot derange split={self.split_name}, label={label}, "
                    f"group={identity}: fewer than two samples"
                )
            stratum_digest = hashlib.sha256(
                f"{self.split_name}:{label}:{identity}".encode("utf-8")
            ).digest()
            stratum_seed = self.seed ^ int.from_bytes(stratum_digest[:8], "big")
            random.Random(stratum_seed).shuffle(samples)
            offset = 1 + stratum_seed % (len(samples) - 1)
            for position, source in enumerate(samples):
                mapping[source] = samples[(position + offset) % len(samples)]
        if any(index == partner for index, partner in mapping.items()):
            raise AssertionError(
                "Frozen class-preserving pairing contains an identity pair"
            )
        expected = set(range(len(self.dataset)))
        if set(mapping) != expected or set(mapping.values()) != expected:
            raise AssertionError("Frozen pairing does not cover every sample exactly once")
        if len(set(mapping.values())) != len(mapping):
            raise AssertionError("Frozen class-preserving pairing is not one-to-one")
        if any(
            self._identity(index) != self._identity(partner)
            for index, partner in mapping.items()
        ):
            raise AssertionError("Frozen pairing crossed the configured group boundary")
        return mapping

    def _manifest(self):
        pairs = sorted(
            f"{self._sample_key(index)}->{self._sample_key(self.mapping[index])}"
            for index in self.mapping
        )
        partner_counts = {}
        for partner in self.mapping.values():
            partner_counts[partner] = partner_counts.get(partner, 0) + 1
        class_preserved_pairs = sum(
            self._label(index) == self._label(partner)
            for index, partner in self.mapping.items()
        )
        group_preserved_pairs = sum(
            self._identity(index) == self._identity(partner)
            for index, partner in self.mapping.items()
        )
        sample_count = len(self.mapping)
        return {
            "schema_version": 1,
            "mode": "frozen_within_group_class_derangement",
            "split": self.split_name,
            "seed": self.seed,
            "protocol_id": self.protocol_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "group_key": self.group_key,
            "samples": sample_count,
            "self_pairs": 0,
            "class_preserved_pairs": class_preserved_pairs,
            "class_preserved_fraction": (
                class_preserved_pairs / sample_count if sample_count else 1.0
            ),
            "group_preserved_pairs": group_preserved_pairs,
            "group_preserved_fraction": (
                group_preserved_pairs / sample_count if sample_count else 1.0
            ),
            "partner_bijection": len(partner_counts) == len(self.mapping),
            "unique_partner_samples": len(partner_counts),
            "maximum_partner_reuse": (
                max(partner_counts.values()) if partner_counts else 0
            ),
            "mapping_sha256": hashlib.sha256(
                "\n".join(pairs).encode("utf-8")
            ).hexdigest(),
        }

    def __getitem__(self, index):
        out = dict(self.dataset[index])
        partner_index = self.mapping[index]
        partner = self.dataset[partner_index]
        if int(out["y"]) != int(partner["y"]):
            raise AssertionError("Frozen pairing violated the same-class control")
        if self._identity(index) != self._identity(partner_index):
            raise AssertionError("Frozen pairing violated the same-group control")
        out["x_2d"] = partner["x"]
        out["pair_source_file_id"] = partner["file_id"]
        out["pair_source_window_id"] = partner["window_id"]
        return out
