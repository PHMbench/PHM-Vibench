import hashlib
import random
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset
from ..grouped_split import write_frozen_json
# Reference:UniTS


class IdIncludedDataset(Dataset):
    def __init__(self, dataset_dict, metadata=None):
        """
        包装一个 PyTorch Dataset 字典，使得每个样本都包含其原始ID。

        Args:
            dataset_dict (dict): 一个字典，键是字符串ID，值是 PyTorch Dataset 对象。
                                 例如：{'id1': train_dataset_for_id1, 'id2': train_dataset_for_id2}
                                 其中 train_dataset_for_id1 等实例的 __getitem__ 返回 (x, y)。
        """
        self.dataset_dict = dataset_dict # 保存对原始数据集字典的引用
        self.file_windows_list = [] # 用于全局索引到 (id, 原始数据集中的索引) 的映射
        self.metadata = metadata # 保存元数据，可能包含数据集的其他信息
        for file_id, original_dataset in self.dataset_dict.items():
            if original_dataset is None:
                print(f"警告: ID '{file_id}' 对应的 dataset 为 None，已跳过。")
                continue
            if len(original_dataset) == 0:
                print(f"警告: ID '{file_id}' 对应的 dataset 为空，已跳过。")
                continue
            # if not isinstance(file_id,str):
            #     print(f"警告: ID '{file_id}' 不是字符串，已跳过。")
            #     continue
            
            for window_id in range(len(original_dataset)): # 数据集id ，样本id； 样本id 当前数据集的id
                self.file_windows_list.append({'file_id': file_id, 'window_id': window_id}) # 1,2,3 | 1,2,3,4 ~ 1,2,3,4,5,6,7
        
        self._total_samples = len(self.file_windows_list) # 计算所有原始数据集的样本总数

    def __len__(self):
        """
        返回所有原始数据集中样本的总数。
        """
        return self._total_samples
    def get_file_windows_list(self):
        """
        获取文件窗口列表。

        Returns:
            list: 包含所有样本的文件窗口列表，每个元素是一个字典，包含 'file_id' 和 'Window_id'。
        """
        return self.file_windows_list
    def get_file_id(self, global_idx):
        """
        根据全局索引获取文件ID。

        Args:
            global_idx (int): 全局样本索引。

        Returns:
            str: 文件ID。
        """
        return self.file_windows_list[global_idx]['file_id']

    def __getitem__(self, global_idx):
        """
        根据全局索引获取样本，并返回 (id, (x, y))。

        Args:
            global_idx (int): 全局样本索引。

        Returns:
            tuple: (str, tuple), 即 (id, (x, y))
                   其中 x 是特征数据, y 是标签。
        """
        if global_idx < 0 or global_idx >= self._total_samples:
            raise IndexError(f"全局索引 {global_idx} 超出范围 (总样本数: {self._total_samples})")

        sample_info = self.file_windows_list[global_idx]

        file_id = sample_info['file_id']
        # dataset_id = self.metadata[data_id]['Dataset_id'] # 获取数据集的ID
        window_id_in_original_dataset = sample_info['window_id']

        # 从原始数据集中获取 (x, y)
        original_dataset_instance = self.dataset_dict[file_id]
        out = original_dataset_instance[window_id_in_original_dataset] # may be (x, y) or (x, y, z)

        out.update({"file_id": file_id, "window_id": window_id_in_original_dataset})
        return  out


class FrozenClassPairDataset(IdIncludedDataset):
    """Attach a deterministic within-group derangement for the 2D view.

    Pairing preserves the class, physical/source group, and partner marginal:
    every source receives a different window and every partner is used once.
    """

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
            raise ValueError("Frozen pairing requires protocol and split-manifest identifiers")
        self.mapping = self._build_mapping()
        manifest = self._manifest()
        write_frozen_json(manifest, Path(manifest_dir) / f"{self.split_name}.json")

    def __len__(self):
        return len(self.dataset)

    def _sample_key(self, index):
        info = self.dataset.file_windows_list[index]
        return f"{info['file_id']}:{info['window_id']}"

    def _label(self, index):
        file_id = self.dataset.file_windows_list[index]['file_id']
        return int(self.dataset.metadata[file_id]['Label'])

    def _identity(self, index):
        file_id = self.dataset.file_windows_list[index]['file_id']
        metadata = self.dataset.metadata[file_id]
        if self.group_key == 'FileParent':
            return str(PurePosixPath(str(metadata['File'])).parent)
        if self.group_key == 'Id':
            return str(file_id)
        if self.group_key not in metadata:
            raise ValueError(f"Pairing group key '{self.group_key}' is absent from metadata")
        return str(metadata[self.group_key])

    def _build_mapping(self):
        by_stratum = {}
        for index in range(len(self.dataset)):
            stratum = (self._label(index), self._identity(index))
            by_stratum.setdefault(stratum, []).append(index)

        mapping = {}
        for label, identity in sorted(by_stratum, key=lambda value: (value[0], value[1])):
            samples = sorted(by_stratum[(label, identity)], key=self._sample_key)
            if len(samples) < 2:
                raise ValueError(
                    f"Cannot derange split={self.split_name}, label={label}, "
                    f"group={identity}: fewer than two samples"
                )
            stratum_digest = hashlib.sha256(
                f"{self.split_name}:{label}:{identity}".encode('utf-8')
            ).digest()
            stratum_seed = self.seed ^ int.from_bytes(stratum_digest[:8], 'big')
            random.Random(stratum_seed).shuffle(samples)
            offset = 1 + stratum_seed % (len(samples) - 1)
            for position, source in enumerate(samples):
                mapping[source] = samples[(position + offset) % len(samples)]
        if any(index == partner for index, partner in mapping.items()):
            raise AssertionError("Frozen class-preserving pairing contains an identity pair")
        expected = set(range(len(self.dataset)))
        if set(mapping) != expected or set(mapping.values()) != expected:
            raise AssertionError("Frozen pairing does not cover every sample exactly once")
        if len(set(mapping.values())) != len(mapping):
            raise AssertionError("Frozen class-preserving pairing is not one-to-one")
        if any(self._identity(index) != self._identity(partner) for index, partner in mapping.items()):
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
        payload = {
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
            "maximum_partner_reuse": max(partner_counts.values()) if partner_counts else 0,
            "mapping_sha256": hashlib.sha256("\n".join(pairs).encode('utf-8')).hexdigest(),
        }
        return payload

    def __getitem__(self, index):
        out = dict(self.dataset[index])
        partner = self.dataset[self.mapping[index]]
        if int(out['y']) != int(partner['y']):
            raise AssertionError("Frozen pairing violated the same-class control")
        if self._identity(index) != self._identity(self.mapping[index]):
            raise AssertionError("Frozen pairing violated the same-group control")
        out['x_2d'] = partner['x']
        out['pair_source_file_id'] = partner['file_id']
        out['pair_source_window_id'] = partner['window_id']
        return out
