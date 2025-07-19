import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from pytorch_lightning.utilities import CombinedLoader

# TODO balance id
# def balance_id

class ID_dataset(Dataset):
    """Return raw arrays and metadata by ID."""

    def __init__(self, metadata, args_data, args_task, mode="train"):
        """
        数据集类，管理数据ID。
        在数据集层面不进行具体的数据处理，只返回ID。
        Args:
            data: 输入数据，字典格式 {ID: 数据}
            metadata: 数据元信息，格式为 {ID: {字段: 值}} 字典
            args_data: 数据处理参数
            args_task: 任务参数
            mode: 数据模式，可选 "train", "valid", "test"
        Returns:
            id
            meta
        """

        self.metadata = metadata
        self.ids = list(self.metadata.keys())
        self.args_data = args_data
        self.args_task = args_task
        self.mode = mode

    def __len__(self):
        """返回数据集长度"""
        return len(self.ids)

    def __getitem__(self, idx):
        """获取指定索引的ID"""
        if idx >= len(self.ids):
            raise IndexError(f"索引 {idx} 超出范围")
        id = self.ids[idx]
        # data = self.data[id]
        return {"id": id,  "metadata": self.metadata[id]}


class set_dataset(ID_dataset):
    """Alias used by :mod:`data_factory` for dynamic ID datasets."""

    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)

