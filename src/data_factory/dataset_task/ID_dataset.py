import numpy as np
import torch
from torch.utils.data import Dataset


class ID_dataset(Dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        """
        数据集类，管理数据ID。
        在数据集层面不进行具体的数据处理，只返回ID。
        Args:
            data: 输入数据，字典格式 ``{"ID": np.ndarray}``
            metadata: 数据元信息，格式为 ``{"ID": {...}}`` 字典
            args_data: 数据处理参数
            args_task: 任务参数
            mode: 数据模式，可选 "train", "valid", "test"
        Example
        -------
        >>> data = {"1001": np.zeros((10, 3))}
        >>> metadata = {"1001": {"Label": 0}}
        """
        self.id = list(data.keys())[0]
        self.data = data[self.id]
        self.metadata = metadata
        self.args_data = args_data
        self.args_task = args_task
        self.mode = mode

    def __len__(self):
        """返回数据集长度

        Returns
        -------
        int
            恒为 ``1``。
        """
        return 1

    def __getitem__(self, idx):
        """获取指定索引的ID

        Parameters
        ----------
        idx : int
            仅支持 ``0``。

        Returns
        -------
        dict
            ``{"id": str, "data": np.ndarray, "metadata": dict}``
        """
        if idx != 0:
            raise IndexError(f"索引 {idx} 超出范围")
        return {'id': self.id, 'data': self.data, 'metadata': self.metadata[self.id]}


class set_dataset(ID_dataset):
    """Alias to maintain compatibility with ``data_factory``."""

    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        """兼容 ``data_factory`` 的旧接口。"""
        super().__init__(data, metadata, args_data, args_task, mode)


# class classification_dataset(Default_dataset):
#     def __init__(self, data, metadata, args_data, args_task, mode="train"):
#         super().__init__(data, metadata, args_data, args_task, mode)


# class RUL_dataset(Default_dataset):
#     def __init__(self, data, metadata, args_data, args_task, mode="train"):
#         super().__init__(data, metadata, args_data, args_task, mode)

# class Anomaly_dataset(Default_dataset):
#     def __init__(self, data, metadata, args_data, args_task, mode="train"):
#         super().__init__(data, metadata, args_data, args_task, mode)

# class DigitalTwin_dataset(Default_dataset):
#     def __init__(self, data, metadata, args_data, args_task, mode="train"):
#         super().__init__(data, metadata, args_data, args_task, mode)


# class FM_dataset(Default_dataset):
#     def __init__(self, data, metadata, args_data, args_task, mode="train"):
#         super().__init__(data, metadata, args_data, args_task, mode)
