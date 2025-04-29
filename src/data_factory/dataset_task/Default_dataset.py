import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class Default_dataset(Dataset): # THU_006or018_basic
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        """
        简化的数据集类
        Args:
            data: 输入数据，可能是单一ID的数据 [L, C]，或字典格式 {ID: 数据}
            metadata: 数据元信息，格式为 {ID: {字段: 值}} 字典
            args_data: 数据处理参数
            args_task: 任务参数
            mode: 数据模式，可选 "train", "valid", "test"
        """
        self.key = list(data.keys())[0]
        self.data = data[self.key]  # 取出第一个键的数据
        self.metadata = metadata
        self.args_data = args_data
        self.mode = mode
        
        # 数据处理参数
        self.window_size = args_data.window_size
        self.stride = args_data.stride
        self.train_ratio = args_data.train_ratio
        
        # 数据预处理
        self.processed_data = []  # 存储处理后的样本
                
        # 处理数据
        self.prepare_data()
        
    def prepare_data(self):
        """
        准备数据：将原始数据按窗口大小和步长分割成样本
        如果mode是train或valid，则划分数据集
        """
        self._process_single_data(self.data)

        # 如果是train或valid模式，进行数据集划分
        if self.mode in ["train", "valid"]:
            self._split_data_for_mode()
            
        self.total_samples = len(self.processed_data)
        self.label = self.metadata[self.key]["Label"]
    
    def _process_single_data(self, sample_data):
        """
        处理单个数据样本，应用滑动窗口
        """
        data_length = len(sample_data)
        num_samples = max(0, (data_length - self.window_size) // self.stride + 1)
        
        for i in range(num_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            self.processed_data.append(sample_data[start_idx:end_idx])

    def _split_data_for_mode(self):
        """
        根据当前模式划分数据集
        """
        if not self.processed_data:
            return
            
        # 计算划分点
        total_samples = len(self.processed_data)
        train_size = int(self.train_ratio * total_samples)
        
        if self.mode == "train":
            # 训练模式只保留训练数据
            self.processed_data = self.processed_data[:train_size]
        elif self.mode == "valid":
            # 验证模式只保留验证数据
            self.processed_data = self.processed_data[train_size:]
        self.total_samples = len(self.processed_data)
    
    def __len__(self):
        """返回数据集长度"""
        return self.total_samples
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围")
        
        sample = self.processed_data[idx]

        
        return sample, self.label


class Classification_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)


class RUL_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)

class Anomaly_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)

class DigitalTwin_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)


class FM_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)