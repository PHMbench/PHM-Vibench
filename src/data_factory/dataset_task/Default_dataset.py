import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from pytorch_lightning.utilities import CombinedLoader
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
        # self.metadata = metadata
        self.args_data = args_data
        self.mode = mode
        
        # 数据处理参数
        self.window_size = args_data.window_size
        self.stride = args_data.stride
        self.train_ratio = args_data.train_ratio
        self.num_window = args_data.num_window
        self.window_sampling_strategy = getattr(args_data, 'window_sampling_strategy', 'evenly_spaced') # 新增：获取窗口采样策略，默认为evenly_spaced

        # 数据预处理
        self.processed_data = []  # 存储处理后的样本
                
        # 处理数据
        self.prepare_data(metadata)
        
    def prepare_data(self,metadata=None):
        """
        准备数据：将原始数据按窗口大小和步长分割成样本
        如果mode是train或valid，则划分数据集
        """
        self._process_single_data(self.data)

        # 如果是train或valid模式，进行数据集划分
        if self.mode in ["train", "valid"]:
            self._split_data_for_mode()
            
        self.total_samples = len(self.processed_data) # L'
        self.label = metadata[self.key]["Label"]
    
    def _sequential_sampling(self, sample_data, data_length):
        """顺序采样"""
        num_samples = max(0, (data_length - self.window_size) // self.stride + 1)
        num_samples = min(num_samples, self.num_window) 
        
        for i in range(num_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            self.processed_data.append(sample_data[start_idx:end_idx])

    def _random_sampling(self, sample_data, data_length):
        """随机采样"""
        if data_length == self.window_size: # 如果数据长度刚好等于窗口大小
             self.processed_data.append(sample_data)
        else:
            possible_starts = np.arange(data_length - self.window_size + 1)
            if len(possible_starts) < self.num_window:
                # 如果可能的起始点少于请求的窗口数，则取所有可能的窗口
                selected_starts = possible_starts
            else:
                selected_starts = np.random.choice(possible_starts, size=self.num_window, replace=False)
            
            for start_idx in selected_starts:
                end_idx = start_idx + self.window_size
                self.processed_data.append(sample_data[start_idx:end_idx])

    def _evenly_spaced_sampling(self, sample_data, data_length):
        """等间隔采样"""
        if self.num_window == 0:
            return
        if data_length == self.window_size: # 如果数据长度刚好等于窗口大小
            self.processed_data.append(sample_data)
        elif self.num_window == 1: # 如果只需要一个窗口，则从中间取
            start_idx = (data_length - self.window_size) // 2
            end_idx = start_idx + self.window_size
            self.processed_data.append(sample_data[start_idx:end_idx])
        else:
            effective_length = data_length - self.window_size 
            if effective_length < 0:
                return

            if self.num_window > effective_length + 1: 
                step = max(0, effective_length / (self.num_window -1) if self.num_window > 1 else 0)
            else:
                step = effective_length / (self.num_window - 1) if self.num_window > 1 else 0
            
            for i in range(self.num_window):
                start_idx = int(round(i * step))
                start_idx = min(start_idx, data_length - self.window_size)
                end_idx = start_idx + self.window_size
                self.processed_data.append(sample_data[start_idx:end_idx])

    def _process_single_data(self, sample_data):
        """
        处理单个数据样本，应用滑动窗口
        """
        # 根据配置转换数据类型
        if self.args_data.dtype:
            if self.args_data.dtype == 'float32':
                sample_data = sample_data.astype(np.float32)
            elif self.args_data.dtype == 'float64':
                sample_data = sample_data.astype(np.float64)
        if sample_data.ndim == 3:
            # 如果数据是三维的，转换为二维
            sample_data = sample_data.reshape(sample_data.shape[0], -1)
        
        data_length = len(sample_data)
        
        if data_length < self.window_size:
            # 如果数据长度小于窗口大小，则不处理或进行填充等操作（此处简单跳过）
            # 可以根据需求添加更复杂的处理逻辑，例如报错、填充等
            print(f"Warning: Data length ({data_length}) is less than window size ({self.window_size}). Skipping this data.")
            return

        # print(self.key,sample_data.shape)
        # windows 
        if self.window_sampling_strategy == 'sequential':
            self._sequential_sampling(sample_data, data_length)
        elif self.window_sampling_strategy == 'random':
            self._random_sampling(sample_data, data_length)
        elif self.window_sampling_strategy == 'evenly_spaced':
            self._evenly_spaced_sampling(sample_data, data_length)
        else:
            raise ValueError(f"Unknown window_sampling_strategy: {self.window_sampling_strategy}")

        # 对已切好的窗口逐个做归一化和加噪声
        normalized_windows = []
        for window in self.processed_data:
            w = self._normalize_window(window)
            w = self._maybe_add_noise(w)
            normalized_windows.append(w)
        self.processed_data = normalized_windows

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        """
        对单个窗口执行归一化。支持与 args_data.normalization 相同的选项，
        但按窗口级别进行统计（min/max 或 mean/std）。
        """
        norm = getattr(self.args_data, 'normalization', 'standardization')
        if norm == 'minmax':
            min_vals = np.min(window, axis=0)
            max_vals = np.max(window, axis=0)
            denominator = max_vals - min_vals
            denominator[denominator == 0] = 1
            return (window - min_vals) / denominator
        elif norm == 'standardization':
            mean_vals = np.mean(window, axis=0)
            std_vals = np.std(window, axis=0)
            return (window - mean_vals) / (std_vals + 1e-8)
        elif norm == 'none':
            return window
        else:
            raise ValueError(f"Unknown normalization method: {norm}")

    def _maybe_add_noise(self, window: np.ndarray) -> np.ndarray:
        """
        根据 args_data.noise_snr 对单个窗口注入 AWGN。
        若未设置 noise_snr 或计算失败，则返回原窗口。
        """
        noise_snr = getattr(self.args_data, 'noise_snr', None)
        if noise_snr is None:
            return window
        try:
            snr_db = float(noise_snr)
            signal_power = np.mean(window.astype(np.float64) ** 2)
            if signal_power <= 0:
                return window
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power)
            noise = np.random.normal(
                loc=0.0,
                scale=noise_std,
                size=window.shape
            ).astype(window.dtype)
            return window + noise
        except Exception:
            return window


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
        # print(f"获取样本 {idx}，数据长度: {sample.shape}")

        out = {
            "x": sample,
            "y": self.label # 所有的label
        }
        return out


class classification_dataset(Default_dataset):
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
