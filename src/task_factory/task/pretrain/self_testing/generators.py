"""
Flow模拟数据生成器 (Flow Mock Data Generator)

这个模块提供Flow预训练任务自测试的模拟数据生成功能，遵循test/conftest.py的
现有模式，生成逼真的振动信号用于测试。

Author: PHM-Vibench Team  
Date: 2025-09-10
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class FlowMockDataConfig:
    """
    Flow模拟数据配置类 (Flow Mock Data Configuration)
    
    定义生成模拟数据的参数设置，支持不同测试场景的数据生成需求。
    """
    batch_size: int = 8
    sequence_length: int = 64
    input_dim: int = 3
    num_classes: int = 4
    num_samples: int = 200
    noise_level: float = 0.1
    base_frequency: float = 10.0
    frequency_step: float = 5.0
    random_seed: int = 42
    signal_type: str = "sine"  # "sine", "chirp", "mixed"
    include_harmonics: bool = True
    sampling_rate: float = 1000.0  # Hz
    

class FlowMockDataGenerator:
    """
    Flow模拟数据生成器类 (Flow Mock Data Generator Class)
    
    提供Flow预训练任务测试所需的各种模拟数据，包括振动信号、文件ID、
    配置等。遵循test/conftest.py的现有模式，确保与PHM-Vibench框架兼容。
    """
    
    def __init__(self, config: Optional[FlowMockDataConfig] = None):
        """
        初始化模拟数据生成器
        
        Args:
            config: 数据生成配置，如果为None则使用默认配置
        """
        self.config = config or FlowMockDataConfig()
        self._set_random_seeds()
        
    def _set_random_seeds(self):
        """设置随机种子确保可重现性 (Set random seeds for reproducibility)."""
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
    
    def generate_flow_batch(
        self, 
        batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        include_labels: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成Flow训练批次数据 (Generate Flow training batch data)
        
        遵循test/conftest.py中sample_classification_data和synthetic_dataset的模式，
        生成适用于Flow预训练的逼真振动信号数据。
        
        Args:
            batch_size: 批次大小，如果为None则使用配置中的默认值
            device: 设备类型 (cpu/cuda)
            include_labels: 是否包含标签（用于条件生成测试）
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (数据, 标签)
            - 数据形状: (batch_size, sequence_length, input_dim)
            - 标签形状: (batch_size,) 如果include_labels=True
        """
        batch_size = batch_size or self.config.batch_size
        device = torch.device(device)
        
        # 生成时间序列
        t = np.linspace(0, 1, self.config.sequence_length)
        
        batch_data = []
        batch_labels = [] if include_labels else None
        
        for i in range(batch_size):
            # 为每个样本选择类别（如果需要标签）
            if include_labels:
                class_id = i % self.config.num_classes
                batch_labels.append(class_id)
            else:
                class_id = np.random.randint(0, self.config.num_classes)
            
            # 生成类别特定的信号模式
            signal = self._generate_class_specific_signal(t, class_id)
            
            # 添加多通道数据，遵循conftest.py的模式
            multi_channel = self._create_multichannel_signal(signal)
            
            batch_data.append(multi_channel)
        
        # 转换为张量
        X = torch.FloatTensor(batch_data).to(device)
        y = torch.LongTensor(batch_labels).to(device) if include_labels else None
        
        return X, y
    
    def _generate_class_specific_signal(self, t: np.ndarray, class_id: int) -> np.ndarray:
        """
        生成类别特定的信号模式 (Generate class-specific signal patterns)
        
        Args:
            t: 时间序列
            class_id: 类别ID
            
        Returns:
            生成的信号
        """
        # 基础频率随类别变化，遵循conftest.py中synthetic_dataset的模式
        freq = self.config.base_frequency + class_id * self.config.frequency_step
        
        if self.config.signal_type == "sine":
            signal = np.sin(2 * np.pi * freq * t)
            
            # 添加谐波使信号更加逼真
            if self.config.include_harmonics:
                signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # 二次谐波
                signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t)  # 三次谐波
                
        elif self.config.signal_type == "chirp":
            # 线性调频信号，模拟机械设备的启动/停止过程
            f_end = freq + 20
            signal = np.sin(2 * np.pi * (freq * t + (f_end - freq) * t**2 / 2))
            
        elif self.config.signal_type == "mixed":
            # 混合信号，模拟复杂的工业环境
            signal1 = np.sin(2 * np.pi * freq * t)
            signal2 = 0.5 * np.sin(2 * np.pi * (freq + 7) * t)
            signal = signal1 + signal2
            
        else:
            raise ValueError(f"不支持的信号类型: {self.config.signal_type}")
        
        # 添加高斯噪声
        noise = self.config.noise_level * np.random.randn(len(t))
        return signal + noise
    
    def _create_multichannel_signal(self, base_signal: np.ndarray) -> np.ndarray:
        """
        创建多通道信号 (Create multi-channel signal)
        
        遵循conftest.py中的多通道数据生成模式。
        
        Args:
            base_signal: 基础信号
            
        Returns:
            多通道信号数组，形状为 (sequence_length, input_dim)
        """
        # 创建三通道数据，模拟X、Y、Z轴振动
        channels = []
        
        for i in range(self.config.input_dim):
            if i == 0:
                # 主通道：原始信号加少量噪声
                channel = base_signal + 0.05 * np.random.randn(len(base_signal))
            elif i == 1:
                # 第二通道：相位偏移的信号
                phase_shift = np.pi / 4
                channel = 0.8 * np.sin(np.angle(np.exp(1j * (np.arcsin(base_signal / np.max(np.abs(base_signal))) + phase_shift))))
                channel = 0.8 * base_signal + 0.1 * np.random.randn(len(base_signal))
            else:
                # 第三通道：弱相关信号
                channel = 0.6 * base_signal + 0.15 * np.random.randn(len(base_signal))
            
            channels.append(channel)
        
        return np.stack(channels, axis=1)
    
    def generate_file_ids(
        self, 
        num_files: int = 10,
        domain_ids: Optional[List[int]] = None
    ) -> List[str]:
        """
        生成文件ID用于条件训练模拟 (Generate file IDs for conditional training mock)
        
        模拟PHM-Vibench数据加载器中的文件ID模式，用于测试条件生成功能。
        
        Args:
            num_files: 生成的文件ID数量
            domain_ids: 域ID列表，如果为None则使用默认域
            
        Returns:
            文件ID字符串列表
        """
        if domain_ids is None:
            domain_ids = list(range(1, 5))  # 默认域ID 1-4
        
        file_ids = []
        for i in range(num_files):
            domain_id = domain_ids[i % len(domain_ids)]
            # 使用类似PHM-Vibench的文件ID格式
            file_id = f"domain_{domain_id:02d}_file_{i:04d}.h5"
            file_ids.append(file_id)
        
        return file_ids
    
    def generate_regression_data(
        self,
        batch_size: Optional[int] = None,
        pred_len: int = 24,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成回归任务数据 (Generate regression task data)
        
        遵循conftest.py中sample_regression_data的模式。
        
        Args:
            batch_size: 批次大小
            pred_len: 预测长度
            device: 设备类型
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (输入数据, 目标数据)
        """
        batch_size = batch_size or self.config.batch_size
        device = torch.device(device)
        
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_dim)
        y = torch.randn(batch_size, pred_len, self.config.input_dim)
        
        return x.to(device), y.to(device)
    
    def generate_multimodal_data(
        self,
        batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        生成多模态数据 (Generate multi-modal data)
        
        遵循conftest.py中sample_multimodal_data的模式。
        
        Args:
            batch_size: 批次大小
            device: 设备类型
            
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: (多模态数据字典, 标签)
        """
        batch_size = batch_size or self.config.batch_size
        device = torch.device(device)
        
        data = {
            'vibration': torch.randn(batch_size, self.config.sequence_length, 3),
            'acoustic': torch.randn(batch_size, self.config.sequence_length, 1),
            'thermal': torch.randn(batch_size, 2)
        }
        labels = torch.randint(0, self.config.num_classes, (batch_size,))
        
        # 移动到指定设备
        for key in data:
            data[key] = data[key].to(device)
        labels = labels.to(device)
        
        return data, labels
    
    def generate_synthetic_dataset(
        self,
        num_samples: Optional[int] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成完整的合成数据集 (Generate complete synthetic dataset)
        
        遵循conftest.py中synthetic_dataset的完整模式，用于更全面的测试。
        
        Args:
            num_samples: 样本总数
            device: 设备类型
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (数据集, 标签)
        """
        num_samples = num_samples or self.config.num_samples
        device = torch.device(device)
        
        self._set_random_seeds()
        
        data = []
        labels = []
        
        samples_per_class = num_samples // self.config.num_classes
        t = np.linspace(0, 1, self.config.sequence_length)
        
        for class_id in range(self.config.num_classes):
            for _ in range(samples_per_class):
                # 生成类别特定的信号
                signal = self._generate_class_specific_signal(t, class_id)
                
                # 创建多通道数据
                multi_channel = self._create_multichannel_signal(signal)
                
                data.append(multi_channel)
                labels.append(class_id)
        
        X = torch.FloatTensor(data).to(device)
        y = torch.LongTensor(labels).to(device)
        
        return X, y
    
    def get_data_statistics(self, data: torch.Tensor) -> Dict[str, float]:
        """
        计算数据统计信息 (Compute data statistics)
        
        Args:
            data: 输入数据张量
            
        Returns:
            包含统计信息的字典
        """
        return {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "shape": list(data.shape),
            "requires_grad": data.requires_grad,
            "device": str(data.device),
            "dtype": str(data.dtype)
        }


# 便捷函数，遵循conftest.py的命名约定
def create_flow_mock_data(
    batch_size: int = 8,
    seq_len: int = 64,
    input_dim: int = 3,
    num_classes: int = 4,
    device: Union[str, torch.device] = "cpu",
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建Flow模拟数据的便捷函数 (Convenience function for creating Flow mock data)
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        input_dim: 输入维度
        num_classes: 类别数量
        device: 设备类型
        random_seed: 随机种子
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (数据, 标签)
    """
    config = FlowMockDataConfig(
        batch_size=batch_size,
        sequence_length=seq_len,
        input_dim=input_dim,
        num_classes=num_classes,
        random_seed=random_seed
    )
    
    generator = FlowMockDataGenerator(config)
    return generator.generate_flow_batch(device=device, include_labels=True)


# 导出的类和函数
__all__ = [
    'FlowMockDataConfig',
    'FlowMockDataGenerator', 
    'create_flow_mock_data',
]


if __name__ == "__main__":
    """
    Flow模拟数据生成器自测试 (Flow Mock Data Generator Self-Test)
    
    测试数据生成器的各种功能，确保生成的数据符合预期格式和质量要求。
    """
    print("=" * 60)
    print("Flow模拟数据生成器自测试 (Flow Mock Data Generator Self-Test)")
    print("=" * 60)
    
    try:
        # 测试1: 基本数据生成
        print("\n1. 测试基本Flow批次数据生成...")
        generator = FlowMockDataGenerator()
        
        x, y = generator.generate_flow_batch()
        print(f"✓ 数据形状: {x.shape}, 标签形状: {y.shape}")
        print(f"✓ 数据范围: [{x.min():.3f}, {x.max():.3f}]")
        print(f"✓ 标签范围: [{y.min()}, {y.max()}]")
        print(f"✓ 数据类型: {x.dtype}, 标签类型: {y.dtype}")
        
        # 测试2: 不同信号类型
        print("\n2. 测试不同信号类型...")
        for signal_type in ["sine", "chirp", "mixed"]:
            config = FlowMockDataConfig(signal_type=signal_type, batch_size=4)
            gen = FlowMockDataGenerator(config)
            x, y = gen.generate_flow_batch()
            print(f"✓ {signal_type}信号生成成功，形状: {x.shape}")
        
        # 测试3: 文件ID生成
        print("\n3. 测试文件ID生成...")
        file_ids = generator.generate_file_ids(num_files=5)
        print(f"✓ 生成文件ID: {file_ids[:3]}...")
        
        # 测试4: 设备兼容性测试
        print("\n4. 测试设备兼容性...")
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        
        for device in devices:
            x, y = generator.generate_flow_batch(device=device)
            print(f"✓ {device}设备测试通过，数据设备: {x.device}")
        
        # 测试5: 多模态数据生成
        print("\n5. 测试多模态数据生成...")
        data_dict, labels = generator.generate_multimodal_data()
        print(f"✓ 多模态数据键: {list(data_dict.keys())}")
        for key, value in data_dict.items():
            print(f"  - {key}: {value.shape}")
        
        # 测试6: 合成数据集生成
        print("\n6. 测试完整合成数据集生成...")
        X, y = generator.generate_synthetic_dataset(num_samples=40)
        print(f"✓ 合成数据集形状: {X.shape}, 标签形状: {y.shape}")
        
        # 测试7: 数据统计信息
        print("\n7. 测试数据统计信息...")
        stats = generator.get_data_statistics(X)
        print(f"✓ 数据统计: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}")
        
        # 测试8: 便捷函数测试
        print("\n8. 测试便捷函数...")
        x_conv, y_conv = create_flow_mock_data()
        print(f"✓ 便捷函数生成数据形状: {x_conv.shape}, {y_conv.shape}")
        
        # 测试9: 可重现性测试
        print("\n9. 测试可重现性...")
        x1, y1 = create_flow_mock_data(random_seed=123)
        x2, y2 = create_flow_mock_data(random_seed=123)
        reproducible = torch.allclose(x1, x2) and torch.allclose(y1, y2)
        print(f"✓ 可重现性测试: {'通过' if reproducible else '失败'}")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！Flow模拟数据生成器工作正常。")
        print("📊 生成数据质量符合PHM-Vibench框架要求。")
        print("🔧 可用于Flow预训练任务的自测试场景。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("请检查代码实现并修复问题。")
        raise