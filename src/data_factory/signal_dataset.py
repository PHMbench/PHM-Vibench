import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from src.data_factory import register_dataset
from src.data_factory.base_dataset import BaseDataset

@register_dataset('DummyDataset')  # 修改注册名称
class DummyDataset(BaseDataset):  # 修改类名
    """
    示例信号数据集类，用于测试框架
    
    该类生成简单的合成信号数据，用于测试 Vbench 框架的功能
    """
    
    def __init__(self, 
                 data_path=None, 
                 train_ratio=0.7, 
                 val_ratio=0.15, 
                 test_ratio=0.15,
                 batch_size=32, 
                 shuffle=True, 
                 random_seed=42,
                 **kwargs):
        """
        初始化信号数据集
        
        Args:
            data_path: 数据文件路径，如果为 None 则生成合成数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            batch_size: 批次大小
            shuffle: 是否打乱数据
            random_seed: 随机种子
            **kwargs: 其他参数
        """
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        # 准备数据
        self._prepare_data()
        
    def _prepare_data(self):
        """准备数据，如果指定了文件则加载，否则生成合成数据"""
        if self.data_path and os.path.exists(self.data_path):
            # 加载数据文件
            print(f"[INFO] 加载数据文件: {self.data_path}")
            try:
                data = np.load(self.data_path)
                x = data['x']
                y = data['y']
            except Exception as e:
                print(f"[ERROR] 加载数据文件失败: {e}")
                print("[INFO] 使用合成数据代替")
                x, y = self._generate_synthetic_data()
        else:
            # 生成合成数据
            print("[INFO] 使用合成数据")
            x, y = self._generate_synthetic_data()
        
        # 划分数据集
        total_samples = len(x)
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)
        
        # 使用全局随机种子，避免重复设置
        indices = np.random.permutation(total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # 创建子数据集
        self.train_dataset = _DummyDatasetSubset(x[train_indices], y[train_indices])  # 修改类名
        self.val_dataset = _DummyDatasetSubset(x[val_indices], y[val_indices])  # 修改类名
        self.test_dataset = _DummyDatasetSubset(x[test_indices], y[test_indices])  # 修改类名
        
        print(f"[INFO] 数据准备完成 - 训练: {len(self.train_dataset)}, "
              f"验证: {len(self.val_dataset)}, 测试: {len(self.test_dataset)}")
    
    def _generate_synthetic_data(self, n_samples=1000, input_dim=10, output_dim=1):
        """生成简单的合成数据"""
        # 创建随机输入特征
        x = np.random.randn(n_samples, input_dim).astype(np.float32)
        
        # 创建目标变量 (简单的线性关系加噪声)
        w = np.random.randn(input_dim, output_dim).astype(np.float32)
        y = np.dot(x, w) + 0.1 * np.random.randn(n_samples, output_dim).astype(np.float32)
        
        return x, y
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            pin_memory=True
        )
    
    def get_val_loader(self):
        """获取验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def __str__(self):
        """返回数据集描述"""
        return (f"DummyDataset(train_size={len(self.train_dataset)}, "  # 修改类名
                f"val_size={len(self.val_dataset)}, "
                f"test_size={len(self.test_dataset)}, "
                f"batch_size={self.batch_size})")


class _DummyDatasetSubset(Dataset):  # 修改类名
    """内部使用的数据集子集类，用于创建 DataLoader"""
    
    def __init__(self, x, y):
        """初始化数据集子集"""
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.x)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    """测试入口点"""
    # 创建数据集实例
    dataset = DummyDataset()  # 修改类名
    
    # 获取数据加载器
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()
    
    # 检查数据集
    print(f"\n数据集信息: {dataset}")
    
    # 检查数据批次
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n批次形状: x={x_batch.shape}, y={y_batch.shape}")
    
    print("\n数据集测试成功!")