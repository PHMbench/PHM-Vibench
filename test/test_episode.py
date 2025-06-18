import torch
import numpy as np
from types import SimpleNamespace
from torch.utils.data import DataLoader
import os
import sys

# 添加项目根目录到路径
sys.path.append('/home/lq/LQcode/2_project/PHMBench/PHM-Vibench')

def test_few_shot_implementations():
    """测试两种 Few-Shot 实现方式是否功能一致"""
    print("=== 测试 Few-Shot 实现方式一致性 ===\n")
    
    # 模拟配置参数
    args_data = SimpleNamespace(
        data_dir='./test_data',
        metadata_file='test_metadata.csv',
        batch_size=1,
        num_workers=0
    )
    
    args_task = SimpleNamespace(
        type='FS',
        name='test_task',
        target_system_id=[1, 2],
        n_way=3,
        k_shot=2,
        q_query=1,
        episodes_per_epoch=5
    )
    
    # 创建模拟数据
    class MockMetadata:
        def __init__(self):
            self._data = {}
            # 5个类别，每个类别4个样本
            for class_id in range(5):
                for sample_id in range(4):
                    id_key = f"class_{class_id}_sample_{sample_id}"
                    self._data[id_key] = {
                        'Label': class_id,
                        'Dataset_id': 1,
                        'Name': 'TestDataset'
                    }
        
        def keys(self):
            return self._data.keys()
        
        def __getitem__(self, key):
            return SimpleNamespace(**self._data[key])
    
    class MockData:
        def __init__(self):
            np.random.seed(42)  # 固定随机种子确保一致性
            self._data = {}
            for class_id in range(5):
                for sample_id in range(4):
                    id_key = f"class_{class_id}_sample_{sample_id}"
                    # 每个类别有不同的数据模式
                    self._data[id_key] = np.random.randn(100, 3) + class_id
        
        def __getitem__(self, key):
            return self._data[key]
        
        def keys(self):
            return self._data.keys()
        
        def items(self):
            return self._data.items()
    
    # 创建模拟的 dataset
    class MockSingleDataset:
        def __init__(self, data_dict, metadata):
            self.samples = []
            for id_key in data_dict.keys():
                meta = metadata[id_key]
                self.samples.append({
                    'data': torch.FloatTensor(data_dict[id_key]),
                    'label': meta.Label,
                    'id': id_key
                })
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    class MockIdIncludedDataset:
        def __init__(self, dataset_dict, metadata):
            self.datasets = dataset_dict
            self.metadata = metadata
            # 合并所有数据集
            self.all_samples = []
            for dataset in dataset_dict.values():
                self.all_samples.extend(dataset.samples)
        
        def __len__(self):
            return len(self.all_samples)
        
        def __getitem__(self, idx):
            return self.all_samples[idx]
        
        def get_samples_by_label(self, label):
            """按标签获取样本"""
            return [sample for sample in self.all_samples if sample['label'] == label]
    
    # 创建模拟数据
    metadata = MockMetadata()
    data = MockData()
    
    # 创建数据集
    train_dataset_dict = {}
    for id_key in data.keys():
        train_dataset_dict[id_key] = MockSingleDataset({id_key: data[id_key]}, metadata)
    
    train_dataset = MockIdIncludedDataset(train_dataset_dict, metadata)
    
    print(f"总样本数: {len(train_dataset)}")
    print(f"类别分布: {[len(train_dataset.get_samples_by_label(i)) for i in range(5)]}")
    
    # ===== 方法1: 使用 episodic_sampler + Episode_dataset =====
    try:
        from src.data_factory.samplers.del.episodic_sampler import Sampler
        from src.data_factory.dataset_task.FS.Episode_dataset import set_dataset
        
        print("\n--- 方法1: episodic_sampler + Episode_dataset ---")
        
        # 创建 episodic sampler
        sampler = Sampler(
            dataset=train_dataset,
            n_way=args_task.n_way,
            k_shot=args_task.k_shot,
            q_query=args_task.q_query,
            episodes_per_epoch=args_task.episodes_per_epoch
        )
        
        # 创建 episode dataset
        episode_dataset = set_dataset(train_dataset, sampler)
        dataloader1 = DataLoader(episode_dataset, batch_size=1, shuffle=False)
        
        results1 = []
        for i, batch in enumerate(dataloader1):
            if i >= 3:  # 只测试前3个episode
                break
            
            support_labels = batch['support_labels'].numpy().flatten()
            query_labels = batch['query_labels'].numpy().flatten()
            
            results1.append({
                'episode': i,
                'support_labels': support_labels,
                'query_labels': query_labels,
                'support_shape': batch['support_data'].shape,
                'query_shape': batch['query_data'].shape
            })
            
            print(f"Episode {i+1}:")
            print(f"  Support labels: {support_labels}")
            print(f"  Query labels: {query_labels}")
            print(f"  Support shape: {batch['support_data'].shape}")
            print(f"  Query shape: {batch['query_data'].shape}")
        
        method1_success = True
        
    except ImportError as e:
        print(f"方法1导入失败: {e}")
        method1_success = False
        results1 = []
    
    # ===== 方法2: 单独的 episode_dataset =====
    try:
        from src.data_factory.dataset_task.FS.episode_dataset import set_dataset
        
        print("\n--- 方法2: 单独的 episode_dataset ---")
        
        # 创建 few-shot episode dataset
        fs_episode_dataset = set_dataset(
            dataset=train_dataset,
            n_way=args_task.n_way,
            k_shot=args_task.k_shot,
            q_query=args_task.q_query,
            episodes_per_epoch=args_task.episodes_per_epoch
        )
        
        dataloader2 = DataLoader(fs_episode_dataset, batch_size=1, shuffle=False)
        
        results2 = []
        for i, batch in enumerate(dataloader2):
            if i >= 3:  # 只测试前3个episode
                break
            
            support_labels = batch['support_labels'].numpy().flatten()
            query_labels = batch['query_labels'].numpy().flatten()
            
            results2.append({
                'episode': i,
                'support_labels': support_labels,
                'query_labels': query_labels,
                'support_shape': batch['support_data'].shape,
                'query_shape': batch['query_data'].shape
            })
            
            print(f"Episode {i+1}:")
            print(f"  Support labels: {support_labels}")
            print(f"  Query labels: {query_labels}")
            print(f"  Support shape: {batch['support_data'].shape}")
            print(f"  Query shape: {batch['query_data'].shape}")
        
        method2_success = True
        
    except ImportError as e:
        print(f"方法2导入失败: {e}")
        method2_success = False
        results2 = []
    
    # ===== 比较结果 =====
    print("\n=== 比较结果 ===")
    
    if method1_success and method2_success:
        print("两种方法都成功运行")
        
        # 比较形状是否一致
        shape_consistent = True
        for r1, r2 in zip(results1, results2):
            if r1['support_shape'] != r2['support_shape'] or r1['query_shape'] != r2['query_shape']:
                shape_consistent = False
                break
        
        print(f"数据形状一致性: {'✓' if shape_consistent else '✗'}")
        
        # 检查是否都包含期望的类别数量
        expected_support_samples = args_task.n_way * args_task.k_shot
        expected_query_samples = args_task.n_way * args_task.q_query
        
        format_consistent = True
        for i in range(min(len(results1), len(results2))):
            r1, r2 = results1[i], results2[i]
            if (len(r1['support_labels']) != expected_support_samples or 
                len(r1['query_labels']) != expected_query_samples or
                len(r2['support_labels']) != expected_support_samples or 
                len(r2['query_labels']) != expected_query_samples):
                format_consistent = False
                break
        
        print(f"Episode格式一致性: {'✓' if format_consistent else '✗'}")
        
        # 检查类别分布
        unique_classes_consistent = True
        for i in range(min(len(results1), len(results2))):
            r1, r2 = results1[i], results2[i]
            unique1 = len(set(r1['support_labels']))
            unique2 = len(set(r2['support_labels']))
            if unique1 != args_task.n_way or unique2 != args_task.n_way:
                unique_classes_consistent = False
                break
        
        print(f"类别数量一致性: {'✓' if unique_classes_consistent else '✗'}")
        
        if shape_consistent and format_consistent and unique_classes_consistent:
            print("\n✅ 两种实现方式功能一致！")
        else:
            print("\n❌ 两种实现方式存在差异")
            
    elif method1_success:
        print("只有方法1成功运行")
    elif method2_success:
        print("只有方法2成功运行")
    else:
        print("两种方法都失败了")

# 运行测试
test_few_shot_implementations()