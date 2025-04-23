# PHMBench/Vbench 数据贡献指南 （AI生成）

本文档详细规定了数据集的标准处理流程，确保所有贡献者遵循统一规范。
内测阶段只需完成[3. 数据标准化](#3-数据标准化)部分，后续由@liq22负责数据集注册和数据加载器实现。

## 1. 任务分配与沟通

- **任务分发**：所有数据集任务通过群组聊天进行分配和协调，参考在线共享文档。
- **进度报告**：在群内定期提供进度更新。
- **问题解决**：遇到技术问题请立即在群内提出，确保及时解决。
- **决策过程**：重大决策需在群内讨论并达成共识后执行。

## 2. 数据下载

- **目录结构**：将原始数据集组织到以下目录：
    ```
    ./PHMBench/Vbench/data/raw/[dataset_name]/
    ```
- **文件组织**：保持原始数据集的文件结构不变
- **来源记录**：记录数据集来源、版本及下载日期，便于追溯

## 3. 数据标准化

### 3.1 格式提取

- 从原始数据提取为标准长度(L)和通道数(C)格式的numpy数组
- 确保数据类型一致性，推荐使用float32类型

### 3.2 命名规范

- **必须严格遵循**以下命名约定：
    ```
    dataset_modality_condition1-condition2-condition3-..._other.npy
    ```
- **条件顺序**规定：
    1. 速度(s)：如 s1970rpm
    2. 负载(l)：如 l1kn
    3. 故障程度：如 3mm
- **文件名示例**：`CWRU_vib_s1970rpm_l1kn_3mm.npy`
- 可根据需要创建分级目录，但最终文件命名必须符合上述规范

### 3.3 元数据生成@liq22 给出模板

创建`metadata.csv`文件，包含以下列：

| 字段名 | 说明 | 示例值 |
|--------|------|------|
| original_shape_L | 原始数据长度 | 4096 |
| original_shape_C | 原始数据通道数 | 1 |
| processed_indices | 后处理数据标号 | 0,1,2,3 |
| is_labeled | 是否有标签 | True/False |
| file_name | 对应的npy文件名 | 'CWRU_vib_s1970rpm_l1kn_3mm.npy' |
| is_normal | 是否为正常样本 | True |
| is_valid | 数据是否有效 | True |
| has_metadata | 是否包含额外元数据 | True |
| source_file | 原始源文件名 | 'original_file.mat' |
| dataset_name | 数据集名称 | 'CWRU', 'THU' |
| class_id | 故障类别ID | 1,2,3 |

- 每行对应一个标准化的npy文件
- CSV文件应与npy文件保存在同一目录下
- 所有标准化的npy文件及metadata.csv将上传至Hugging Face

## 4. 数据结构

创建数据读取类，该类应：

- 加载并解析标准化的`.npy`文件和`metadata.csv`
- 组织成以下字典结构：
    ```python
    {
            'data': numpy_array,  # 形状为(L, C)的数组
            'condition_speed': speed_value,  # 从文件名解析
            'condition_load': load_value,    # 从文件名解析
            'fault_size': fault_size,        # 从文件名解析
            'dataset_name': dataset_name,    # 从metadata获取
            'class_id': class_id,            # 从metadata获取
            # 其他相关元数据
    }
    ```
- 根据pipeline需求，将处理好的数据保存为`pipeline_name_cache.hdf5`

## 5. 预处理

实现预处理函数，这些函数应：

- 对numpy数组应用归一化（Min-Max或Z-score标准化）
- 处理缺失值和异常值
- 应用领域特定信号处理变换（FFT、小波变换等）
- 记录所有预处理步骤，确保可重现性
- 建立缓存机制：`cache_path = f"cache/{dataset_name}_{pipeline_name}.json"`

## 6. 任务类实现

通过继承PyTorch的Dataset创建数据集类：

```python
class YourDataset(torch.utils.data.Dataset):
        def __init__(self, standardized_data, **config):
                # 使用标准化数据初始化
                # 根据参数配置数据分割方式
                # 将数据转换为B, L, C格式用于分类任务
                
        def transform_to_BLC(self):
                # 实现L,C到B,L,C的转换
                
        def __getitem__(self, index):
                # 返回单个数据样本及其标签
                
        def __len__(self):
                # 返回数据集大小
```

## 7. 数据集注册

在DataFactory中注册您的数据集：

```python
# 在datafactory.py中
from .your_dataset import YourDataset

def register_datasets():
        DATA_REGISTRY["your_dataset_name"] = YourDataset
```

## 8. DataLoader封装

为您的数据集创建DataLoader实例：

```python
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
)
```

## 9. 多DataLoader实现

如有需要，为模型训练实现MultipleDataLoader包装器：

```python
class MultipleDataLoader:
        def __init__(self, dataloaders):
                self.dataloaders = dataloaders
                # 额外配置
                
        def __iter__(self):
                # 自定义迭代逻辑，如轮流从不同数据加载器取批次
```

## 质量检查清单

- [ ] 数据文件命名符合规范
- [ ] metadata.csv包含所有必需字段且格式正确
- [ ] 预处理步骤有详细文档记录
- [ ] 数据集类正确实现所有必需方法
- [ ] 数据集在DataFactory中正确注册
- [ ] 测试确认数据能够正确加载和预处理