# PHM-Vibench Data Factory Skill

## 功能概述

PHM-Vibench数据处理核心Skill，专门管理30+工业数据集的加载、预处理、特征提取和优化。

**核心优势**:
- **多数据集支持**: 统一接口处理30+工业数据集（CWRU、XJTU、FEMTO、MFPT等）
- **智能数据发现**: 自动识别和配置本地数据集
- **跨域数据管理**: 专门支持跨域泛化实验的数据配置
- **性能优化**: H5文件处理和内存优化策略
- **标准化输出**: 统一的数据格式和元数据管理

## 主要功能

### 1. 多系统数据集配置
支持30+工业数据集的统一管理：
- **CWRU** (System ID: 1): Case Western Reserve University轴承数据
- **XJTU** (System ID: 2): 西安交通大学轴承数据
- **FEMTO** (System ID: 3): FEMTO-ST预测性维护数据
- **MFPT** (System ID: 4): 机械故障预防技术学会数据
- **Ottawa** (System ID: 5): 渥太华大学轴承数据
- **THU** (System ID: 6): 清华大学轴承数据
- **HUST** (System ID: 7): 华中科技大学数据
- **SEU** (System ID: 8): 东南大学数据
- **NUAA** (System ID: 9): 南京航空航天大学数据
- **SJTU** (System ID: 10): 上海交通大学数据
- **MIT** (System ID: 11): 麻省理工学院数据
- **JNU** (System ID: 12): 暨南大学数据

### 2. 跨域数据配置
自动配置跨域泛化实验的数据流：
- **源域选择**: 选择一个或多个源数据集
- **目标域选择**: 选择目标数据集进行泛化测试
- **数据对齐**: 自动处理不同数据集的格式对齐
- **标签映射**: 统一不同数据集的故障标签体系

### 3. ModelScope集成优化
与ModelScope数据平台的深度集成：
- **本地缓存**: ModelScope数据集的本地缓存管理
- **增量更新**: 支持增量下载和更新机制
- **离线模式**: 完全支持离线数据处理
- **版本管理**: 数据集版本控制和回滚

### 4. 数据预处理流水线
标准化的数据预处理流程：
- **信号标准化**: Z-score标准化、Min-Max标准化
- **分段处理**: 可配置的信号分段和重叠策略
- **特征工程**: 统计特征、频域特征、时频特征
- **数据增强**: 噪声添加、时域变换等增强技术

### 5. 小样本学习数据生成
专门支持小样本学习的数据组织：
- **N-way K-shot配置**: 灵活的N-way K-shot数据采样
- **Episode生成**: 自动生成训练和测试episodes
- **数据平衡**: 确保类别平衡和分布合理
- **元数据管理**: 完整的episode元数据记录

## 使用方法

### 基本用法

#### 配置多系统数据集
```bash
skill: "data-factory"

# 中文交互示例
"配置CWRU和XJTU数据集进行跨域实验"
"使用系统ID [1,2,6,5,12] 的数据集"
"设置跨域从CWRU到THU的数据映射"
```

#### 数据预处理配置
```python
# 预处理配置
data_config = skill.configure_preprocessing(
    normalize=True,
    segment_length=2048,
    overlap=0.75,
    window_function="hann"
)
```

#### 小样本学习数据生成
```python
# 小样本数据配置
few_shot_config = skill.create_few_shot_data(
    n_way=5,
    k_shot=5,
    n_query=15,
    n_episodes=1000
)
```

### 高级配置

#### 跨域实验数据配置
```python
# 跨域数据配置
cross_domain_config = skill.configure_cross_domain(
    source_domains=[1, 2],      # CWRU, XJTU
    target_domains=[6, 5, 12],  # THU, Ottawa, JNU
    data_alignment=True,
    label_mapping="standard"
)
```

#### ModelScope数据集成
```python
# ModelScope数据配置
modelscope_config = skill.configure_modelscope(
    dataset_id="PHMbench/PHM-Vibench",
    cache_dir="./data/modelscope",
    use_local_cache=True,
    auto_update=False
)
```

#### 自定义数据流水线
```python
# 自定义数据处理流水线
pipeline = skill.create_processing_pipeline([
    ("load", DataLoader),
    ("normalize", StandardScaler),
    ("segment", SignalSegmenter),
    ("feature", FeatureExtractor),
    ("augment", DataAugmenter)
])
```

## 嵌套子Skills

### dataset-loader-skill
专门负责数据集加载和基础处理：
- 数据集发现和验证
- 格式转换和标准化
- 元数据提取和管理

### modelscope-integration-skill
ModelScope平台集成专用：
- 数据集下载和缓存
- 版本管理和更新
- 离线模式支持

### data-preprocessing-skill
数据预处理和特征工程：
- 信号处理算法
- 特征提取和选择
- 数据增强技术

### cross-domain-manager-skill
跨域数据管理专用：
- 域适配和对齐
- 标签映射和统一
- 分布差异分析

### few-shot-generator-skill
小样本学习数据生成：
- Episode采样策略
- 类别平衡算法
- 元数据管理

## 资源需求

- **内存**: 最低4GB，推荐8GB+（处理大型数据集）
- **磁盘**: 根据数据集大小，建议100GB+存储空间
- **CPU**: 多核处理器推荐，用于并行数据处理
- **网络**: ModelScope数据下载需要稳定网络连接

## 输出格式

### 数据集配置输出
```yaml
data:
  # 数据集系统配置
  target_system_id: [1, 2, 6, 5, 12]

  # 跨域配置
  cross_domain:
    enabled: true
    source_domains: [1, 2]
    target_domains: [6, 5, 12]

  # 预处理配置
  preprocessing:
    normalize: true
    standardize: true
    segment_length: 2048
    overlap: 0.75

  # 数据增强配置
  augmentation:
    noise_level: 0.05
    time_shift: 0.1
    amplitude_scale: 0.2

  # 小样本配置
  few_shot:
    enabled: true
    n_way: 5
    k_shot: [1, 3, 5, 10]
    n_episodes: 1000
```

### 元数据输出
```python
{
    "dataset_info": {
        "name": "CWRU",
        "system_id": 1,
        "samples": 12000,
        "classes": 10,
        "sampling_rate": 12000,
        "fault_types": ["Normal", "IR007", "IR014", "IR021", "OR007", "OR014", "OR021", "B007", "B014", "B021"]
    },
    "preprocessing_applied": ["normalization", "segmentation", "windowing"],
    "feature_statistics": {
        "mean": 0.0,
        "std": 1.0,
        "min": -3.2,
        "max": 3.1
    },
    "cross_domain_info": {
        "domain_shift": 0.23,
        "covariance_difference": 0.15
    }
}
```

## 依赖关系

### 必需的父Skills
- config-manager-skill: 配置管理和验证

### 嵌套的子Skills
- dataset-loader-skill: 数据集加载
- modelscope-integration-skill: ModelScope集成
- data-preprocessing-skill: 数据预处理
- cross-domain-manager-skill: 跨域管理
- few-shot-generator-skill: 小样本生成

### 外部依赖
- `src.data_factory`: PHM-Vibench数据处理工厂
- `modelscope`: ModelScope数据平台SDK
- `h5py`: H5文件处理
- `numpy`, `scipy`: 数值计算和信号处理
- `pandas`: 数据处理和分析

## 最佳实践

### 1. 数据集组织结构
```
data/
├── raw/                      # 原始数据
│   ├── CWRU/
│   ├── XJTU/
│   └── ...
├── processed/                # 处理后数据
│   ├── h5_files/
│   └── features/
├── modelscope/               # ModelScope缓存
│   ├── cache/
│   └── metadata/
└── metadata/                 # 元数据文件
    ├── metadata_CWRU.xlsx
    └── dataset_registry.json
```

### 2. 数据预处理流水线
- 使用标准化流程确保数据一致性
- 保存预处理参数用于推理时重现
- 定期检查数据质量和分布
- 使用交叉验证确保预处理效果

### 3. 跨域实验建议
- 选择具有明显差异的源域和目标域
- 进行域适应分析指导模型选择
- 使用多种评估指标衡量跨域性能
- 记录详细的域间差异统计

### 4. 小样本学习优化
- 合理设置N-way K-shot参数
- 确保episodes的类别平衡
- 使用分层采样避免偏差
- 验证数据分布的合理性

## 错误处理

### 常见问题及解决方案

1. **数据集找不到**
   ```
   错误: Dataset with system_id 1 not found
   解决: 检查data/raw/目录下是否有对应数据集
   ```

2. **内存不足**
   ```
   错误: MemoryError during data loading
   解决: 使用增量加载或减少batch_size
   ```

3. **跨域标签不匹配**
   ```
   错误: Label mismatch between domains
   解决: 使用自动标签映射功能
   ```

## 性能优化

### 数据加载优化
- 使用H5格式加速数据访问
- 实现多进程并行数据加载
- 缓存预处理结果避免重复计算
- 使用内存映射处理大型数据集

### 内存管理
- 实现数据流式处理
- 及时释放不需要的数据
- 使用生成器减少内存占用
- 监控内存使用情况

## 更新日志

### v1.0.0 (2025-01-13)
- 初始版本发布
- 支持30+工业数据集统一管理
- 实现跨域数据配置功能
- 集成ModelScope数据平台
- 添加小样本学习数据生成
- 支持中文自然语言交互

---

**注意**: 此Skill完全兼容PHM-Vibench的data_factory模块，提供统一的数据处理接口和优化的性能。