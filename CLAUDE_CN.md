# CLAUDE_CN.md

**说明**: 本文档为PHM-Vibench项目的中文使用指南。后续与Claude Code的交互将默认使用中文进行。

本文档为Claude Code (claude.ai/code)在此代码库中工作时提供指导。

## 仓库概述

PHM-Vibench是一个全面的工业设备振动信号分析基准平台，专注于故障诊断和预测性维护。它采用模块化工厂设计模式，广泛支持多种数据集、模型和任务。

## 关键架构组件

### 工厂设计模式
代码库使用工厂模式以实现最大的模块化：
- **data_factory/**: 数据集加载和处理，支持30+工业数据集（CWRU、XJTU、FEMTO等）
- **model_factory/**: 神经网络架构，包括Transformers、CNNs、RNNs和专门的基础模型
- **task_factory/**: 任务定义（分类、预测、少样本学习、域泛化）
- **trainer_factory/**: 使用PyTorch Lightning的训练编排

### 管道系统
框架支持多种实验管道：
- `Pipeline_01_default`: 标准训练管道
- `Pipeline_02_pretrain_fewshot`: 两阶段预训练+少样本学习
- `Pipeline_03_multitask_pretrain_finetune`: 多任务基础模型训练
- `Pipeline_ID`: 基于ID的数据处理管道

### 配置驱动的实验
PHM-Vibench v5.0 配置系统提供了极简而强大的实验管理能力：

**核心优势**:
- **统一接口**: 单一`load_config()`函数处理所有配置需求
- **4×4灵活性**: 支持预设/文件/字典/ConfigWrapper × 4种覆盖方式
- **智能合并**: 递归合并嵌套配置，点号展开自动处理
- **链式操作**: 支持copy().update()链式配置构建
- **100%兼容**: 所有现有Pipeline无需修改即可使用

**快速示例**:
```python
from src.configs import load_config
# 从预设加载并覆盖参数
config = load_config('isfm', {'model.d_model': 512, 'task.lr': 0.001})
```

📖 **详细文档**: [配置系统v5.0完整指南](./src/configs/CLAUDE.md)

配置部分包括：
- `data`: 数据集配置和预处理参数
- `model`: 模型架构和超参数
- `task`: 任务类型、损失函数和训练设置
- `trainer`: 训练编排和硬件设置

## 模块特定文档

有关特定组件的详细指导，请参阅：
- [配置系统](./src/configs/CLAUDE.md) - 统一配置管理、YAML模板和多阶段管道
- [数据工厂](./src/data_factory/CLAUDE.md) - 数据集集成、处理和读取器实现
- [模型工厂](./src/model_factory/CLAUDE.md) - 模型架构、ISFM基础模型和实现
- [任务工厂](./src/task_factory/CLAUDE.md) - 任务定义、训练逻辑和损失函数
- [训练器工厂](./src/trainer_factory/CLAUDE.md) - 训练编排和PyTorch Lightning集成
- [工具类](./src/utils/CLAUDE.md) - 实用函数、配置助手和注册模式

## 常用开发命令

### 运行实验
```bash
# 基础单数据集实验
python main.py --config configs/demo/Single_DG/CWRU.yaml

# 跨数据集域泛化
python main.py --config configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml

# 预训练+少样本学习管道
python main.py --pipeline Pipeline_02_pretrain_fewshot --config_path configs/demo/Pretraining/Pretraining_demo.yaml --fs_config_path configs/demo/GFS/GFS_demo.yaml

# 所有数据集实验
python main.py --config configs/demo/Multiple_DG/all.yaml
```

### 测试

测试文件要放在 test 目录下
```bash
# 运行全面测试套件
python run_tests.py

# 运行特定测试类别
pytest test/ -m "not slow"  # 跳过慢速测试
pytest test/ -m "unit"      # 仅单元测试
pytest test/ -m "gpu" --tb=short  # GPU测试
```

### Streamlit GUI
```bash
# 启动交互式实验界面
streamlit run streamlit_app.py
```

### 测试配置
- 测试在`pytest.ini`中配置，具有全面的覆盖率设置
- `requirements-test.txt`中的测试要求包括pytest、coverage和ML测试工具
- 要求最低80%代码覆盖率

## 数据集集成

### 数据结构
- `data/raw/<DATASET_NAME>/`中的原始数据集
- 描述数据集结构的元数据文件：`metadata_*.xlsx`
- 用于高效加载的H5处理文件
- 每个数据集的读取器位于`src/data_factory/reader/RM_*.py`

### 添加新数据集
1. 将原始数据放入`data/raw/<DATASET_NAME>/`
2. 创建描述结构的元数据文件
3. 实现继承自BaseReader的读取器类
4. 在`data_factory/__init__.py`中注册

## 模型架构

### 基础模型（ISFM - 工业信号基础模型）
- **M_01_ISFM**: 基于Transformer的基础基础模型
- **M_02_ISFM**: 高级多模态基础模型
- **M_03_ISFM**: 专门的时态动力学模型

### 骨干网络
- **B_08_PatchTST**: 基于补丁的时间序列Transformer
- **B_04_Dlinear**: 直接线性预测模型
- **B_06_TimesNet**: 时间序列分析网络
- **B_09_FNO**: 信号处理的傅里叶神经算子

### 任务头
- **H_01_Linear_cla**: 线性分类头
- **H_09_multiple_task**: 多任务学习头
- **H_03_Linear_pred**: 线性预测头

## 任务类型和用例

### 支持的任务
- **分类**: 故障诊断和设备状态分类
- **CDDG**: 跨数据集域泛化以提高鲁棒性
- **FS/GFS**: 少样本和广义少样本学习
- **预训练**: 基础模型的自监督预训练

### 域泛化
- 单域：在同一数据集上训练和测试
- 跨数据集：在一个数据集上训练，在另一个数据集上测试
- 多域：使用多个源域提高鲁棒性

## 环境设置

### 依赖项
- Python 3.8+、PyTorch 2.6.0、PyTorch Lightning
- 科学计算：numpy、pandas、scipy、scikit-learn
- 可视化：matplotlib、seaborn、plotly
- ML实用程序：wandb、tensorboard、transformers、timm

### 关键环境变量
在配置文件中设置`data_dir`，指向包含元数据Excel文件和H5数据集文件的数据目录。

## 结果和输出

### 目录结构
结果保存在`save/`下的分层结构中：
```
save/{metadata_file}/{model_name}/{task_type}_{trainer_name}_{timestamp}/
├── checkpoints/     # 模型权重
├── metrics.json     # 性能指标
├── log.txt         # 训练日志
├── figures/        # 可视化图表
└── config.yaml     # 实验配置备份
```

### 日志记录和监控
- WandB集成用于实验跟踪
- 全面的指标记录
- 自动生成分析图表

## 重要说明

- 代码库广泛使用工厂模式 - 始终在适当的工厂中注册新组件
- 配置文件驱动所有实验 - 避免硬编码参数
- 结果按元数据文件、模型和时间戳自动组织
- 框架支持传统ML方法和现代基础模型
- 多任务和跨数据集能力是工业应用的核心功能