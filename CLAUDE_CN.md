# CLAUDE_CN.md

**说明**：本文档用于描述 PHM-Vibench 的定位、架构与“修改门禁”（double-check gate）。
运行/验证命令请优先参考 `AGENTS.md`（如需中文 runbook，可使用 `AGENTS_CN.md`）。
项目概览与上手路径请优先参考 `README_CN.md`（配置体系细节见 `configs/README.md`）。

## 仓库概述

PHM-Vibench是一个全面的工业设备振动信号分析基准平台，专注于故障诊断和预测性维护。它采用模块化工厂设计模式，广泛支持多种数据集、模型和任务。

本仓库的关键目标是：
- **可复现**：实验由配置文件定义，避免硬编码与环境依赖漂移。
- **可扩展**：新增 data/model/task/trainer 走工厂/注册表入口，保持 wiring 清晰。

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

📖 **从这里开始**: [`configs/README.md`](configs/README.md)（30 秒冒烟 + override 规则 + 配置工具）

配置部分包括：
- `data`: 数据集配置和预处理参数
- `model`: 模型架构和超参数
- `task`: 任务类型、损失函数和训练设置
- `trainer`: 训练编排和硬件设置

### 单一入口契约（避免歧义）
本仓库维护的入口为：
```bash
python main.py --config <yaml> [--override key=value ...]
```
pipeline 由 YAML 顶层 `pipeline:` 选择（不使用 `--pipeline`）。

### 配置“单一事实源”（SSOT）与工具链
- Registry：`configs/config_registry.csv`（字段说明：`docs/config_registry_schema.md`）
- Atlas：`docs/CONFIG_ATLAS.md`（生成：`python -m scripts.gen_config_atlas`）
- Inspect：`python -m scripts.config_inspect`（最终配置 + 字段来源 + 实例化落点 + sanity）
- 校验：`python -m scripts.validate_configs`（loader 驱动 + pydantic；schema 在 `src/config_schema/`）

## Paper / 研究工作流（与主仓库解耦）

论文级实验放在 git submodule 中，避免与主仓库 onboarding/demo 混用：
- `paper/2025-10_foundation_model_0_metric/`（初始化需要网络）：
  - `git submodule update --init --recursive paper/2025-10_foundation_model_0_metric`
  - 说明见 `paper/README_SUBMODULE.md`

原则：主仓库的验证门禁不依赖 paper-only 脚本/配置。

### HSE / HSE-Prompt（论文级）
- 位置：`paper/2025-10_foundation_model_0_metric/`（submodule）
- 目标：HSE/HSE-Prompt 的跨系统泛化实验与论文复现
- 若 submodule 未初始化：以主仓库可运行 demo 为准（`configs/demo/05_pretrain_fewshot/`、`configs/demo/06_pretrain_cddg/`）

## 模块特定文档

有关特定组件的详细指导，请参阅：
- [`configs/README.md`](configs/README.md) - config 目录结构、模板、override 规则与工具链（维护入口）
- [配置系统（源码侧）](./src/configs/CLAUDE.md) - loader/ConfigWrapper 的更深说明（内部实现）
- [数据工厂](./src/data_factory/CLAUDE.md) - 数据集集成、处理和读取器实现
- [模型工厂](./src/model_factory/CLAUDE.md) - 模型架构、ISFM基础模型和实现
- [任务工厂](./src/task_factory/CLAUDE.md) - 任务定义、训练逻辑和损失函数
- [训练器工厂](./src/trainer_factory/CLAUDE.md) - 训练编排和PyTorch Lightning集成
- [工具类](./src/utils/CLAUDE.md) - 实用函数、配置助手和注册模式

## 常用开发命令

### 运行实验
```bash
# 0) 离线冒烟（仓库内置 Dummy_Data；无需下载数据）
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# 1) DG 示例（domain split；具体系统见 task.target_system_id）
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml

# 2) CDDG 示例（多系统请调整 task.target_system_id）
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml

# 3) 预训练 + few-shot 管道示例（pipeline 由 YAML 的 pipeline 字段选择）
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml
```

### 测试

测试文件要放在 test 目录下
```bash
# 维护中的 pytest 套件
python -m pytest test/
```

### 额外自检（可选）
```bash
# YAML 语法检查
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# loader 冒烟（确认可加载）
python - <<'PY'
from src.configs import load_config
cfg = load_config('configs/demo/00_smoke/dummy_dg.yaml')
print('loaded:', type(cfg), 'keys:', list(cfg.__dict__.keys()))
PY
```

### Streamlit GUI
```bash
# 启动交互式实验界面
streamlit run streamlit_app.py
```
状态：实验性功能（TODO），建议以 `configs/demo/` 的命令行 demo 为准。

### 测试配置
- 本仓库根目录没有 `pytest.ini`；`pytest` 使用默认发现规则。
- `dev/test_history/pytest.ini` 仅用于历史 runner（可选）。

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
默认结果保存在 `save/` 下；若设置了 `environment.output_dir`（demo 常用 `results/demo/...`），则以其为准。
最终目录结构为：`base_dir/<experiment_name>/iter_<k>/`（见 `src/configs/config_utils.py:path_name`）。

示意（具体文件随 trainer/task 变化）：
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
- vibecoding（AI 辅助编码）更新默认选择“最简单可行”方案：避免过度工程化与不必要的防御性设计；遵循奥卡姆剃刀，
  立足第一性原理，渐进式开发与验证。

## 模型与任务（高层地图）

### 常见基础模型/组件
- ISFM 家族：`M_01_ISFM`、`M_02_ISFM`、`M_03_ISFM`
- backbone 示例：`B_04_Dlinear`、`B_06_TimesNet`、`B_08_PatchTST`、`B_09_FNO`
- head 示例：`H_01_Linear_cla`（分类）、`H_03_Linear_pred`（预测）

### 任务类型（示例）
- 分类 / DG / CDDG（域泛化）
- FS / GFS（少样本）
- pretrain（自监督/对比预训练）

## 变更门禁（double-check）

### 不做破坏性变更
- 不随意改 `main.py` 公共 CLI 或核心 YAML keyspace（environment/data/model/task/trainer）。
- 如确需调整：必须提供兼容层 + 迁移说明。

### 稳定执行顺序（避免“文档/配置漂移”）
1) Registry（`configs/config_registry.csv`）
2) Atlas（`docs/CONFIG_ATLAS.md`）
3) Inspect（`scripts/config_inspect.py`）
4) Schema validate（`scripts/validate_configs.py`）
5) `configs/**/README.md`（先命令后解释）
6) CI/tests + 最终验收
