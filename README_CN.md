# PHM-Vibench: 工业设备振动信号基准平台

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench Logo" width="300"/>
  
  <p>
    <a href="README.md">English</a> | 
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>
  
  <p><strong>面向工业振动故障诊断与预测性维护的配置优先基准工作台。</strong></p>
  <p><em>内测阶段 - 仅限邀请访问。</em></p>

  <p>
    <img src="https://img.shields.io/badge/状态-内测中-orange" alt="Status: Alpha"/>
    <img src="https://img.shields.io/badge/版本-0.2.0--alpha-blue" alt="Version"/>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="License"/>
    <img src="https://img.shields.io/badge/维护_demo-7-purple" alt="Maintained demos"/>
    <img src="https://img.shields.io/badge/注册状态-sanity__ok-red" alt="Registry status"/>
  </p>

  <p>
    <a href="#-快速开始">快速开始</a> •
    <a href="#-使用指南">使用文档</a> •
    <a href="#-项目亮点">核心特性</a> •
    <a href="#-开发指南">参与贡献</a>
  </p>
</div>

---


## 📖 目录
- [✨ 项目亮点](#-项目亮点)
- [📝 项目背景与简介](#-项目背景与简介)
- [🔄 支持的模型与数据集](#-支持的模型与数据集)
- [🛠️ 安装指南](#️-安装指南)
- [🚀 快速开始](#-快速开始)
- [📘 使用指南](#-使用指南)
- [📂 项目结构](#-项目结构)
- [🧑‍💻 开发指南](#-开发指南)
- [📃 用了该项目发表的文章](#-用了该项目发表的文章)
- [🔮 项目路线图](#-项目路线图)
- [👥 贡献者与社区](#-贡献者与社区)
- [🏛  许可证](#-许可证)
- [📎 引用方式](#-引用方式)

## ✨ 项目亮点

- **模块化工厂架构**：数据读取、模型、任务、训练器和 pipeline 均通过显式配置选择。
- **维护中的 demo 面**：当前公开 smoke 面包含 7 个注册表跟踪的可运行 demo 配置。
- **可追踪配置系统**：`configs/config_registry.csv`、`docs/CONFIG_ATLAS.md` 与 `scripts.config_inspect` 展示维护配置的字段来源。
- **验证门禁**：配置校验、文档校验、维护测试与离线 smoke 是 release gate。
- **配置优先工作流**：研究者可从 `configs/demo/` 复制模板到 `configs/experiments/`，再通过 YAML 或 CLI override 修改行为。
- **研究扩展入口**：新数据集、模型和任务头通过已文档化的 factory 接口集成。

## 📝 项目背景与简介

**❓为什么需要 PHM-Vibench**

### 🎯 A. 项目定位与价值

工业设备故障诊断和预测性维护技术在工业4.0时代具有重要的战略意义，对提高生产效率、降低维护成本和延长设备使用寿命至关重要。然而，随着机器学习和深度学习技术在该领域的广泛应用，研究成果的评估与比较面临以下挑战：

1. 🔍 **实验环境碎片化**：不同研究使用各自的数据预处理流程、模型实现和评估指标
2. 🔄 **可复现性困难**：缺乏标准化的实验流程和完整的实现细节
3. ⚖️ **公平比较的障碍**：数据划分、预处理和评估标准的不一致性导致结果难以直接比较

PHM-Vibench 是 PHMbench 生态中面向工业振动故障诊断的工作台，重点是让实验更容易配置、检查和重复运行。

### 🛠️ B. 核心功能与特性

1. **统一的接口设计**：数据加载、模型构建、任务装配和训练器设置使用共享 factory 接口。
2. **配置记录**：维护中的 demo 由注册表索引，并可通过仓库工具检查。
3. **比较纪律**：共享配置模式和验证门禁减少数据划分、预处理和指标口径漂移。
4. **扩展入口**：新数据集、模型和任务可沿已文档化的 factory 边界添加。

## 🔄 支持的模型与数据集

当前维护范围以 `SUPPORTED_COMPONENTS.md`、`SUPPORTED_COMBINATIONS.md` 与 `KNOWN_LIMITATIONS.md` 为准。
历史配置或参考配置不自动属于 release-supported surface。

### 📊 支持的数据集 见
- [ModelScope 处理后文件](https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files)
- [PHMbench raw data group](https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data)
- [Hugging Face mirror](https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main)




### 🧠 支持的算法模型

维护中的模型/任务组合见 `SUPPORTED_COMBINATIONS.md`。




## 🛠️ 安装指南

> ⚠️ **注意**：当前项目处于内测阶段，仅限获得邀请的用户安装使用。

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.1+ 

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

# 安装依赖
conda create -n PHM python=3.10
conda activate PHM
pip install -r requirements.txt

# （可选）下载数据集（H5 / raw）

例如在 configs/base/data/base_classification.yaml 中
data:
  data_dir: "/path/to/PHM-Vibench"
  metadata_file: "metadata.xlsx"

```




## 🚀 快速开始

使用以下命令运行维护中的 demo 面：

- 入口：`python main.py --config <yaml> [--override key=value ...]`
- 模板来源：`configs/demo/`（本地变体放到 `configs/experiments/`）
- 配置文档与工具：`configs/README.md`
- 变更/运行门禁：`AGENTS.md`（runbook）与 `CLAUDE.md`（change strategy gate）

工程规则：变更应小、明确、可验证；避免投机式抽象和隐藏兜底逻辑。

离线冒烟（无需下载数据）：
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

配置工具链：
```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
```

说明：本文档后半部分包含背景/路线图等内容；以 `configs/README.md` + `docs/CONFIG_ATLAS.md` 作为“可运行配置”的
最新依据。


```bash
# 0. 离线冒烟（仓库内置 Dummy_Data；无需下载数据）
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# 1. DG 示例（domain split；具体系统见 `task.target_system_id`）
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 2. CDDG 示例（多系统请调整 `task.target_system_id`）
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 3. 单系统 Few-shot 示例（FS）
python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 4. 跨系统 Few-shot 示例（GFS）
python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 5. HSE 预训练单阶段示例（通过 Pipeline_02_pretrain_fewshot）
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0

# 6. 面向 CDDG 的 HSE 预训练示例
python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

### Streamlit 图形界面（实验性）

使用 Streamlit 提供的图形界面运行实验（实验性功能）：

```bash
streamlit run streamlit_app.py
```

当前界面仍属实验性功能：基础配置编辑与 pipeline 启动可用；release 验证仍以 CLI demo 为准。

支持的 UI 工作流与验证命令见 `apps/streamlit/README.md`。

## 📘 使用指南

### 1. 配置文件详解 ⚙️

PHM-Vibench 使用 YAML 配置和维护中的 `base_configs + override` 模式。
当前可运行模板位于 `configs/demo/`，维护行由 `configs/config_registry.csv` 索引。

#### 核心配置特性
- **Base + Override 组合**：demo 配置复用共享 base block，并通过局部 override 调整。
- **点号参数覆盖**：CLI override（如 `trainer.num_epochs=1`）可直接更新嵌套字段。
- **注册表驱动文档**：`configs/config_registry.csv` 渲染为 `docs/CONFIG_ATLAS.md`。
- **检查工具**：`scripts.config_inspect` 输出最终配置、字段来源和实例化落点。

📖 **从这里开始**: [`configs/README.md`](configs/README.md)（30 秒冒烟 + override 规则 + 配置工具）

配置工具：
- Registry → Atlas：`python -m scripts.gen_config_atlas`（生成 `docs/CONFIG_ATLAS.md`）
- Inspect（最终配置/来源/实例化落点）：`python -m scripts.config_inspect --config <yaml> --override key=value`
- Schema 校验 demo：`python -m scripts.validate_configs`



### 配置参考

根 README 不维护完整字段表，权威配置参考位于：

- `configs/README.md`：组合规则、smoke 命令和 override 示例
- `docs/CONFIG_ATLAS.md`：由注册表生成的 atlas，包含 owner code、keyspace、最小运行命令和输出模式
- `src/task_factory/README.md`：任务注册表与 task/dataset 映射

不要在根 README 复制字段大表；它容易与注册表和 schema 漂移。

### 结果输出

Demo 配置通过 `environment.output_dir` 写出结果；当前维护 demo 使用 `results/demo/...`。生成的 atlas 中记录了每个配置的输出模式：`{environment.output_dir}/{experiment_name}/iter_{i}/`。

常见输出包括 Lightning checkpoint、CSV log、指标摘要和复制后的 resolved config，具体取决于 trainer 或 pipeline。`results/` 是运行产物目录，不是配置来源；配置定义仍以 `configs/` 为准。

可视化/绘图脚本位于 `plot/`，应读取明确的运行产物。

## 📂 项目结构

```bash
PHM-Vibench/
├── README.md
├── README_CN.md
├── main.py
├── configs/        # 实验 YAML + 注册表
├── src/            # pipelines + factories
├── dev/            # 开发辅助与实验性脚本（如 HSE demo）
├── docs/           # 文档
├── test/           # pytest 测试集
├── plot/           # 可视化/绘图工具
├── pic/            # README/docs 使用的图片
├── data/           # 仓库 smoke 数据 + 本地用户数据
└── results/        # 运行产出（除 README 外不纳入版本控制）
```

**核心目录说明**：

- **src/**: 模块化源代码，采用工厂模式设计
- **configs/**: 实验 YAML、base block、demo 和注册表
- **results/**: 运行产物；见 `results/README.md`
- **test/**: 维护中的 pytest 测试集
- **dev/**: 开发辅助与实验性脚本
- **plot/**: 可视化/绘图工具

## 🧑‍💻 开发指南

PHM-Vibench 的扩展方式是“工厂 + 注册表”，避免在 pipeline 中硬编码 import。

- 贡献指引（维护入口）：`CONTRIBUTING.md`（English）/ `CONTRIBUTING_CN.md`（中文）
- 扩展指南：
  - 数据集：`src/data_factory/contributing.md`
  - 模型：`src/model_factory/contributing.md`
  - 任务：`src/task_factory/contributing.md`
  - 训练器：`src/trainer_factory/contributing.md`
- 测试：`docs/testing.md` 与 `python -m pytest test/`

实现级文档建议从 `src/*_factory/README.md` 与 `configs/README.md` 开始读（字段说明 + wiring 入口）。

## 📃 用了该项目发表的文章

此处暂未收录公开发表文章。若您使用 PHM-Vibench 发表论文，请在此补充引用信息（论文 + 链接）。

## 🔮 项目路线图

- 稳定 v0.2.x 维护 demo 面和 release 证据。
- 保持配置注册表、生成的 atlas、支持组件和已知限制同步。
- 新增数据集、模型或任务路径时，同时补充注册表、验证命令和聚焦 smoke 证据。


## 👥 贡献者与社区

### 核心团队
- [Qi Li](https://github.com/liq22)
- [Xuan Li](https://github.com/Xuan423)
### Contributors

<a href="https://github.com/PHMbench/PHM-Vibench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PHMbench/PHM-Vibench" />
</a>


### 参与贡献
我们非常欢迎各种形式的贡献！无论是新功能开发、文档改进还是问题反馈。请参阅[贡献指南](CONTRIBUTING.md)了解详情。

### 社区交流
- bug、复现实验问题和聚焦功能请求请使用 GitHub issues。
- 内测协作请使用维护者提供的邀请渠道。
## 🏛 许可证

该基准测试平台采用 [Apache License (Version 2.0)](https://github.com/PHMbench/PHM-Vibench/blob/master/LICENSE) 许可。对于模型和数据集，请参考原始资源页面并遵循相应的许可证。

## 📎 引用方式

项目仍处于 alpha 阶段。稳定引用条目会随公开 release 补充；在此之前，请引用实验所用的仓库 commit 或 release tag。

---

<p align="center">如有任何问题或建议，请联系我们或提交 <a href="https://github.com/PHMbench/PHM-Vibench/issues">Issue</a>。</p>
