# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory 标志" width="260"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>面向工业信号的配置驱动 PHM 实验框架。</strong></p>

  <p>
    <img src="https://img.shields.io/badge/状态-发布受阻-critical" alt="发布受阻"/>
    <img src="https://img.shields.io/badge/版本-0.3.0rc1-blue" alt="版本 0.3.0rc1"/>
    <img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB" alt="Python 3.10 或更高版本"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="核心质量检查"/></a>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>
</div>

PHMFactory 使用一份可见配置运行故障诊断及相关 PHM 实验。程序实际使用的数据、模型、
任务、训练策略、checkpoint 和评价方式，必须与用户声明一致。

仓库仍为 [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench)；项目、
Python 包和命令分别为 **PHMFactory**、`phmfactory` 和 `phmfactory`。

> **当前状态：** 离线 Dummy 主路径已维护；MFPT 透明实验仍为 `smoke_only`，等待当前
> 源码复核。当前没有包索引发布，也没有 `baseline_valid` 参考配置，因此发布仍受阻。

## 快速开始

首次运行完全离线，只使用仓库内置 Dummy 数据。

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .

phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

成功后终端直接给出：

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

按这些路径检查结果即可。Dummy 示例只验证安装和维护运行路径，不代表真实数据 benchmark
或算法性能。

完整步骤见[快速开始](docs/quickstart.md)，平台说明见[安装指南](docs/installation.md)。

## 运行实验

正式实验必须显式指定配置：

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1
```

本机专用配置只有通过 `--local-config` 显式传入时才生效。配置组合与优先级见
[`configs/README.md`](configs/README.md)。

## 项目结构

```text
PHM-Vibench/
├── phmfactory/           # 公共命令与配置入口
├── configs/              # 示例和研究实验配置
├── src/
│   ├── data_factory/     # 数据读取、数据集、采样与加载
│   ├── model_factory/    # 模型与表示模块
│   ├── task_factory/     # 目标函数、指标与优化策略
│   ├── trainer_factory/  # 设备、回调与模型选择
│   └── runtime/          # 实验执行
├── data/                 # 内置 Dummy 数据与数据布局说明
├── test/                 # 运行路径与组件测试
├── apps/streamlit/       # 可选浏览器工作区
├── docs/                 # 用户与开发文档
├── doc/changelog/        # 升级记录
└── paper/project/        # 研究源码与迁移说明
```

运行实验从 `configs/` 开始；新增组件从相应 Factory 开始。实验结果以命令返回的实际路径
为准，目录树不规定固定的结果保存位置。

## 运行结构

```text
解析完成的配置
    ↓
canonical Pipeline
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → 选定 checkpoint → test → 有限指标
    ↓
直接结果路径
```

| 边界 | 责任 |
| --- | --- |
| Data Factory | metadata、reader、样本选择、dataset、sampler、loader |
| Model Factory | 模型身份、模型构造、显式权重 |
| Task Factory | 目标函数、指标、optimizer 和 scheduler |
| Trainer Factory | 设备、callback、checkpoint、fit/test 生命周期 |
| Pipeline | 编排与成功判定 |

替换一个兼容模块时，应只修改该模块及其配置，不应修改其他 Factory 或公共命令入口。

## 失败行为

问题应在负责该问题的边界直接失败。数据源、设备、任务、checkpoint 或指标失败后，
程序不得自动改跑更容易的实验。错误信息应说明请求值、实际值、预期合同和最小修复。

## 支持术语

| 术语 | 含义 |
| --- | --- |
| `discoverable` | 源码或注册项存在 |
| `runnable` | 已有经过审阅的执行路径 |
| `execution-verified` | 精确命令具有当前受控执行证据 |
| `baseline-valid` | 精确完整实验通过当前科学协议 |

支持状态属于精确配置，不能由源码存在或 import 成功推导。详情见
[支持组合](SUPPORTED_COMBINATIONS.md)、[已知限制](KNOWN_LIMITATIONS.md)和
[发布状态](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)。

## 文档

| 任务 | 入口 |
| --- | --- |
| 安装并完成首次运行 | [快速开始](docs/quickstart.md) |
| 配置实验 | [配置指南](configs/README.md) |
| 接入本地数据 | [数据布局](data/README.md) |
| 选择或新增模型 | [Model Factory](src/model_factory/README.md) |
| 选择或新增任务 | [Task Factory](src/task_factory/README.md) |
| 配置训练 | [Trainer Factory](src/trainer_factory/README.md) |
| 使用浏览器工作区 | [Streamlit](apps/streamlit/README.md) |
| 贡献代码 | [贡献指南](CONTRIBUTING_CN.md) |
| 理解项目约束 | [核心合同](CORE.md) |

完整导航见 [`docs/index.md`](docs/index.md)。

## 开发原则

遵循奥卡姆剃刀：

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

一个 PR 只保护一个主要不变量，并产生一个用户可观察结果。优先使用直接、清楚的代码和
错误信息，不增加兜底、包装层、重复注册表或面向假想未来的抽象。注释应解释科学或兼容
原因，不重复代码本身。

常规工作从最新 `dev` 创建并合入 `dev`。广泛修改前先阅读 [`CORE.md`](CORE.md)和
[`CONTRIBUTING_CN.md`](CONTRIBUTING_CN.md)。

## 论文与研究

### 项目论文

Qi Li, Bojian Chen, Xuan Li, Qitong Chen, Liang Chen, Changqing Shen, Lu Lu,
Zhaoye Qin, Fulei Chu.
**[PHM-Vibench: A Unified and Factory-Style Vibration Benchmarking Framework for the Foundation Model Era](https://papers.phmsociety.org/index.php/phmap/article/view/4303)**.
*PHM Society Asia-Pacific Conference*, 5(1)，2025 年会议论文集；
在线发表日期为 2026 年 1 月 13 日。DOI：[10.36001/phmap.2025.v5i1.4303](https://doi.org/10.36001/phmap.2025.v5i1.4303)。

该论文介绍 PHM-Vibench。当前 PHMFactory 源码的能力范围，以
[支持组合](SUPPORTED_COMBINATIONS.md)和[已知限制](KNOWN_LIMITATIONS.md)为准。

### 相关方法

Qi Li, Bojian Chen, Qitong Chen, Xuan Li, Zhaoye Qin, Fulei Chu.
**[HSE: A plug-and-play module for unified fault diagnosis foundation models](https://doi.org/10.1016/j.inffus.2025.103277)**.
*Information Fusion*, 123, 103277, 2025。

HSE 在此列为相关表示方法，不表示论文的全部实验使用了当前软件版本。在研项目和历史
论文源码不属于已发表结果，相关说明见[研究源码入口](paper/project/README.md)。

### 使用本项目的研究

收录研究时，请通过 [Issue](https://github.com/PHMbench/PHM-Vibench/issues) 提供论文链接、
代码或实验配置，以及实际使用的软件版本。本栏目只收录具有明确项目使用关系的研究。

## 项目路线图

| 阶段 | 内容 |
| --- | --- |
| 已有能力 | 配置驱动 CLI、离线 Dummy 首跑、直接结果路径 |
| 下一步 | 补齐声明指标与结果语义，重新验证真实数据参考实验 |
| 研究方向 | 验证可解释模型的语言解释与异构信号扩展，再决定是否纳入维护示例 |

[升级记录](doc/changelog/)说明已完成的修改，
[发布状态](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)说明当前阻塞。研究方向不代表发布承诺。

## 贡献者与社区

项目核心贡献者包括 [Qi Li](https://github.com/liq22) 和
[Xuan Li](https://github.com/Xuan423)。完整贡献历史见
[全部贡献者](https://github.com/PHMbench/PHM-Vibench/graphs/contributors)。

可复现问题与具体功能建议请提交 [Issue](https://github.com/PHMbench/PHM-Vibench/issues)，
使用问题与研究讨论请前往 [Discussions](https://github.com/PHMbench/PHM-Vibench/discussions)。
参与开发前请阅读[贡献指南](CONTRIBUTING_CN.md)和[行为准则](CODE_OF_CONDUCT.md)。

[Star 历史](https://www.star-history.com/#PHMbench/PHM-Vibench&Date)

## 引用与许可

PHMFactory 使用 [Apache License 2.0](LICENSE)。引用信息见 [`CITATION.cff`](CITATION.cff)。
数据集与第三方组件许可相互独立。
