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

## 引用与许可

PHMFactory 使用 [Apache License 2.0](LICENSE)。引用信息见 [`CITATION.cff`](CITATION.cff)。
数据集与第三方组件许可相互独立。
