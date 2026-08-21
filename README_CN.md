# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory 标志" width="300"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>配置优先、失败即停的工业信号 PHM 实验框架。</strong></p>
  <p><em>声明一个实验，执行同一个实验。</em></p>

  <p>
    <img src="https://img.shields.io/badge/状态-发布受阻-critical" alt="等待当前源码基线验证"/>
    <img src="https://img.shields.io/badge/版本-0.3.0rc1-blue" alt="版本 0.3.0rc1"/>
    <img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB" alt="Python 3.10 或更高版本"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="核心质量门禁"/></a>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>

  <p>
    <a href="#快速开始">快速开始</a> •
    <a href="#科学合同">科学合同</a> •
    <a href="#运行机制">运行机制</a> •
    <a href="#支持边界">支持边界</a> •
    <a href="#文档导航">文档导航</a>
  </p>
</div>

---

PHMFactory 是面向故障诊断及相关 PHM 实验的模块化研究运行框架。一份可见配置直接连接
数据、模型、任务目标、训练器、checkpoint 选择、评价和用户结果路径。

仓库仍为 [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench)；项目与
Python 包名分别为 **PHMFactory** 和 `phmfactory`。

> **当前源码状态。** 源码版本为 `0.3.0rc1`，但发布 readiness 当前受阻。离线 Dummy
> 主路径已维护；MFPT 透明实验是等待当前源码科学复核的 `smoke_only` 候选。当前没有
> current-source `baseline_valid` 注册项，也没有 RC1 tag、GitHub Release 或包索引发布。

项目最小核心 authority 见 [`CORE.md`](CORE.md)。

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

成功后终端直接输出：

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

这些路径是用户结果 authority。运行成功不依赖 manifest、attestation、evidence index、
receipt、ledger 或 hash。

Dummy 示例只能证明安装和维护运行路径能够完成一次边界明确的实验。它不证明真实数据
benchmark、强诊断性能、SOTA 或仓库组件可以任意组合。

详细步骤见[快速开始](docs/quickstart.md)和[安装指南](docs/installation.md)。

## 科学合同

核心不变量是：

```text
用户声明的实验 = 程序实际执行的实验
```

实验表示为：

$$
\mathcal E=(\mathcal D,\Pi,f_\theta,\mathcal L,\widehat R),
$$

其中 $\mathcal D$ 为数据总体，$\Pi$ 为协议，$f_\theta$ 为实际模型，$\mathcal L$ 为实际
优化目标，$\widehat R$ 为最终报告的估计量。五项都必须与可见请求一致。

因此，PHMFactory 会直接拒绝，而不是静默修复：

- 缺失或损坏的配置；
- 不可用的显式设备；
- 不可能的数据划分或目标域；
- 非法标签、reader 输出、patch 尺寸、metric 或 regularization；
- 缺失或不兼容的选定 checkpoint；
- 空、不完整、非标量、NaN 或 Inf 的评价结果；
- 请求后端或数据源失败后自动切换到另一条路径。

训练可以随机，但验证和测试仍必须形成定义清楚的估计量。

## 运行机制

```text
YAML 或 preset
    ↓
一份解析完成的配置
    ↓
canonical Pipeline
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → 选定 checkpoint → test → 完整有限指标
    ↓
直接结果路径
```

| 边界 | 负责 | 禁止负责 |
| --- | --- | --- |
| **Data Factory** | reader、metadata、selected IDs、dataset、sampler、loader | 修复模型、任务、设备或指标配置 |
| **Model Factory** | 模型身份、构造、显式权重 | 选择 split 或移动模型设备 |
| **Task Factory** | 任务身份、目标函数、metric 生命周期 | 控制硬件或 checkpoint 选择 |
| **Trainer Factory** | 设备、callback、checkpoint、fit/test 生命周期 | 补造缺失的数据或任务语义 |
| **Pipeline** | 编排、成功门控、直接结果位置 | 静默修复任何 Factory 输入 |

替换一个兼容模块时，应只修改该模块及其配置，而不是修改其他 Factory 或公共入口。

## 配置

维护实验统一使用顶层 Pipeline 和五个逻辑块：

```yaml
pipeline: "Pipeline_01_Fault_Diagnosis"

environment:  # 输出根目录、seed、重复次数
  ...
data:         # metadata、原始数据、窗口和采样
  ...
model:        # 模型身份与参数
  ...
task:         # objective、metrics、optimizer/scheduler
  ...
trainer:      # device、epoch、checkpoint 与日志生命周期
  ...
```

Preflight 和运行必须使用同一组可见输入：

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

本机专用 YAML 只有通过 `--local-config` 显式传入时才生效。组合和优先级见
[`configs/README.md`](configs/README.md)。

## 支持边界

PHMFactory 明确区分：

```text
discoverable       源码或注册项存在
runnable           已有经过审阅的执行路径
execution-verified 精确命令具有当前受控执行证据
baseline-valid     精确完整实验通过当前科学协议
```

`baseline-valid` 只属于完整配置，不能由 import 成功、源码存在、其他配置或历史结果推导。

当前 authority：

- [支持组件](SUPPORTED_COMPONENTS.md)
- [支持组合](SUPPORTED_COMBINATIONS.md)
- [配置注册表](configs/config_registry.csv)
- [已知限制](KNOWN_LIMITATIONS.md)
- [发布 readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

MFPT + `GlobalAverageLinear` 仍是透明真实数据候选。其历史三 seed 结果在当前 metric 与复核
门禁通过前，不作为 current-source 晋级证据。

## 公共入口

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

正常使用推荐 `phmfactory`；`python main.py` 只作为仓库兼容入口。

轻量命令：

```bash
phmfactory doctor
phmfactory preflight --config <preset-or-yaml>
phmfactory demo
phmfactory data --help
```

## 文档导航

| 目标 | 入口 |
| --- | --- |
| 安装并完成首次运行 | [快速开始](docs/quickstart.md) |
| 理解项目科学与工程 authority | [核心合同](CORE.md) |
| 配置实验 | [配置指南](configs/README.md) |
| 接入本地数据 | [数据布局](data/README.md)与[自定义数据集](docs/custom_dataset.md) |
| 选择或新增模型 | [Model Factory](src/model_factory/README.md) |
| 选择或新增任务 | [Task Factory](src/task_factory/README.md) |
| 使用可选浏览器工作区 | [Streamlit](apps/streamlit/README.md) |
| 扩展仓库 | [贡献指南](CONTRIBUTING_CN.md) |
| 核对发布 blocker | [Release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) |

完整导航见 [`docs/index.md`](docs/index.md)。

## 开发原则

遵循奥卡姆剃刀：

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

禁止新增无消费者 hash、静默 fallback、面向假想未来的大抽象、Factory/Manager/Registry
套娃或第二套 config/runtime/result authority。一个 PR 只保护一个主要不变量，并产生一个
用户可观察结果。

常规 PR 从最新 `dev` 创建并合入 `dev`。广泛修改前先阅读 [`CORE.md`](CORE.md)、
[`AGENTS.md`](AGENTS.md)和[`CONTRIBUTING_CN.md`](CONTRIBUTING_CN.md)。

## 引用与许可

PHMFactory 使用 [Apache License 2.0](LICENSE)。引用信息见 [`CITATION.cff`](CITATION.cff)。
数据集与第三方组件许可相互独立，重新分发或商业使用前必须单独核验。
