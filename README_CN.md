# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory 标志" width="300"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>配置优先、失败即停的工业信号 PHM 实验框架。</strong></p>
  <p><em>用一份配置贯通数据、模型、任务、训练与评价，不用隐式兜底改变实验。</em></p>

  <p>
    <img src="https://img.shields.io/badge/状态-alpha-orange" alt="状态：alpha"/>
    <img src="https://img.shields.io/badge/版本-0.3.0.dev0-blue" alt="版本 0.3.0.dev0"/>
    <img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB" alt="Python 3.10 或更高版本"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="核心质量门禁"/></a>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>

  <p>
    <a href="#快速开始">快速开始</a> •
    <a href="#为什么是-phmfactory">为什么使用</a> •
    <a href="#运行机制">运行机制</a> •
    <a href="#按任务选择文档">使用文档</a> •
    <a href="#支持边界">支持边界</a> •
    <a href="#贡献与引用">参与贡献</a>
  </p>
</div>

---

> **当前仓库身份。** 项目名称和 Python 包名已经统一为 **PHMFactory**。在 v0.3
> 预发布阶段，GitHub 仓库仍为
> [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench)。在后续改名被明确
> 公布之前，请始终使用这里给出的真实仓库地址。

PHMFactory 面向工业信号故障诊断及相关 PHM 实验，提供模块化、配置驱动的研究运行
框架。一份解析完成的配置直接连接数据、模型、任务目标、训练器、checkpoint、评价与
用户可见结果。

仓库的核心不变量是：

```text
用户声明的实验 = 程序实际执行的实验
```

当请求的数据、任务、设备、目标函数、checkpoint 或评价无法按配置执行时，程序应给出
明确错误并停止，不得偷偷换成更容易运行的另一套实验。

## 为什么是 PHMFactory

工业 PHM 仓库常常拥有较强的算法，却缺少清晰的实验边界。结果可能是代码成功退出，
但真实执行的数据划分、任务、设备、目标函数或估计量已经偏离用户声明。

| PHM 实验中的常见问题 | PHMFactory 的处理方式 |
| --- | --- |
| 默认值在不知情时改变实验 | 显式解析配置，并在不满足条件时立即失败 |
| 数据、模型、任务和训练器彼此耦合 | 四个 Factory 分工清晰，责任可以独立审查 |
| 评价结果受无约束随机采样影响 | 维护路径中的验证与测试采用确定性边界 |
| 源码文件存在就被误认为“已支持” | 自动生成支持表，区分可发现、可运行和维护证据 |
| 替换一个组件需要修改运行时 | 组件选择保持配置优先 |

由此形成两条直接的设计规则：

```text
替换一个模块
→ 只修改该模块及其配置
```

```text
训练可以具有随机性
→ 评价仍必须对应一个定义明确的估计量
```

<details>
<summary><strong>PHMFactory 是什么，也不是什么</strong></summary>

PHMFactory 主要提供：

- 一条统一的配置优先运行路径；
- 清晰的数据、模型、任务与训练器边界；
- 可执行的错误提示，而不是静默兜底；
- 维护中的冒烟配置和自动生成的支持文档；
- CLI 与可选 Streamlit 共用的同一运行时。

PHMFactory 不宣称仓库中的所有实现都可以任意组合，也不因源码存在、import 成功或一次
运行成功就宣称结果已经达到 benchmark-valid、SOTA 或允许重新分发。此类结论必须建立
在明确维护的配置和科学闭合的协议上。

</details>

## 快速开始

首次运行完全离线，只使用仓库自带的合成数据，不下载外部数据集。

### 1. 安装

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

项目要求 Python 3.10 或更高版本。CPU-only PyTorch、GPU 和不同操作系统的安装方式见
[安装指南](docs/installation.md)。

### 2. 检查、预检与运行

```bash
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

三条命令分别承担不同责任：

```text
doctor
→ 检查安装环境和仓库必需资源

preflight
→ 解析并验证同一份配置，但不启动训练

demo
→ 在 CPU 上执行 data → model → task → trainer 的完整受控链路
```

成功运行后，结果位于：

```text
results/demo/dummy_dg_smoke/
```

终端会打印运行记录的位置。预期输出和故障排查见[快速开始](docs/quickstart.md)。

### 离线示例能够证明什么

它能够证明软件已正确安装，并且维护中的公共运行时可以完成一次边界明确的实验。它不
证明真实数据 benchmark 已经有效，也不证明算法性能优越或仓库内任意组件都可以组合。

## 运行机制

```text
YAML 或 preset
    ↓
解析完成的配置
    ↓
canonical Pipeline
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → best checkpoint → evaluation → finite metrics
    ↓
用户可见结果路径
```

### Factory 责任边界

| 边界 | 负责 | 不应负责 |
| --- | --- | --- |
| **Data Factory** | reader、metadata、selected IDs、dataset、sampler、loader | 修复模型或任务配置 |
| **Model Factory** | 模型身份、构造、显式请求的外部权重 | 选择数据划分或移动模型设备 |
| **Task Factory** | 任务身份、目标函数、指标生命周期 | 控制硬件或改写用户任务 |
| **Trainer Factory** | 设备、callback、checkpoint、logger、训练与评价生命周期 | 补齐缺失的任务或数据语义 |
| **Pipeline** | 编排、成功门控、结果位置 | 静默修复任何 Factory 输入 |

这些边界刻意保持简洁。新增数据集、模型、任务和训练器时，通常应扩展对应 Factory，
而不是在 `main.py` 中增加项目专用分支。

## 核心能力

| 能力 | 用户可观察行为 |
| --- | --- |
| **配置优先运行** | CLI、Python 模块和兼容入口使用同一份解析配置。 |
| **科学语义 fail-fast** | 非法标签、不可用设备、不可能的数据划分、缺失 checkpoint 和非法指标会终止运行。 |
| **确定性评价边界** | 维护中的验证与测试不依赖无约束的 patch 或增强随机性。 |
| **离线首次运行** | `doctor`、`preflight` 和 `demo` 不需要下载外部数据。 |
| **模块化替换** | 数据、模型、任务和训练器的选择显式且可以独立审查。 |
| **单一运行时** | CLI 是权威入口；Streamlit 只是同一命令的界面适配。 |

## 按任务选择文档

| 你的目标 | 从这里开始 |
| --- | --- |
| 理解第一次运行及其输出 | [快速开始](docs/quickstart.md) |
| 在 CPU、GPU、Linux、macOS 或 Windows 上安装 | [安装指南](docs/installation.md) |
| 运行已有的维护实验 | [配置系统](configs/README.md) |
| 接入本地 PHM 数据 | [数据目录](data/README.md)和[自定义数据集](docs/custom_dataset.md) |
| 选择或新增模型 | [模型工厂](src/model_factory/README_CN.md) |
| 选择或新增任务 | [任务工厂](src/task_factory/README.md) |
| 使用浏览器工作区 | [Streamlit 工作区](apps/streamlit/README.md) |
| 扩展或维护框架 | [开发者指南](docs/developer_guide.md) |
| 核对真正维护的组合 | [支持组合](SUPPORTED_COMBINATIONS.md) |

完整文档地图见 [docs/index.md](docs/index.md)。

## 配置合同

维护实验统一使用一个顶层 `pipeline` 和五个逻辑块：

```yaml
pipeline: "Pipeline_01_Fault_Diagnosis"

environment:  # 输出路径、随机种子、重复次数和进程级设置
  ...
data:         # metadata、原始数据根目录、窗口、worker 和采样策略
  ...
model:        # 模型家族及模型专有参数
  ...
task:         # 诊断、泛化、小样本或预训练目标
  ...
trainer:      # 设备、epoch、精度、日志和 checkpoint 行为
  ...
```

本地实验从 `configs/demo/` 中最接近的维护配置开始，研究变体放入
`configs/experiments/`，本机路径通过显式 override 传入：

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

预检通过后，执行同一份配置：

```bash
phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

配置组合与优先级的权威说明见 [configs/README.md](configs/README.md)。

<details>
<summary><strong>公共入口</strong></summary>

以下三个进程入口具有相同的配置和退出码语义：

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

正常使用推荐安装后的 `phmfactory` 命令；`python main.py` 只作为仓库兼容入口保留。

常用轻量命令：

```bash
phmfactory doctor
phmfactory preflight --config <preset-or-yaml>
phmfactory demo
phmfactory data --help
```

</details>

## 支持边界

PHMFactory 明确区分三个层级：

```text
discoverable  = 源码或注册表条目存在
runnable      = 已建立经过审查的执行路径
supported     = 维护配置具有当前功能冒烟证据
```

必须满足：

```text
supported ⊆ runnable ⊆ discoverable
```

源码文件、注册表行或 import 成功都不是支持声明。当前维护范围由仓库配置和运行时描述
自动生成：

- [支持组件](SUPPORTED_COMPONENTS.md)
- [支持组合](SUPPORTED_COMBINATIONS.md)
- [配置注册表](configs/config_registry.csv)
- [配置图谱](docs/CONFIG_ATLAS.md)

`sanity_ok` 只表示已有边界明确的功能冒烟，不表示 benchmark-valid、SOTA、允许任意重分发
外部数据，或任意组件笛卡尔积都能组合。

## 可选 Streamlit 工作区

浏览器工作区适配同一条公共 CLI：

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

首次使用选择 **Use safe CPU smoke defaults**。界面可以准备配置、验证、启动公共命令并
显示日志和输出。单 worker 边界和故障处理见
[apps/streamlit/README.md](apps/streamlit/README.md)。

<details>
<summary><strong>开发者架构与仓库地图</strong></summary>

```text
phmfactory 命令 / python -m phmfactory / main.py
  └── 公共命令路由
      └── 已解析配置 + canonical Pipeline
          └── 受保护的 src 运行时
              ├── data factory
              ├── model factory
              ├── task factory
              └── trainer factory
```

主要目录：

- `phmfactory/`：公开包、命令、配置解析、Pipeline descriptor 和运行控制；
- `configs/`：复用块、维护 demo、研究实验和配置注册表；
- `src/data_factory/`：metadata、reader、dataset、sampler 和数据装配；
- `src/model_factory/`：模型家族和模型构造；
- `src/task_factory/`：任务、目标函数、指标和任务构造；
- `src/trainer_factory/`：训练器构造和扩展；
- `apps/streamlit/`：可选浏览器工作区；
- `test/`：维护中的 pytest 测试；
- `docs/`：用户、扩展、开发、发布和历史文档。

提交审阅前运行：

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
python -m pytest test/ -q
```

聚焦测试和术语说明见 [docs/testing.md](docs/testing.md)。

</details>

## 当前预发布边界

PHMFactory 仍是 alpha 阶段的 `0.3.0.dev0` 源码版本：

- 只有 Dummy demo 完全离线并随仓库提供；
- 大部分真实数据配置需要本地 metadata 和原始信号；
- 尚无真实数据配置被提升为首个科学闭合的 `baseline_valid` 参考实验；
- CWRU provider、reader 和最终 acceptance 条件仍在收敛；
- GitHub 仓库尚未改名；
- 当前不宣称已有最终 `v0.3.0` tag 或正式包发布；
- experimental Pipeline 和未列出的模型/任务组合不属于发布支持范围。

进行发布或 benchmark 声明前，请阅读[已知限制](KNOWN_LIMITATIONS.md)和
[v0.3 发布就绪状态](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)。

## 分支策略

`main` 是面向用户的稳定默认分支，`dev` 是集成分支。常规功能、修复、文档、测试、CI、
清理和迁移 PR 都从最新 `dev` 开始并指向 `dev`。

只有明确授权的 release promotion PR 或紧急 hotfix 可以指向 `main`。完整流程见
[CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。

## 贡献与引用

提交 Issue 或 PR 前请阅读 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。有效的问题报告应
包含准确 commit、配置、override、环境、数据来源和完整终端输出。

- Bug 与功能建议：[GitHub Issues](https://github.com/PHMbench/PHM-Vibench/issues)
- 开发流程：[开发者指南](docs/developer_guide.md)
- 发布状态：[v0.3 发布状态](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- 软件引用信息：[CITATION.cff](CITATION.cff)
- 许可：[Apache License 2.0](LICENSE)

数据集和模型产物可能适用独立来源许可。每次实验应记录并引用准确的 commit 或 tag。
