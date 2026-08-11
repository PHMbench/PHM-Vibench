# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory 标志" width="280"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>面向工业信号、强调可复现性的配置优先 PHM 实验框架。</strong></p>

  <p>
    <img src="https://img.shields.io/badge/状态-alpha-orange" alt="状态：alpha"/>
    <img src="https://img.shields.io/badge/v0.3-预发布-blue" alt="v0.3 预发布"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="核心质量门禁"/></a>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>
</div>

> **当前仓库身份。** 项目名称和 Python 包名已经统一为 **PHMFactory**，但在
> v0.3 预发布阶段，GitHub 仓库仍是
> [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench)。正式改名前请始终
> 使用这里给出的真实仓库地址。

PHMFactory 用一个配置优先入口连接数据加载、模型构建、任务逻辑、训练、评估和
运行记录。用户选择一个维护中的配置，只覆盖本机路径或实验参数，即可通过命令行、
Python 模块或兼容入口执行同一份实验语义。

## 先运行完全离线的示例

下面的路径只使用仓库自带的合成数据，不下载外部数据集：

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

首次运行成功时，应满足：

- `doctor` 的必需检查全部显示 `PASS`；
- `preflight` 打印 `status=passed`，且不会启动训练；
- `demo` 在 CPU 上完成一次 Dummy 数据的 data → model → task → trainer 链路；
- 终端打印 `run_manifest.json` 的路径；
- 结果写入 `results/demo/dummy_dg_smoke/`；
- 三条命令的进程退出码均为 `0`。

命令失败时请保留完整终端输出，并根据[快速开始](docs/quickstart.md)中的对应故障
处理。CPU-only PyTorch、GPU 和不同操作系统的安装方式见[安装指南](docs/installation.md)。

## 根据任务选择文档

| 你的目标 | 从这里开始 |
| --- | --- |
| 理解第一次运行及其输出 | [快速开始](docs/quickstart.md) |
| 在 CPU、GPU、Linux、macOS 或 Windows 上安装 | [安装指南](docs/installation.md) |
| 运行一个已有的维护实验 | [配置系统](configs/README.md) |
| 接入本地 PHM 数据 | [数据目录](data/README.md)和[自定义数据集](docs/custom_dataset.md) |
| 选择或新增模型 | [模型工厂](src/model_factory/README_CN.md) |
| 选择或新增任务 | [任务工厂](src/task_factory/README.md) |
| 使用浏览器界面 | [Streamlit 工作区](apps/streamlit/README.md) |
| 扩展或维护框架 | [开发者指南](docs/developer_guide.md) |
| 核对当前真正支持的组合 | [支持组合](SUPPORTED_COMBINATIONS.md) |

完整文档地图见 [docs/index.md](docs/index.md)。

## 配置的五个逻辑块

维护中的配置统一使用：

```yaml
environment:  # 输出路径、随机种子、重复次数和进程级设置
  ...
data:         # metadata、原始数据根目录、窗口和 worker
  ...
model:        # 模型家族及模型专有参数
  ...
task:         # 诊断、域泛化、小样本或预训练逻辑
  ...
trainer:      # 设备、epoch、精度、日志和 checkpoint
  ...
```

顶层 `pipeline` 只负责选择编排路径。新增数据集、模型、任务和训练器时，原则上应扩展
对应 factory，不应在 `main.py` 中加入项目专用分支。

本地实验从 `configs/demo/` 中最接近的维护配置开始，研究变体放到
`configs/experiments/`，本机路径通过显式 override 传入：

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

预检通过后，去掉 `preflight` 即可执行同一份配置：

```bash
phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

配置组合和优先级的权威说明位于 [configs/README.md](configs/README.md)。

## 公开入口

以下三个进程入口具有相同的配置和退出码语义：

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

正常使用时推荐安装后的 `phmfactory` 命令；`python main.py` 只作为仓库兼容入口保留。
需要直接读取结构化 Python 返回值的调用者可以导入 `phmfactory.cli.main`。

常用的轻量命令：

```bash
phmfactory doctor
phmfactory preflight --config <preset-or-yaml>
phmfactory demo
phmfactory data --help
```

## 如何理解“支持”

PHMFactory 明确区分：

```text
discoverable  = 源码或注册表条目存在
runnable      = 已建立可审查的执行路径
supported     = 维护配置具有当前的功能冒烟结果
```

必须满足：

```text
supported ⊆ runnable ⊆ discoverable
```

源码文件、模型注册表条目或 import 成功都不自动等于“已支持”。当前维护范围由配置注册表
和运行时 descriptor 生成：

- [支持组件](SUPPORTED_COMPONENTS.md)
- [支持组合](SUPPORTED_COMBINATIONS.md)
- [配置注册表](configs/config_registry.csv)
- [配置图谱](docs/CONFIG_ATLAS.md)

`sanity_ok` 只表示已有边界明确的功能冒烟，不表示达到 SOTA、任意组件都可组合，也不
表示外部数据可以重新分发。

## 可选 Streamlit 工作区

Web 工作区只是同一公共 CLI 的适配层，不是第二套训练框架：

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

首次使用选择 **Use safe CPU smoke defaults**。界面可以准备配置、验证、启动公共命令，
并查看日志和产物。其单 worker 边界和故障处理见
[apps/streamlit/README.md](apps/streamlit/README.md)。

## 开发者架构

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

- `phmfactory/`：公开包、命令、配置解析、Pipeline descriptor 和运行控制层；
- `configs/`：复用块、维护 demo、研究实验和配置注册表；
- `src/data_factory/`：metadata、reader、dataset、sampler 和数据装配；
- `src/model_factory/`：模型家族和模型构造；
- `src/task_factory/`：任务、损失、指标和任务构造；
- `src/trainer_factory/`：训练器构造和扩展；
- `apps/streamlit/`：可选浏览器工作区；
- `test/`：维护中的 pytest 测试；
- `docs/`：用户、扩展、开发、发布和历史文档。

提交 PR 前运行：

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
python -m pytest test/ -q
```

聚焦测试和验证术语见 [docs/testing.md](docs/testing.md)。

## 分支策略

`main` 是面向用户的稳定默认分支，`dev` 是集成分支。常规功能、修复、文档、测试、CI、
清理和迁移 PR 都应以最新 `dev` 为起点并合入 `dev`。

只有明确授权的发布提升 PR 或紧急 hotfix 可以指向 `main`；hotfix 必须同步回 `dev`。
完整流程见 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。

## 当前预发布限制

PHMFactory 仍是 alpha 阶段的 `0.3.0.dev0` 源码版本：

- 只有 Dummy demo 完全离线并随仓库提供；
- 大部分真实数据 demo 需要本地 metadata 和原始数据；
- CWRU provider revision 和必需文件 hash 尚未最终冻结；
- GitHub 仓库尚未改名；
- 当前不宣称已有最终 `v0.3.0` tag 或包发布；
- experimental Pipeline 和未列出的模型/任务组合不属于发布支持范围。

进行发布或 benchmark 声明前，请阅读[已知限制](KNOWN_LIMITATIONS.md)和
[v0.3 发布就绪状态](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)。

## 贡献、支持与引用

提交 Issue 或 PR 前请阅读 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。问题报告应包含
准确 commit、配置、override、环境、数据来源和完整错误输出。

- Bug 与功能建议：[GitHub Issues](https://github.com/PHMbench/PHM-Vibench/issues)
- 安全问题：[SECURITY.md](SECURITY.md)
- 开发流程：[docs/developer_guide.md](docs/developer_guide.md)
- 发布状态：[docs/PHMFACTORY_V0_3_RELEASE_READINESS.md](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

PHMFactory 使用 [Apache License 2.0](LICENSE)。数据集和模型产物可能适用独立来源许可。
软件引用信息见 [CITATION.cff](CITATION.cff)，每次实验应记录并引用准确的 commit 或 tag。
