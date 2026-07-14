# PHM-Vibench

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench 标志" width="300"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>面向工业振动故障诊断实验的配置优先工作台。</strong></p>

  <p>
    <img src="https://img.shields.io/badge/状态-alpha-orange" alt="状态：alpha"/>
    <img src="https://img.shields.io/badge/维护中%20demo-7-blue" alt="7 个维护中 demo"/>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>
</div>

PHM-Vibench 通过一个维护中的入口连接数据加载、模型构建、任务逻辑、训练和实验配置：

```bash
python main.py --config <yaml> [--override key=value ...]
```

项目仍处于 alpha 阶段。仓库中的文件或注册表条目数量大于当前 release-supported surface。
组件可被发现或导入，并不等于其已受支持；支持声明需要维护配置和运行证据。

## 当前维护范围

当前公开维护面包含 7 个 demo，覆盖：

- 使用 Dummy 数据的离线域泛化（DG）冒烟路径；
- 跨域 DG；
- 跨系统/跨数据集域泛化（CDDG）；
- 小样本（FS）和广义小样本（GFS）分类；
- 两个边界明确的 HSE 预训练视角。

准确的模型、任务、pipeline、数据和 trainer 组合见：

- [支持组件](SUPPORTED_COMPONENTS.md)
- [支持组合](SUPPORTED_COMBINATIONS.md)
- [已知限制](KNOWN_LIMITATIONS.md)

冒烟证据只说明软件路径能够运行，不代表基准精度、SOTA 性能、任意兼容性或数据再分发权利。

## 安装

Python 3.10 是当前文档和 CI 基线。

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

conda create -n phm-vibench python=3.10
conda activate phm-vibench
python -m pip install -r requirements.txt
```

CPU-only PyTorch、CUDA 选择、平台边界和环境检查见[安装指南](docs/installation.md)。

## 运行离线冒烟实验

以下命令使用仓库自带的 Dummy 数据和 CPU：

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

成功运行应以退出码 `0` 结束，打印完成信息，并在以下目录下生成产物：

```text
results/demo/dummy_dg_smoke/
```

配置检查、预期证据、外部数据 override 和后续实验步骤见[快速开始](docs/quickstart.md)。

## 文档导航

- [文档索引](docs/index.md)
- [安装](docs/installation.md)
- [快速开始](docs/quickstart.md)
- [配置系统](configs/README.md)
- [生成的配置图谱](docs/CONFIG_ATLAS.md)
- [数据目录与许可边界](data/README.md)
- [测试与证据](docs/testing.md)
- [故障排查](docs/troubleshooting.md)
- [开发者指南](docs/developer_guide.md)
- [中文贡献指南](CONTRIBUTING_CN.md)

历史、论文、开发日志和 Agent 工作流材料不属于当前用户路径。
其状态和保留规则记录在[文档审计](docs/DOCUMENTATION_AUDIT.md)中。

## 配置优先工作流

维护配置使用五个逻辑段：

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

个人实验变体应放在 `configs/experiments/`，不要直接修改维护 demo。运行前先检查解析后的值和来源：

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

`configs/config_registry.csv` 是配置清单的事实源；`docs/CONFIG_ATLAS.md` 由它生成，
不应手工编辑。

## 仓库结构

```text
configs/             base block、维护 demo、实验配置与注册表
src/data_factory/    metadata、reader、dataset、sampler 与数据构建
src/model_factory/   模型家族、组件与模型构建
src/task_factory/    任务实现、loss、metric 与任务注册表
src/trainer_factory/ trainer 构建与扩展
apps/streamlit/      围绕公开 CLI 的可选浏览器工作区
docs/                用户、开发、release、迁移与设计文档
test/                维护中的 pytest 测试集
```

扩展应留在现有 factory 边界内，不要在 `main.py` 中增加模型或数据集专用分支。

## 验证改动

开发时先运行最聚焦的测试。运行时或配置改动合并前，应执行适用的维护门禁：

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

当前分支实际自动执行的任务以 `.github/workflows/core-quality-gates.yml` 为准。
本地输出不能表述为 GitHub Actions 证据。详见[测试与证据](docs/testing.md)。

## 可选 Streamlit 工作区

Streamlit 工作区是围绕同一配置优先 CLI 的可选适配层，不是第二套训练框架。

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

使用说明见 [Streamlit 指南](docs/app_usage.md)。

## 参与贡献

提交 Issue 或 PR 前请阅读 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。
公开组件贡献应包含实现、注册表/配置可追踪性、聚焦测试、文档、适用的冒烟路径和明确兼容边界。

Factory 专项说明：

- [数据与 reader](src/data_factory/contributing.md)
- [模型](src/model_factory/contributing.md)
- [任务](src/task_factory/contributing.md)
- [Trainer](src/trainer_factory/contributing.md)

## 引用、许可与支持

在稳定论文或 DOI 发布前，请记录并引用实验使用的准确 Git tag 或 commit。
不要从 Dummy 冒烟结果或注册表清单推导科学结论。

PHM-Vibench 源代码使用 [Apache License 2.0](LICENSE)。数据集、预训练权重和第三方模型
可能适用其原始来源的独立许可。

可复现 Bug 和功能建议请使用 GitHub Issues。不要公开提交安全漏洞；请遵循 [SECURITY.md](SECURITY.md)。
