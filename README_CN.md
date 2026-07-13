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

PHM-Vibench 将数据加载、模型构建、任务装配、训练和实验配置统一到一个维护入口：

```bash
python main.py --config <yaml> [--override key=value ...]
```

项目仍处于 alpha 阶段。当前 release-supported surface 明显小于仓库中全部文件和注册表条目。
请从维护中的 demo 开始；历史、参考和研究材料在没有独立运行证据前，不应视为已验证能力。

## 从这里开始

运行仓库自带的离线冒烟 demo：

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

在不修改维护 YAML 的情况下检查最终配置：

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

权威入口：

- [配置系统指南](configs/README.md)
- [生成的配置图谱](docs/CONFIG_ATLAS.md)
- [支持组件](SUPPORTED_COMPONENTS.md)
- [支持组合](SUPPORTED_COMBINATIONS.md)
- [已知限制](KNOWN_LIMITATIONS.md)
- [数据目录边界](data/README.md)
- [贡献指南](CONTRIBUTING_CN.md)

## 维护中的 demo 面

配置注册表当前将 7 个 demo 标记为 `sanity_ok`。该状态只代表配置具有功能冒烟证据，
不代表基准精度、SOTA 性能或任意组件之间都兼容。

| 场景 | 配置 | 数据要求 |
| --- | --- | --- |
| 离线冒烟 / DG | `configs/demo/00_smoke/dummy_dg.yaml` | 仓库自带 Dummy 数据 |
| 跨域 DG | `configs/demo/01_cross_domain/cwru_dg.yaml` | 本地 PHM-Vibench metadata/raw 数据 |
| 跨系统 CDDG | `configs/demo/02_cross_system/multi_system_cddg.yaml` | 本地 PHM-Vibench metadata/raw 数据 |
| 小样本 FS | `configs/demo/03_fewshot/cwru_protonet.yaml` | 本地 PHM-Vibench metadata/raw 数据 |
| 跨系统小样本 GFS | `configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml` | 本地 PHM-Vibench metadata/raw 数据 |
| HSE 预训练视角 | `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml` | 本地 PHM-Vibench metadata/raw 数据 |
| 面向 CDDG 的 HSE 预训练 | `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml` | 本地 PHM-Vibench metadata/raw 数据 |

当前 v0.2.0 支持文档将维护模型路径限定为 `ISFM/M_01_ISFM`、`E_01_HSE`、
`B_04_Dlinear`、`H_01_Linear_cla`，以及
[SUPPORTED_COMPONENTS.md](SUPPORTED_COMPONENTS.md) 中列出的任务组合。
注册表中存在条目本身并不构成支持声明。

## 安装

最小环境示例：

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

conda create -n phm-vibench python=3.10
conda activate phm-vibench
pip install -r requirements.txt
```

维护中的运行证据来自项目专用的 `LQ_signal` conda 环境。通用环境仍可能需要依赖或平台调整，
详见 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。

只有 Dummy 冒烟 demo 完全离线。其他 demo 应通过 override 指向本地数据根目录，
不要直接修改维护配置：

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/PHM-Vibench-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

处理后或原始数据也可能位于：

- [ModelScope 处理后文件](https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files)
- [PHMbench 原始数据组](https://www.modelscope.cn/datasets/PHMbench/PHMbench-raw_data)
- [Hugging Face 镜像](https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main)

使用或再分发前必须核验来源许可和可用性。

## 配置优先工作流

维护配置由五个逻辑块组成：

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

Demo 通过 `base_configs` 组合共享 block，再应用 YAML 和 CLI override。建立实验变体时：

1. 从 `configs/demo/` 复制最接近的模板到 `configs/experiments/`；
2. 只修改实验真正需要的字段；
3. 检查最终配置和字段来源；
4. 运行最小适用冒烟命令；
5. 若要提升为维护配置，同步更新注册表和生成的 atlas。

常用命令：

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <yaml> --override key=value
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
```

## 合并前验证门禁

开发时先运行最聚焦的测试；运行时或配置改动在合并前应执行维护门禁：

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
python -m pytest test/ -q
```

本地验证证据不能表述为 GitHub Actions 证据。仓库仍需要启用并设为 required 的 CI workflow，
才能自动执行这些门禁。

## 可选 Streamlit 工作区

Streamlit 工作区是同一配置优先契约的可选界面。它不替代 CLI release gate，
也不应直接导入 pipeline 内部实现。

```bash
streamlit run streamlit_app.py
```

配置、运行、结果查看和聚焦测试见[维护中的 Streamlit 指南](apps/streamlit/README.md)。

## 架构

```text
main.py
  └── 由 YAML 选择 pipeline
      ├── data factory
      ├── model factory
      ├── task factory
      └── trainer factory
```

主要目录：

- `configs/`：base block、维护 demo、实验配置和注册表
- `src/data_factory/`：metadata、reader、dataset、sampler 与数据构建
- `src/model_factory/`：模型家族、组件注册表与模型构建
- `src/task_factory/`：任务实现和任务注册表
- `src/trainer_factory/`：训练器实现
- `apps/streamlit/`：可选实验工作区
- `test/`：维护中的 pytest 门禁
- `docs/`：release、配置、迁移和工程文档
- `results/`：运行输出，不是配置事实来源

## 扩展 PHM-Vibench

扩展应留在 factory 边界内，不要在 `main.py` 中加入模型或数据集专用分支。

- [添加数据集或 reader](src/data_factory/contributing.md)
- [添加模型](src/model_factory/contributing.md)
- [添加任务](src/task_factory/contributing.md)
- [添加训练器](src/trainer_factory/contributing.md)

公开组件改动应同时包含实现、注册表或配置入口、聚焦测试、文档和适用的冒烟路径。
研究想法在协议与验证证据明确前，应留在清楚标注的 project 或 experiment 区域。

## 证据边界

PHM-Vibench 当前为有限配置矩阵提供功能冒烟和契约证据。它本身不能证明：

- 达到 SOTA 性能；
- 任意外部实验之间都能公平比较；
- 所有注册表组件组合均受支持；
- 所有引用数据集都可获得或允许再分发；
- 在未记录的数据与环境设置下仍可复现。

报告实验时，应记录准确的仓库 commit、配置、override、数据来源、随机种子和运行环境。

## 贡献、许可与引用

提交 PR 前请阅读 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。每个改动应保持小、明确、可审查，
并给出可复制的验证命令。

PHM-Vibench 使用 [Apache License 2.0](LICENSE)。数据集和模型产物可能适用其原始来源的独立许可。

项目仍处于 alpha 阶段。在稳定论文引用发布前，请引用实验所使用的准确 Git commit 或 release tag。
