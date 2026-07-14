# PHM-Vibench

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHM-Vibench 标志" width="280"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>面向工业振动故障诊断实验的配置优先工作台。</strong></p>

  <p>
    <img src="https://img.shields.io/badge/状态-alpha-orange" alt="状态：alpha"/>
    <img src="https://img.shields.io/badge/维护中%20demo-7-blue" alt="7 个维护中 demo"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="核心质量门禁"/></a>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>
</div>

PHM-Vibench 通过一个公开入口连接数据加载、模型构建、任务逻辑、训练和实验配置：

```bash
python main.py --config <yaml> [--override key=value ...]
```

项目仍处于 alpha 阶段。发布支持范围仅限
[SUPPORTED_COMBINATIONS.md](SUPPORTED_COMBINATIONS.md) 中列出的维护配置。
仓库中的其他文件、注册表条目、研究笔记和历史配置不应自动视为受支持能力。

## 运行离线示例

安装环境后，运行仓库自带的 Dummy 配置：

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench
conda create -n phm-vibench python=3.10
conda activate phm-vibench
python -m pip install -r requirements.txt

python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

成功运行应正常退出、打印完成信息，并在 `results/demo/dummy_dg_smoke/`
下写入输出。该命令验证软件链路，不代表基准性能。

详细安装、预期行为和故障排查：

- [安装指南](docs/installation.md)
- [快速开始](docs/quickstart.md)
- [已知限制](KNOWN_LIMITATIONS.md)

## 当前维护范围

当前维护中的 demo 覆盖：

- 离线 Dummy 域泛化冒烟；
- 跨域和跨系统分类示例；
- 小样本和广义小样本示例；
- 边界明确的 HSE 预训练示例。

只有 Dummy 示例完全离线。其他 demo 需要通过配置覆盖传入本地
PHM-Vibench metadata 和原始数据。准确的模型、任务、数据和训练器组合见：

- [支持组件](SUPPORTED_COMPONENTS.md)
- [支持组合](SUPPORTED_COMBINATIONS.md)
- [配置注册表](configs/config_registry.csv)
- [生成的配置图谱](docs/CONFIG_ATLAS.md)

`sanity_ok` 表示已有功能冒烟证据，不表示达到 SOTA、任意组件均兼容，
也不表示外部数据集可以自由再分发。

## 配置优先工作流

维护配置使用五个逻辑块：

```text
environment / data / model / task / trainer
```

从 `configs/demo/` 中最接近的配置开始，将本地实验变体放在
`configs/experiments/`，机器相关值使用 CLI override：

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

配置组合和优先级的权威说明位于[配置指南](configs/README.md)，数据布局和
外部数据边界位于[数据指南](data/README.md)。

## 架构

```text
main.py
  └── 配置选择的 pipeline
      ├── data factory
      ├── model factory
      ├── task factory
      └── trainer factory
```

主要目录：

- `configs/`：base block、维护 demo、实验配置和配置注册表；
- `src/data_factory/`：metadata、reader、dataset、sampler 与数据装配；
- `src/model_factory/`：模型家族、组件与模型构建；
- `src/task_factory/`：任务、loss、metric 与任务注册表；
- `src/trainer_factory/`：训练器构建和扩展；
- `apps/streamlit/`：围绕同一 CLI 的可选浏览器工作区；
- `test/`：维护中的 pytest 测试；
- `docs/`：用户、开发、发布和历史文档。

添加数据集或模型时，应扩展现有 factory，不要在 `main.py` 中加入专用分支。

## 验证改动

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
```

运行时或配置改动还应执行前面的离线冒烟命令。GitHub Actions 当前执行
文档/配置一致性和聚焦的 UXFD 装配测试。证据术语和更聚焦的命令见
[测试指南](docs/testing.md)。

## 文档

通过[文档索引](docs/index.md)查找安装、配置、数据、开发、测试、Streamlit、
发布和历史材料。

可选 Web 界面：

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

其本地单 worker 边界见 [apps/streamlit/README.md](apps/streamlit/README.md)。

## 贡献与支持

提交 Issue 或 Pull Request 前请阅读 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。
保持改动小而可审查，修改权威文档而不是复制内容，并提供准确的 commit、
配置、override、环境和日志。社区参与遵循
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

- Bug 与功能建议：[GitHub Issues](https://github.com/PHMbench/PHM-Vibench/issues)
- 安全问题：[SECURITY.md](SECURITY.md)
- 开发流程：[docs/developer_guide.md](docs/developer_guide.md)

## 引用与许可

PHM-Vibench 使用 [Apache License 2.0](LICENSE)。数据集和模型产物可能适用
原始来源的独立许可。

软件引用元数据见 [CITATION.cff](CITATION.cff)。请引用实验所使用的准确 Git
commit 或 release tag，并记录配置、override、数据来源、随机种子和运行环境。
