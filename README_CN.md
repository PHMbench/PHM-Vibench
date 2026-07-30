# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory 标志" width="280"/>

  <p>
    <a href="README.md">English</a> |
    <a href="README_CN.md"><strong>中文</strong></a>
  </p>

  <p><strong>面向工业振动信号的配置优先 PHM 研究与评估框架。</strong></p>

  <p>
    <img src="https://img.shields.io/badge/状态-alpha-orange" alt="状态：alpha"/>
    <img src="https://img.shields.io/badge/v0.3-预发布-blue" alt="v0.3 预发布"/>
    <a href="https://github.com/PHMbench/phmfactory/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/phmfactory/actions/workflows/core-quality-gates.yml/badge.svg" alt="核心质量门禁"/></a>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="Apache 2.0 许可"/>
  </p>
</div>

PHMFactory 通过一个公开 dispatcher 连接数据加载、模型构建、任务逻辑、训练、
评估和实验配置。以下三个入口具有相同语义：

```bash
python main.py --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
phmfactory --config <yaml> [--override key=value ...]
```

PHMFactory 是 PHM-Vibench 的 v0.3 后继项目。v0.3 兼容性版本增加公开的
`phmfactory` Python 包，同时将成熟的 `src.*` 运行时保留为受保护的内部内核。
项目仍处于 alpha 阶段；发布支持范围仅限
[SUPPORTED_COMBINATIONS.md](SUPPORTED_COMBINATIONS.md) 中记录的维护配置。

## 运行离线示例

安装核心环境后，运行仓库自带的 Dummy 配置：

```bash
git clone https://github.com/PHMbench/phmfactory.git
cd phmfactory
conda create -n phmfactory python=3.10
conda activate phmfactory
python -m pip install -r requirements.txt

python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

成功运行应正常退出、打印完成信息，并在 `results/demo/dummy_dg_smoke/`
下写入输出。该命令验证维护中的软件链路，不代表基准性能。

详细安装和故障排查：

- [安装指南](docs/installation.md)
- [快速开始](docs/quickstart.md)
- [已知限制](KNOWN_LIMITATIONS.md)
- [v0.2 到 v0.3 迁移说明](RELEASE_NOTES_v0.3.0.md)

## 公共数据包接口

v0.3 源码提供面向 CWRU 的 provider-neutral 数据包接口：

```text
metadata.xlsx          必需
RM_001_CWRU.h5         必需
corpus.xlsx            可选
```

公开命令：

```bash
python main.py data download --source huggingface
python main.py data download --source modelscope
python main.py data validate --path <bundle-dir>
python main.py data compare --left <hf-dir> --right <modelscope-dir>
```

最终 v0.3 发布仍被阻塞，直到 Hugging Face 与 ModelScope 均使用不可变 revision，
且必需文件的 SHA-256 完全一致。详见
[docs/CWRU_DEMO_V0_3.md](docs/CWRU_DEMO_V0_3.md)。

## 当前维护范围

维护中的 demo 覆盖：

- 完全离线的 Dummy 域泛化冒烟；
- 跨域和跨系统分类示例；
- 小样本和广义小样本示例；
- 边界明确的 HSE 预训练示例；
- 围绕同一公共 CLI 的可选 Streamlit 工作区。

维护范围之外的文件、注册表条目、研究笔记和历史配置不应自动视为受支持能力。
准确的模型、任务、数据和训练器组合见：

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

配置组合和优先级的权威说明位于 [configs/README.md](configs/README.md)，数据布局和
外部数据边界位于 [data/README.md](data/README.md)。

## 架构

```text
main.py / python -m phmfactory / phmfactory
  └── phmfactory.cli
      └── 已解析配置 + canonical Pipeline
          └── 受保护的 src 运行时
              ├── data factory
              ├── model factory
              ├── task factory
              └── trainer factory
```

主要目录：

- `phmfactory/`：公开 Python 包、CLI、配置解析器、Pipeline 注册表和数据 provider；
- `configs/`：base block、维护 demo、实验配置和配置注册表；
- `src/data_factory/`：metadata、reader、dataset、sampler 与数据装配；
- `src/model_factory/`：模型家族、组件与模型构建；
- `src/task_factory/`：任务、loss、metric 与任务注册表；
- `src/trainer_factory/`：训练器构建和扩展；
- `apps/streamlit/`：围绕同一 CLI 的可选浏览器工作区；
- `test/`：维护中的 pytest 测试；
- `docs/`：用户、开发、迁移、发布和历史文档。

v0.3 不会机械移动或重写成熟的数据 reader。添加数据集或模型时应扩展现有 factory，
不要在 `main.py` 中加入专用分支。

## 验证改动

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
python tools/repo/check_case_collisions.py
python tools/repo/check_release_readiness.py --mode audit
```

运行时或配置改动还应执行前面的离线冒烟命令。证据术语和聚焦命令见
[测试指南](docs/testing.md)。

## 可选 Streamlit 工作区

```bash
python -m pip install -r requirements.txt
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

`apps/streamlit/app.py` 是唯一维护中的 Web 入口。它将实验执行委托给公共 CLI，
不会定义第二套训练框架。

## 贡献与支持

提交 Issue 或 Pull Request 前请阅读 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)。
保持改动边界清晰，修改权威文档而不是复制内容，并提供准确的 commit、配置、
override、环境和日志。社区参与遵循 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

- Bug 与功能建议：[GitHub Issues](https://github.com/PHMbench/phmfactory/issues)
- 安全问题：[SECURITY.md](SECURITY.md)
- 开发流程：[docs/developer_guide.md](docs/developer_guide.md)
- 发布就绪状态：[docs/PHMFACTORY_V0_3_RELEASE_READINESS.md](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

## 引用与许可

PHMFactory 使用 [Apache License 2.0](LICENSE)。数据集和模型产物可能适用
原始来源的独立许可。

软件引用元数据见 [CITATION.cff](CITATION.cff)。请引用实验所使用的准确 Git commit
或 release tag，并记录配置、override、数据来源及 revision、随机种子和运行环境。
