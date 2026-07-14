# 为 PHM-Vibench 做贡献

<div align="center">
  <a href="CONTRIBUTING.md">English</a> |
  <a href="CONTRIBUTING_CN.md"><strong>中文</strong></a>
</div>

感谢参与 PHM-Vibench。项目采用配置优先和工厂驱动架构；贡献应让一个具体行为更清楚、
更可靠或得到更明确的支持，而不是另建一套平行框架。

开始前请阅读：

- [文档索引](docs/index.md)
- [开发者指南](docs/developer_guide.md)
- [测试与证据](docs/testing.md)
- [配置系统](configs/README.md)
- [已知限制](KNOWN_LIMITATIONS.md)

维护中的运行契约是：

```bash
python main.py --config <yaml> [--override key=value ...]
```

维护中的配置结构保持为：

```text
environment / data / model / task / trainer
```

## 可以贡献什么

欢迎以下贡献：

- 报告可复现 Bug；
- 提出边界明确的功能建议；
- 修复运行时、配置、测试、文档或 CI 问题；
- 添加或改进 dataset reader、model、task、trainer、sampler 或 config；
- 在不重复事实源的前提下改进文档；
- 审查兼容性、可复现性、数据许可或 release 证据。

大型架构变更、新公开 pipeline、数据集再分发和科学性能声明应先讨论，再实现。

## 报告问题

提交前先搜索已有 Issue。一份有效的 Bug 报告应包含：

- 简明的问题描述；
- 完整复现步骤；
- 预期行为和实际行为；
- 操作系统和硬件；
- Python、PyTorch、CUDA、PyTorch Lightning 及相关包版本；
- 仓库 commit 或 release tag；
- 配置文件和全部 CLI override；
- 完整命令、退出码、traceback 和日志；
- 合法可共享的最小数据/配置复现案例。

环境信息可通过以下命令记录：

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

如果可能，先使用离线配置复现：

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

安全漏洞不得提交到公开 Issue，请遵循 [SECURITY.md](SECURITY.md)。

## 提出功能建议

功能建议应说明：

- 用户或研究场景；
- 当前限制；
- 期望行为和验收标准；
- 为什么该问题应由 PHM-Vibench 解决；
- 已考虑的更简单替代方案；
- 受影响的 factory、config、test 和文档；
- 向后兼容影响；
- 依赖、运行、数据和维护成本；
- 哪些内容仍明确不受支持。

注册表条目或论文引用本身不足以证明一个功能应进入 release-supported surface。

## 开发环境

使用 Python 3.10，与当前文档和 CI 基线一致：

```bash
git clone https://github.com/<your-account>/PHM-Vibench.git
cd PHM-Vibench

conda create -n phm-vibench-dev python=3.10
conda activate phm-vibench-dev
python -m pip install -r requirements.txt
```

CPU/CUDA 和平台边界见[安装指南](docs/installation.md)。

修改前先验证 checkout：

```bash
python main.py --help
python -m scripts.validate_configs
python -m scripts.validate_docs
```

## 分支与 Commit 规范

从最新 `main` 创建聚焦分支：

```bash
git switch main
git pull --ff-only origin main
git switch -c <type>/<short-topic>
```

建议前缀：

```text
fix/       缺陷或回归
feature/   新的边界明确能力
docs/      仅文档
test/      仅测试/证据
ci/        workflow 或自动化
cleanup/   经审查的删除或收敛
release/   仅 release 准备
```

使用清楚的祈使式 Commit，推荐 Conventional Commit：

```text
fix: reject unknown pipeline overrides
docs: clarify external data setup
test: cover sampler metadata errors
```

不要在一个 Commit 或 PR 中混合无关格式化、文件移动、文档清理和运行时行为。

## 提交代码修改

标准流程：

```text
创建聚焦分支
→ 完成最小一致修改
→ 添加/更新聚焦测试
→ 更新权威文档
→ 运行适用门禁
→ 检查最终 diff
→ 提交 Pull Request
```

保持以下边界：

- 数据集成：`src/data_factory/`；
- 模型构建：`src/model_factory/`；
- task/loss/metric：`src/task_factory/`；
- trainer 构建：`src/trainer_factory/`；
- 共享配置：`configs/`；
- 公开入口：`main.py`。

不要在 `main.py` 中加入 model、task、dataset 或 trainer 专用分支；不要创建第二套配置加载器、
registry 或训练框架。

行为变化必须满足以下之一：

- 向后兼容；
- 提供兼容 alias/adapter；
- 提供明确迁移说明和弃用路径。

静默 fallback 不是兼容策略。非法组合应尽早给出可操作错误。

## 贡献数据集或 Reader

请先阅读：

- [数据目录边界](data/README.md)
- [自定义数据集教程](docs/custom_dataset.md)
- [Data Factory 贡献指南](src/data_factory/contributing.md)

需要提供：

- 原始来源和稳定下载标识；
- 数据 License 和再分发限制；
- 原始/处理后格式；
- metadata 字段和单位；
- 预处理、window、split 方法；
- reader 实现及 registry/factory 可追踪路径；
- 合法的小 fixture 或 synthetic contract fixture；
- 首先放入 `configs/experiments/` 的配置；
- 最小 inspect 和运行命令；
- 预期输出结构，而不是编造性能指标；
- 已知限制与可复现性说明。

除非仓库政策明确允许，不要提交完整外部数据集或个人绝对路径。

## 贡献模型

请阅读 [Model Factory 贡献指南](src/model_factory/contributing.md)。

需要提供：

- 实现模块和公开模型标识；
- 输入/输出、shape、dtype、device 契约；
- 构造参数和默认值；
- model registry/config 可追踪路径；
- 聚焦构造和 forward 测试；
- task/loss 兼容性与应拒绝组合；
- 相关 checkpoint 行为；
- 最小实验配置和冒烟命令；
- 外部实现的论文来源和 License；
- 已知限制和不支持模式。

从 `configs/experiments/` 开始。提升到 `configs/demo/` 需要运行证据、registry 状态、文档和维护者审查。

## 贡献 Task、Sampler、Trainer 或 Pipeline

专项指南：

- [Task Factory](src/task_factory/contributing.md)
- [Trainer Factory](src/trainer_factory/contributing.md)
- [Data Sampler](src/data_factory/samplers/README.md)

应说明 batch 契约、model 输出契约、loss/metric 行为、配置参数、device 行为和非法组合。

只有现有 pipeline 无法表达一个完整运行阶段时，才应新增 pipeline；新 pipeline 仍须使用五段式配置和现有 factory。

## 贡献配置

本地或研究配置放在 `configs/experiments/`。

提出维护 demo 前执行：

```bash
python -m scripts.config_inspect --config <yaml> --override trainer.num_epochs=1
python -m scripts.validate_configs
python main.py --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

提升维护配置时更新：

- `configs/config_registry.csv`；
- 相关 config README；
- 通过 `python -m scripts.gen_config_atlas` 生成的 `docs/CONFIG_ATLAS.md`；
- 仅在公开支持边界变化时更新 `SUPPORTED_COMPONENTS.md`、
  `SUPPORTED_COMBINATIONS.md` 或 `KNOWN_LIMITATIONS.md`。

在冒烟命令实际通过前，不得把配置标记为 `sanity_ok`。

## 贡献文档

先确定读者和权威位置：

- 项目定位与最短路径：`README.md`；
- 安装：`docs/installation.md`；
- 首次运行：`docs/quickstart.md`；
- 配置：`configs/README.md`；
- 数据政策：`data/README.md`；
- 测试与证据：`docs/testing.md`；
- 架构与开发：`docs/developer_guide.md`；
- 贡献流程：`CONTRIBUTING.md`；
- 组件局部细节：最近的维护 `README.md`；
- 历史/研究：明确标记的历史或研究位置。

文档规则：

- 不把已有流程复制到第二个页面，使用链接；
- 验证全部命令、路径、配置键和相对链接；
- 缩写首次出现时解释；
- 区分维护中、实验性、计划中、已废弃和历史；
- 不添加无证据的性能、规模、兼容性或状态声明；
- 新页面加入 `docs/index.md` 或最近的局部导航；
- 代码行为变化必须在同一 PR 更新文档。

英文/中文根 README 与贡献指南应保持相同结构和支持边界。深层技术页面可以先保持英文，
不要创建无人维护的残缺翻译。

## 运行测试和质量检查

先运行最聚焦测试，再执行更广的维护门禁。详见[测试与证据](docs/testing.md)。

常用命令：

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
git diff --check
```

不适用的门禁应解释原因；无法执行时标记 `NOT_EXECUTED` 并说明限制。
不得把缺少数据/依赖或 skip 的测试写成通过。

## Pull Request 要求

PR 应包含：

- 问题与动机；
- 准确范围和 non-goals；
- 修改文件和公开行为；
- 兼容性和迁移影响；
- 执行命令、结果和环境；
- 新增/修改测试；
- 文档和 registry 变化；
- 风险、已知限制和回滚方式；
- 证据路径或 CI artifact；
- 相关 Issue、论文、数据集或设计来源。

请求审查前：

```bash
git status --short
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
```

保持 diff 可审查。聚焦 PR 在必需检查和维护者审查后优先 squash merge。

## 不会按原样接受的修改

- 个人绝对路径、凭据、cache、log 或本地 goal 包；
- 无行为等价证据的大范围重构；
- 没有通过冒烟路径的新公开 demo；
- 为掩盖失败而削弱测试或添加宽泛 skip；
- 无来源和 License 的第三方代码/数据；
- 无 provenance 的结果或 benchmark 声明；
- 重复的配置加载器、factory、registry 或入口；
- 没有清单和恢复证据的历史/研究文件大规模删除；
- AI 模板填充、占位联系方式或未经验证的声明。

## 社区与行为规范

保持尊重、准确和建设性。社区行为要求见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

非 Bug 问题可在 GitHub Discussions 中讨论；安全报告必须遵循 [SECURITY.md](SECURITY.md)。
