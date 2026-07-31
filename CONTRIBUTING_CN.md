# 为 PHM-Vibench 做贡献

<div align="center">
  <p>
    <a href="CONTRIBUTING.md">English</a> |
    <a href="CONTRIBUTING_CN.md"><strong>中文</strong></a>
  </p>
</div>

PHM-Vibench 欢迎聚焦的 Bug 修复、测试、文档、配置、数据 reader、模型、任务、
训练器和可复现性改进。

项目采用配置优先入口：

```bash
python main.py --config <yaml> [--override key=value ...]
```

贡献必须保持五个公开配置块：

```text
environment / data / model / task / trainer
```

进行较大修改前，请先阅读[文档索引](docs/index.md)、
[开发指南](docs/developer_guide.md)和[测试指南](docs/testing.md)。

## 提交 Issue 或 PR 之前

1. 搜索现有 Issue 和 Pull Request。
2. 用户可见问题尽可能在最新 `main` 上复现；开发修复应在最新 `dev` 上验证。
3. 从 `configs/demo/` 中最接近的维护配置开始。
4. 将本地变体放在 `configs/experiments/` 或未跟踪的本地配置中；不要提交个人绝对路径。
5. 将无关的运行时、文档、数据产物和仓库清理改动拆开。
6. 不要因为注册表中存在条目，就宣称该组件已受支持。

重大架构调整、新 pipeline、公开兼容性变化或大型数据/模型贡献，应先创建 Issue 讨论。

## 报告 Bug

使用 Bug Report 模板，并提供：

- 简明的问题描述；
- 可重复的复现步骤；
- 预期行为和实际行为；
- 操作系统与硬件；
- Python、PyTorch、PyTorch Lightning，以及适用时的 CUDA 版本；
- 仓库 commit 或 release tag；
- 配置路径和全部 CLI override；
- 数据来源，以及使用的是 Dummy 数据还是外部数据；
- 文本形式的完整错误日志；
- 可共享的最小配置或最小复现测试。

缺少依赖、无效配置和代码缺陷是不同类型的问题。请提供命令退出码；能提供文本日志时，
不要只提交截图。

安全漏洞不得通过公开 Issue 报告，请遵循 [SECURITY.md](SECURITY.md)。

## 提出功能建议

功能建议应说明：

- 用户或研究场景；
- 当前限制；
- 建议行为；
- 为什么应由 PHM-Vibench 解决，而不是保留为本地实验；
- 已考虑的更简单替代方案；
- 对兼容性、依赖、测试、文档和长期维护的影响；
- 该能力应定位为维护中、实验性还是研究性。

新增方法应附主要论文或稳定技术来源，但论文存在本身不能证明仓库实现可运行。

## 分支模型与开发环境

仓库保留两个长期分支：

```text
main  面向用户的稳定分支；是 clone、用户文档、发布和受支持状态的权威来源
dev   集成与开发分支；是常规 Pull Request 的目标分支
```

常规功能、修复、文档、测试、CI、清理和迁移 PR 必须指向 `dev`。topic branch
应从最新 `dev` 创建：

```bash
git switch dev
git pull --ff-only origin dev
git switch -c <type>/<short-topic>
```

`main` 只接受：

1. 经过明确授权、由 `dev` 或 `release/<version>` 发起的发布提升 PR；或
2. 从 `main` 创建的已授权紧急 hotfix，且合并后必须立即同步或回补到 `dev`。

`main` 与 `dev` 都只能通过 Pull Request 更新，不得直接 push。已经打开的
canonical v0.3 集成 PR #127 早于本策略，是唯一的过渡例外：其 source ancestry
合同要求继续指向 `main`。#127 合并后，在接收新的开发工作前，必须先将产生的
`main` merge commit 同步到 `dev`。

按照[安装指南](docs/installation.md)配置本地环境。

建议前缀：

```text
fix/       Bug 或兼容性修复
feat/      用户可见能力
docs/      仅文档改动
test/      测试和 fixture
ci/        Workflow 或自动化
cleanup/   有边界的删除或仓库清理
release/   发布准备
```

保持分支聚焦。Commit 建议使用：

```text
<type>: <祈使式摘要>
```

例如：

```text
fix: reject unknown task registry entries
test: cover TSPN_UXFD CPU assembly
docs: clarify external data layout
```

不需要人为制造大量 commit。没有 provenance 或 ancestry 合同的常规 topic PR
可以 squash merge 到 `dev`；发布提升以及依赖提交祖先关系的迁移/来源证明 PR
必须使用 merge commit，禁止 squash 或 rebase。

## 提交代码贡献

标准流程：

```text
从 dev 创建聚焦分支
→ 完成最小且完整的改动
→ 添加或更新聚焦测试
→ 更新权威文档
→ 运行适用门禁
→ 检查 diff 中是否夹带无关修改
→ 向 dev 提交包含准确证据的 PR
```

架构约束：

- 扩展 `src/data_factory/`、`src/model_factory/`、`src/task_factory/` 或
  `src/trainer_factory/`，不要在 `main.py` 中加入组件专用分支；
- 保持公开 CLI 和五块配置模型兼容；
- 非法组合应尽早失败，并给出可理解错误；
- 避免隐藏 fallback、静默部分 checkpoint 加载和机器专用默认值，除非它们是已测试、已文档化的兼容行为；
- 行为变化必须提供迁移说明或兼容层；
- 不得通过修改测试来掩盖真实失败。

Factory 专用指南：

- [数据和 reader](src/data_factory/contributing.md)
- [模型](src/model_factory/contributing.md)
- [任务](src/task_factory/contributing.md)
- [训练器](src/trainer_factory/contributing.md)

## 贡献数据集或 reader

应提供适用的全部信息：

- 数据集名称、原始来源、稳定下载地址和引用；
- License 与再分发限制；
- 目录和 metadata 格式；
- reader 实现及输入/输出契约；
- 预处理和数据划分方法；
- `configs/experiments/` 下的配置，或将其提升为维护 demo 的充分理由；
- 原始数据不能再分发时，提供合法的小 fixture 或 synthetic 契约测试；
- 实际执行的 inspect 和 smoke 命令；
- 预期输出结构，而不是编造基准指标；
- 已知限制和可复现性说明。

大型数据通常不应进入 Git。参考资料和 metadata 的存在不代表拥有再分发权。
详见 [data/README.md](data/README.md)。

## 贡献模型、任务、训练器或配置

公开组件通常应同时包含：

- 正确 factory 边界内的实现；
- 必要的注册表或配置入口；
- 构造参数、batch、tensor shape、dtype、device 和输出契约；
- 正向和负向聚焦测试；
- 适用时的 checkpoint 或状态行为；
- 最小可运行配置；
- 明确的兼容与不兼容组件；
- 复制或改编代码的来源和 License；
- 已知限制与证据等级。

本地实验放在 `configs/experiments/`。提升到 `configs/demo/` 和 `sanity_ok`
必须有可审查的运行证据。只有公开维护面确实变化时，才更新
`configs/config_registry.csv`、重新生成 `docs/CONFIG_ATLAS.md` 并同步支持文档。

## 贡献文档

新增页面前先检查[文档索引](docs/index.md)。若已有权威入口，应修改权威入口而不是再复制一份。

文档贡献必须：

- 明确读者和任务；
- 定义首次出现的缩写和术语；
- 内部文件优先使用仓库相对链接；
- 验证命令、路径、配置键和文件名；
- 区分维护中、实验性、计划中、已废弃和历史行为；
- 不加入无证据的性能、兼容性、数据集数量或成熟度声明；
- 将新的维护页面加入文档导航；
- 删除会破坏外部引用或历史证据时，优先归档。

不要再次复制安装、quickstart、配置优先级、测试门禁或支持矩阵；请链接到权威页面。

## 运行验证

从 [docs/testing.md](docs/testing.md) 选择适用命令。通用本地门禁：

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
python -m pytest test/ -q
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

先运行聚焦测试。若文档 PR 没有修改命令、配置或运行时声明，可使用更窄门禁，
但必须在 PR 中说明未运行其他命令的原因。

结果必须准确记录为：

```text
PASS
FAIL
EXPECTED FAILURE
NOT EXECUTED — <原因>
```

本地证据不能表述为 GitHub Actions 证据。

## 提交 Pull Request

常规 PR 必须以 `dev` 为 base。指向 `main` 的 PR 必须在描述中注明已批准的
发布提升或紧急 hotfix 例外。

PR 必须说明：

- 问题和修改理由；
- 准确范围与明确的 non-goals；
- 公开行为变化和迁移影响；
- 涉及的文件或组件；
- 已运行命令和结果；
- 新增或修改的测试；
- 文档和注册表更新；
- 风险与限制；
- 回滚方法；
- 未包含本地 goal pack、缓存、日志、凭据、原始数据和机器路径。

不要把大范围格式化与行为改动混在一起。不要单独提交生成的 atlas 而不提交注册表来源变化。
不要为了让 PR 通过而降低测试或支持标准。

合并前需要至少一名维护者 review 并通过必要检查。常规 topic PR 合并到 `dev`；
由 `dev` 或 `release/<version>` 提升到 `main` 时使用 merge commit，以保留发布边界和
`dev` 祖先关系。

## 通常不会接受的改动

- 直接 push 到 `main` 或 `dev`，或常规 PR 绕过 `dev` 直接指向 `main`；
- 个人路径、凭据或私有基础设施细节；
- 绕过现有配置和 factory 契约的平行框架；
- 将无关运行时、清理、文档和研究工作混在一个巨型 PR；
- 未核对仓库事实的生成式、复制式或 AI 堆砌文档；
- 无可复现证据的精度、效率、SOTA 或通用兼容性声明；
- 未提供来源和 License 的第三方代码；
- 只通过 skip、捕获或压制失败来“修复”测试；
- 未审查 License 和存储策略的大型数据或模型产物。

## 社区与许可

社区参与遵循 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。提交贡献即表示同意其可在
仓库 [Apache License 2.0](LICENSE) 下分发，同时遵守单独标注的第三方许可。

一般问题可使用 GitHub Issues，或在启用时使用 Discussions。安全与行为问题不要放在公开 Issue 中。
