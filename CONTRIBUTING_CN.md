# 为 PHMFactory 贡献代码

<div align="center">
  <p>
    <a href="CONTRIBUTING.md">English</a> |
    <a href="CONTRIBUTING_CN.md"><strong>中文</strong></a>
  </p>
</div>

修改仓库前，先阅读 [`README_CN.md`](README_CN.md)、[`CORE.md`](CORE.md) 和相关
Factory 指南。

PHMFactory 采用配置优先主路径。所有贡献必须保持：

```text
用户声明的实验 = 程序实际执行的实验
```

公共入口是：

```bash
phmfactory --config <yaml> [--local-config <yaml>] [--override key=value ...]
```

## 1. 先判断改动是否应进入核心仓库

提案必须说明：

```text
当前哪个用户动作或科学主张存在问题
已经核验的失败或不确定性
最小有效干预
考虑过的更简单方案
用户可观察的验收结果
```

不能仅以“未来可能支持更多数据集、后端、模型、分布式系统或工作流”为理由扩建主架构。
尚无稳定维护需求的研究变体应先放在 `configs/experiments/` 或独立研究仓库。

## 2. 永久约束

以下改动通常会被拒绝：

- 无消费者的 hash、checksum、digest、receipt、ledger 或 attestation；
- 数据源、后端、模型、任务、设备、loss、metric、checkpoint 或测试总体的静默 fallback；
- 通过 warning + continue 删除用户请求的样本或指标；
- 自动修复标签、通道、split、domain、patch 或 objective；
- Factory/Manager/Registry 套娃，或第二套 config/runtime/result authority；
- 隐藏真实根因的 broad exception wrapper；
- 没有当前维护消费者、只服务于假想未来的大重构；
- 重复 `CORE.md`、配置注册表或直接结果路径的 goal registry、policy tree 或 manifest family；
- 仅用于压制真实失败的测试修改。

优先顺序：

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

新抽象至少需要两个当前维护消费者，并且必须立即删除重复逻辑、不能增加新的用户概念。

## 3. 分支与 PR

长期分支：

```text
main  稳定/发布线
dev   日常集成线
```

常规分支从最新 `dev` 创建，PR 目标为 `dev`：

```bash
git switch dev
git pull --ff-only origin dev
git switch -c <type>/<short-topic>
```

建议前缀：

```text
fix/       正确性或兼容修复
docs/      仅文档
feat/      有界用户能力
test/      测试或合法 fixture
ci/        与路径相关的自动化
cleanup/   删除或简化
release/   经明确授权的发布工作
```

`main` 只接受经授权的 release promotion 或 emergency hotfix。常规 PR 不合入 `main`。

一个 PR 只保护一个主要不变量，并产生一个用户可观察结果。如果同时修改 Data、Model、
Task、Trainer、Pipeline、UI、release claim 和大范围文档，必须拆分。

常规 PR 可以 squash merge；rollback 即 revert 该 squash commit。

## 4. Factory 边界

```text
Data Factory    reader、metadata、selected IDs、dataset、sampler、loader
Model Factory   模型身份、构造、显式权重
Task Factory    任务身份、objective、metric 生命周期
Trainer Factory 设备、callback、checkpoint、fit/test 生命周期
Pipeline        编排、成功门控、直接结果位置
```

一个边界不得修复另一个边界。

Factory 贡献指南：

- [Data](src/data_factory/contributing.md)
- [Model](src/model_factory/contributing.md)
- [Task](src/task_factory/contributing.md)
- [Trainer](src/trainer_factory/contributing.md)

## 5. Bug 报告

请提供：

- 精确命令和退出码；
- config 路径、显式 local config 和全部 overrides；
- 预期与实际行为；
- 完整文本错误与 traceback；
- 相关 OS、Python、PyTorch、Lightning 和 CUDA 版本；
- 使用 Dummy 还是外部数据；
- 可共享的最小 config/fixture；
- 仓库 commit。

不要用截图替代文本日志。安全问题按 [`SECURITY.md`](SECURITY.md) 报告。

## 6. 数据与 reader

应提供：

- 数据来源、稳定版本/地址、引用、许可和再分发边界；
- 本地目录和 metadata 布局；
- reader 输入输出 shape、dtype、通道顺序、单位和预处理；
- 损坏输入的明确失败行为；
- 小型合法或合成 fixture；
- focused reader test 和最小 config；
- split 与 estimator 的 claim 边界。

Reader 不得生成替代数据、猜测不兼容格式、静默重排通道或在失败后跳过 selected file。
大型原始数据与权重通常不进入 Git。

## 7. Model、Task、Trainer 和配置

公共组件通常需要：

- 在现有 Factory 边界中的实现；
- 构造函数和 tensor/dtype/device 合同；
- focused 正反测试；
- 最小可运行配置；
- 明确的兼容/不兼容组合；
- 可选依赖与许可；
- 必要的 checkpoint 行为；
- 真实证据等级：discoverable、runnable、execution-verified 或 baseline-valid。

兼容组件不得通过修改 `main.py` 接入。源码能 import 不等于受到支持。

新实验配置放入 `configs/experiments/`。晋级为 maintained demo 或 baseline 必须有当前精确
执行证据和对应 registry 变更。

## 8. 文档

新增页面前先检查 [`docs/index.md`](docs/index.md)，优先更新现有 authority。

文档必须：

- 核验命令、路径、配置字段和文件名；
- 区分 current、experimental、deferred、unsupported 和 historical；
- 区分 smoke evidence 与科学 protocol evidence；
- 避免无证据的准确率、SOTA、兼容性、release 或数据集数量主张；
- 链接安装、配置、支持与限制 authority，而不是重新复制。

`README.md` 是用户入口，`CORE.md` 保存项目不变量，`KNOWN_LIMITATIONS.md` 保存当前边界，
release blocker 由 release readiness 文档负责。

## 9. 验证

先跑 focused tests。典型 runtime 改动：

```bash
python -m pytest <focused-tests> -q
python -m scripts.validate_configs
python -m scripts.validate_docs
phmfactory preflight --config smoke
phmfactory demo
```

仅在 source registry 或 generator 变化时运行生成文档检查：

```bash
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md

python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
```

只有改动可能影响某个真实数据协议时，才运行其 heavy workflow。不要让每个 PR 都下载外部数据。

记录结果：

```text
PASS
FAIL
EXPECTED FAILURE
NOT EXECUTED — <原因>
```

本地结果不等于 GitHub Actions 结果。

## 10. PR 描述

必须包含：

```text
已核验问题与根因
科学或用户不变量
最小改动
明确 non-goals
修改后的公共行为
focused tests 与结果
剩余限制
rollback：revert squash commit
```

相关检查失败时不得合并。不得降低测试、状态或 release 标准来通过 PR。只有原测试 authority
已经明确过时时才修改测试，并说明原因。

## 11. 当前 release 边界

源码版本为 `0.3.0rc1`，但当前 release readiness 受阻，直到至少一个精确真实数据实验在
当前源码上重新晋级为 `baseline_valid`。不得为了通过 gate 提前恢复该 claim。

IoTDB 和 `phm-data-factory` 仍为 optional/deferred，不是 core 依赖，也不得通过 fallback 或
宽泛 backend abstraction 接入。

## 12. 社区与许可

参与规则见 [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。贡献按仓库
[Apache License 2.0](LICENSE) 分发，但第三方组件和数据集许可需单独标明。
