# 2026-09-07：项目入口恢复、能力吸收与分支清理

本记录汇总已执行的内容；不把历史分支盘点当成完成迁移。主线推广随本记录所在 PR 合并生效，不创建软件发布标签或上传安装包。

## 已恢复的项目入口

[PR #226](https://github.com/PHMbench/PHM-Vibench/pull/226) 已合入 dev：恢复中英文 README 的真实项目结构、项目论文、相关方法、路线图、贡献者与社区入口。项目论文和 HSE 方法论文分栏展示，未恢复占位文章、过期季度承诺或带凭据的图表链接。

`paper/project/README.md` 改为直接描述配置、数据总体、划分、checkpoint、指标和结果，不再要求 hash-bound runs。详见 [项目主页升级记录](2026-09-07-project-homepage.md)。

## 已吸收的可执行能力

[PR #223](https://github.com/PHMbench/PHM-Vibench/pull/223) 已合入 dev，复用现有 XOAN 与 TSPN-UXFD 的同次前向轨迹，提供显式的 LLM 回调。完成真实模型集成测试，并修正推断分支、熵字段、重复类别名和干预信息问题。详见 [解释接口升级记录](2026-09-07-explanation-integration.md)。

没有整体合并旧 UXFD、生成模型、论文或历史运行框架；结构化引用检查不被表述为自然语言或物理机理忠实性证明。

## 已执行的分支清理

以最初盘点的 239 个远端分支为起点，以下三个批次实际删除了 173 个原有分支引用：

| 批次 | 已执行操作 | 数量 | 执行记录 |
| --- | --- | ---: | --- |
| 已合并内容 | 核验主线祖先关系或已合并 PR 的最终 head 后移除临时分支 | 145 | [运行记录](https://github.com/PHMbench/PHM-Vibench/actions/runs/34076710957) |
| 历史快照与准备链 | 先以少量 Git 归档标签保留原提交和祖先历史，再移除分支引用 | 22 | [首批](https://github.com/PHMbench/PHM-Vibench/actions/runs/34076710957)、[后续准备链](https://github.com/PHMbench/PHM-Vibench/actions/runs/34077535324) |
| 冗余祖先与等价修复 | 保留明确后继分支，或保留归档标签后删除引用 | 6 | [运行记录](https://github.com/PHMbench/PHM-Vibench/actions/runs/34080492307) |

第三批的 `fix/pipeline02-evaluation-completion` 虽然四个修改路径已经与 dev 相同，仍先用 `archive/pipeline02-evaluation-source-20260907` 保留其开发历史。归档标签不是软件发布版本，也不代表历史代码受到当前维护支持。

`cleanup/repo-slim-2026-07-05` 因当前文件仍引用该分支而保留，没有只因存在同提交归档标签就删除。`main`、`dev`、活动 PR 分支、受保护引用和仍有独有研究内容的分支不在上述删除名单中。

上述数字不包含本轮操作中新建并在完成后移除的临时分支。PR #223、#226 及本次推广的 topic 分支在合并后单独核对并清理，不把它们混入历史批次数字。

## 归档恢复

按需要读取指定的 Git 归档标签，不要重新合入整条历史运行线。例如：

```bash
git fetch origin tag archive/v030-cleanup-source-20260907
git switch -c inspect/historical-cleanup archive/v030-cleanup-source-20260907
```

归档标签保留 Git 已跟踪的提交历史；外部数据、未跟踪实验结果、LFS 实体和子模块仓库并不因一个标签自动得到额外备份。

## 保留待审内容

`Feature_factory-update`、`lq_merge_UXFD`、`research/2025-2026-method-expansion-current` 和仍含独有材料的论文分支继续保留。后续按一个方法、一个配置和对应测试选择性吸收；尚未核对到目标研究仓库的材料不得当作已迁移删除。

## 验证与范围

主页恢复经过文档和本地链接检查。PR #223 最终候选通过 Core quality gates、public package、repository layout 和 submodule policy，现有 CI 包含解释接口的真实模型测试、Dummy 训练与结果检查。

分支操作使用完整 Git 历史、当前保护/PR 状态和精确分支位置核验，没有重写 main/dev 历史，没有将一次性维护工作流引入主线，没有重复下载或重跑 THU，没有改写 MFPT 或包发布状态。
