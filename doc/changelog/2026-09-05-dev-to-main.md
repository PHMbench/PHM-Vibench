# 2026-09-06：dev → main 升级记录

本记录始建于 2026-09-05，随本次推广 PR 合入 main 生效。维护者已确认 THU 验证完成并再次授权合并。main 保留稳定／发布线定位；不将其改为自动接收所有研究分支的开发线。

## 推广范围

- 推广前 main：`21821c3a1bd7f3467e1590eb6f59c4ce11c45751`。
- 推广的 dev 快照：`9c0672f4beddd76cd58f50cc4efd0daaf8ad8923`，包含 PR #221 和 #224。
- dev 在该快照上领先 main 117 个提交，main 是其祖先。
- 使用 merge commit 保留历史，不对长期分支全量 squash，不强制推送。
- 推广分支额外更新两份 Changelog，并修复审阅发现的两个构建／导入问题；不修改实验配置、数据、模型或指标定义。
- 开放研究 PR #223 不在本次范围内。
- 不创建版本标签、GitHub Release，不上传 wheel 或发布到包索引。

## 安装与首次使用

PR #221 已完成正常 wheel 安装后的仓库外 Dummy 首跑。Demo 从已安装的数据包读取 metadata 和 raw CSV，在临时可写目录保存派生 cache；用户的显式路径覆盖仍然优先。

```bash
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

preflight 仅检查配置与早期运行条件，不代表训练或科学验证已经完成。运行成功后使用终端返回的当前结果路径。

## 配置与执行改进

- 普通实验必须显式提供配置，空命令不启动实验。
- 公共入口复用完整配置的严格接受边界。
- seed、iterations、num_epochs、分类路径 test_after_fit，以及 device/devices 已显式化。
- CUDA 请求不可满足时失败，不切换 CPU。
- 每次调用建立一个结果根目录，各 seed 使用其下的 iter_i。
- 维护分类路径恢复选中的 checkpoint 后评价，直接返回 checkpoint、指标和汇总路径。
- 用户可见的配置／运行摘要 hash 已移除；不据此宣称内部相关清理已经全部完成。

## 合并审阅中的最小修正

- 修正 `src/trainer_factory/extensions/__init__.py` 未闭合的文档字符串。改动为补一个引号；已用原源码复现 SyntaxError，并验证修复后可编译、模块可执行且导出列表仍为空。
- `pyproject.toml` 的 setuptools 下限由 69 改为 77.0.3，匹配现有 `license = "Apache-2.0"` SPDX 字段；保留许可证和运行依赖不变。依据为 [PyPA 构建配置说明](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)。
- 未采纳“Registry.get 不抛 KeyError”的评论：当前对象是 `src.utils.registry.Registry`，而非内置 dict，源码明确在缺键时抛出 KeyError。没有为此新增检查或包装。

## 用户升级注意事项

1. 使用 `trainer.devices`，不再使用公共配置字段 `trainer.gpus`。
2. 显式提供 `environment.seed`、`environment.iterations`、`trainer.num_epochs`、`trainer.device` 和 `trainer.devices`；分类实验还需 `trainer.test_after_fit`。
3. 使用 `phmfactory --config <yaml>`；机器路径使用显式 `--local-config` 或 `--override`。
4. 使用 CLI 返回的 result_dir、best_checkpoint、test_metrics 和 run_summary，不根据历史目录名或修改时间猜结果。
5. 历史论文子模块迁移见现有[迁移来源说明](../../paper/project/SOURCE_MAP.yaml)，不在本次重新导入。

## THU 验收与合并决定

维护者于 2026-09-06 明确确认，THU 下载与验证已在[共享工作会话](https://chatgpt.com/share/6a9d5c2a-4b1c-83ee-ba19-22b9b1025dc5)完成，并据此批准本次 dev → main 合并。

本次接受这一维护者验收，不重复下载 THU，不重跑已确认的 THU 实验，不再索取相同合并授权。执行环境未能取得共享页面正文，因此该项明确记录为维护者提供的验证确认，不声称本轮独立读取或复算了该会话中的数值。

本次源码推广与 benchmark 自动晋级、公开发版分别记录：

- 原有 MFPT `smoke_only` 状态不改写为 `baseline_valid`。
- 不伪造 THU 的配置、seed、准确率、macro-F1 或独立复算产物。
- 不修改发布检查器，不删除失败检查，不把 audit workflow 的绿色写成 release mode 通过。
- 现有程序化发布检查尚未登记这份 THU 验收；其 baseline blocker 保持可见。维护者本次批准的是在已有验证基础上的源码主线合并，不是包发布。

## 合并检查与后续

本次通过正常推广 PR 执行，检查最终候选的相关 GitHub Actions 和 review threads。完成合并后核对 main 包含整个 dev 快照，再同步 dev 的祖先关系；不纳入 #223 的未合并修改。

后续继续修复已知限制，但不重复创建已完成的首跑、配置接受边界或结果根任务。完整 benchmark 晋级和包发布仍按各自已有合同处理。

## 依据

- [PR #221](https://github.com/PHMbench/PHM-Vibench/pull/221)：installed-wheel 首跑修复与验证。
- [PR #224](https://github.com/PHMbench/PHM-Vibench/pull/224)：此前的升级准备记录。
- [维护者提供的验证会话](https://chatgpt.com/share/6a9d5c2a-4b1c-83ee-ba19-22b9b1025dc5)及本轮明确的合并确认。
- [现有发布检查合同](../../docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)：未被本次源码推广修改。
