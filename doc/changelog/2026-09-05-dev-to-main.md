# 2026-09-05：dev → main 升级记录（方案 A）

状态：升级准备，尚未合入 main。main 继续表示稳定／发布线，不改为公开开发分支。

## 本次范围

- 当前 main：`21821c3a1bd7f3467e1590eb6f59c4ce11c45751`。
- 待推广源码：`a23d2e44e59e0803c36f4a52e431876dc8da5c01`，包含已合并的 PR #221。
- 后续推广保留 dev 的提交历史，使用 merge commit，不把长期分支全量 squash。
- 尚未合并的研究 PR #223 不在本次范围内。
- 本记录不创建版本标签、GitHub Release 或包索引发布。

## 已完成的升级

### 安装与首次使用

PR #221 已合并：正常安装 wheel 后，可以离开仓库目录运行内置 Dummy 实验。Demo 从已安装的数据包读取输入，在临时可写目录保存派生 cache；用户的显式路径覆盖仍然优先。

维护入口为：

```bash
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

其中 preflight 检查配置与早期运行条件，不代表已完成训练或科学验证。

### 配置与执行

- 普通实验必须显式提供配置，空命令不启动实验。
- 公共入口复用同一个完整配置严格接受边界。
- seed、iterations、num_epochs、分类路径 test_after_fit，以及 device/devices 已显式化。
- CUDA 请求不可满足时失败，不切换 CPU。
- 每次调用建立一个结果根目录，各 seed 使用其下的 iter_i。
- 维护分类路径恢复选中的 checkpoint 后评价，直接返回 checkpoint、指标和汇总路径。
- 用户可见的配置／运行摘要 hash 已移除；不据此宣称内部清理全部完成。

## 用户升级注意事项

1. 使用 `trainer.devices`，不再使用公共配置字段 `trainer.gpus`。
2. 保留显式的 `environment.seed`、`environment.iterations`、`trainer.num_epochs`、`trainer.device` 和 `trainer.devices`；分类实验还需 `trainer.test_after_fit`。
3. 运行使用 `phmfactory --config <yaml>`；机器路径使用显式 `--local-config` 或 `--override`。
4. 使用 CLI 返回的当前 result_dir、best_checkpoint、test_metrics 和 run_summary，不根据历史目录名或修改时间猜结果。
5. 历史论文子模块迁移见现有 [迁移来源说明](../../paper/project/SOURCE_MAP.yaml)，不在本次重新导入。

## THU 已有验证

维护者已确认 THU 数据集下载与本地验证完成。现有本地审查摘要也记录了小规模真实 THU 实验可执行。本次复用这一事实，不重复下载 THU，不新建 benchmark，不编造数值或运行产物。

该证据目前支持“THU 本地执行验证完成”。本次可查材料没有列出其 exact config、运行源码版本、划分记录、逐 seed 指标和独立指标复算结果，因此这里不将 THU 改称 baseline_valid，也不把它当成 MFPT 协议的重新验证。

## 方案 A 的剩余合并条件

当前 [发布检查合同](../../docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) 仍为 BLOCKED，缺少当前源码上的 baseline_valid 参考。正常安装／Dummy 检查通过不等于该科学条件通过；audit workflow 绿色也不等于 release mode 通过。

后续仅补齐本次推广需要的证据：

1. 定位并复核已存在的 THU 配置和验证产物，核对源码、数据总体、split、selected checkpoint、声明指标及独立复算；缺失项明确标记，不重复下载数据。
2. 按已有标准判断该 exact experiment 能否作为当前真实数据验收，不能只修改 registry 状态来放行。
3. 在最终推广候选上运行相关用户／科学合同检查，并要求实际 release mode 通过。
4. 条件满足后执行已获维护者授权的 main 推广 PR，保留历史，随后同步 dev；将本记录状态更新为“已合并”。不再额外索取同一项合并授权。

不扩展到 #223、其他研究方法、依赖大重构或新的审计体系。当前 main 不变；本记录不将尚未完成的稳定推广写成已完成。

## 依据

- [PR #221](https://github.com/PHMbench/PHM-Vibench/pull/221)：已合并的 installed-wheel 首跑修复与验证范围。
- [发布检查合同](../../docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)：稳定推广所沿用的科学条件。
- 维护者于 2026-09-05 提供的 THU 验证确认，以及本地仓库审查摘要；未随摘要提供的原始运行结果不在本记录中重建。
