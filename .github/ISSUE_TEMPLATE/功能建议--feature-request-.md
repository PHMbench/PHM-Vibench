---
name: 功能建议（Feature Request）
about: 提出有明确边界的 PHM-Vibench 能力或改进
title: "[FEAT] "
labels: enhancement
assignees: ''
---

## 提交前检查

- [ ] 我已搜索现有 Issue 和 Pull Request。
- [ ] 我已阅读 [CONTRIBUTING_CN.md](https://github.com/PHMbench/PHM-Vibench/blob/main/CONTRIBUTING_CN.md) 和当前支持边界。
- [ ] 我已考虑该需求是否更适合本地实验、外部包或 PHM-Vibench 核心。

## 用户或研究场景

谁需要该能力？他们要完成的具体任务是什么？

## 当前限制

当前配置优先入口、factory、维护配置或已有扩展点为什么无法完成该任务？

尽可能提供当前最小命令或配置：

```bash
python main.py --config <yaml> [--override key=value ...]
```

## 建议行为

请描述用户可观察行为，而不是只描述实现想法。

```text
输入：
输出：
失败行为：
配置键：
```

选择预期成熟度：

- [ ] 维护中的公开能力
- [ ] 实验性能力
- [ ] 仅研究原型
- [ ] 文档或工具改进

## 更简单的替代方案

考虑过哪些更小的 workaround、配置变化、factory 扩展或外部工具？为什么不足？

## 架构与兼容性影响

```text
涉及的 factory 或模块：
新增依赖：
CLI/配置兼容性：
Checkpoint/数据兼容性：
CPU/GPU 影响：
迁移或弃用需求：
```

当现有 factory 可以表达该能力时，不要建议在 `main.py` 中加入组件专用分支。

## 证据和验证计划

哪些测试、fixture、配置、smoke 命令、artifact 或 benchmark protocol 可以证明功能有效？

若建议新增模型或算法，请链接主要论文或稳定规范，并说明参考代码 License。论文引用本身不构成运行证据。

## 维护成本与风险

说明潜在 owner、可选依赖处理、数据或 License 限制、失败模式，以及长期文档和测试成本。

## 补充信息

仅在有助于解释用户问题或兼容边界时添加图、示例或相关项目链接。
