---
name: 功能建议（中文）
about: 提出边界明确的 PHM-Vibench 能力
title: "[FEAT] "
labels: enhancement
assignees: ''
---

## 提交前检查

- [ ] 我已搜索现有 [Issues](https://github.com/PHMbench/PHM-Vibench/issues) 和 Discussions。
- [ ] 我已阅读[中文贡献指南](../../CONTRIBUTING_CN.md)和[已知限制](../../KNOWN_LIMITATIONS.md)。
- [ ] 我已考虑更简单的 config、文档或现有 extension point 方案。

## 使用场景

谁需要该能力？用于什么工作流？涉及哪些 data、model 或 task？

## 当前限制

说明当前行为，以及为什么现有 config、factory、component 或仓库外工具不能充分解决问题。

## 建议行为

描述最小可用行为和可测量的验收标准。

## 已考虑的替代方案

包括更简单的 workaround、仓库外方案或更窄范围。

## 架构与兼容性

```text
受影响的入口/pipeline：
受影响的 factory 或 registry：
新增/修改的 config key：
输入/输出或 batch 契约：
向后兼容影响：
应拒绝的非法组合：
所需依赖/硬件/数据：
```

说明为什么该功能应由 PHM-Vibench 解决，以及它是否改变 release-supported surface。

## 测试与证据计划

列出需要的聚焦 unit/contract test、config inspection、smoke command、负向案例和文档。
Synthetic data 只能验证软件路径，不能证明科学性能。

## 维护成本与风险

说明持续维护成本、可选依赖、数据/License 问题、安全影响、迁移需求和仍不支持的内容。

## 参考资料

可附 primary paper、标准、数据集、License 或已有实现。论文存在不等于仓库实现已经工作。
