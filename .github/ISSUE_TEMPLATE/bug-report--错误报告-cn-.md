---
name: Bug Report（错误报告）
about: 报告可复现的 PHM-Vibench 缺陷
title: "[BUG] "
labels: bug
assignees: ''
---

## 提交前检查

- [ ] 我已搜索现有 Issue 和 Pull Request。
- [ ] 我已在最新 `main` 上复现，或明确记录受影响的 release/commit。
- [ ] 我已阅读 [CONTRIBUTING_CN.md](https://github.com/PHMbench/PHM-Vibench/blob/main/CONTRIBUTING_CN.md)。
- [ ] 本报告不包含凭据、私人数据或尚未披露的安全漏洞。

安全敏感问题请停止公开提交，并遵循 [SECURITY.md](https://github.com/PHMbench/PHM-Vibench/blob/main/SECURITY.md)。

## 问题描述

说明故障和影响。请区分代码缺陷、缺少可选依赖、外部数据不可用、不受支持的组合和文档问题。

## 最小复现

**仓库 commit 或 tag：**

```text
<git rev-parse HEAD>
```

**配置：**

```text
<configs/ 下的路径，或附最小 YAML>
```

**命令：**

```bash
python main.py --config <yaml> [--override key=value ...]
```

**步骤：**

1.
2.
3.

## 预期行为

原本应发生什么？

## 实际行为

实际发生了什么？请包含命令退出码。

```text
<以文本形式粘贴完整 traceback 或日志>
```

## 环境

```text
操作系统：
Python 版本：
PyTorch 版本：
PyTorch Lightning 版本：
CPU/GPU：
CUDA 版本（如适用）：
安装方式：
```

辅助命令：

```bash
python --version
python -m pip freeze
```

环境输出较长时请作为文件附件。尽量删除秘密信息和私人路径。

## 数据与产物

```text
数据来源：仓库 Dummy 数据 | 外部数据
Metadata 文件：
相关输入 shape：
输出或 checkpoint 路径：
```

只有 License 允许时才能上传数据或模型产物。优先提供合法的小 fixture 或 synthetic 复现。

## 补充信息

列出尝试过的 workaround、相关 Issue、怀疑的文件或最后正常工作的 commit。若没有相同命令、数据、seed 和环境证据，不要将问题描述为性能回退。
